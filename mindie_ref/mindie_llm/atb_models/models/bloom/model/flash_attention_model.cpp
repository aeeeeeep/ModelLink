/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "flash_attention_model.h"

#include <algorithm>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtype-limits"
#include <nlohmann/json.hpp>
#pragma GCC diagnostic pop
#include <atb/atb_infer.h>

#include "atb_speed/log.h"
#include "layers/parallel_layer.h"
#include "models/bloom/layer/flash_attention_layer.h"
#include "atb_speed/utils/model_factory.h"

namespace atb_speed {
namespace bloom_7b {

REGISTER_MODEL(bloom_7b, FlashAttentionModel);

const int WEIGHT_COUNT_PER_LAYER = 16;
const int EMBEDDING_WEIGHT_COUNT = 1;
const int EMBEDDING_WEIGHT_NORM_COUNT = 2;
const int FINAL_LINEAR_WEIGHT_COUNT = 1;
const int FINAL_NORM_WEIGHT_COUNT = 2;
const int IN_TENSOR_NUM = 7;
const int OUT_TENSOR_NUM = 4;         // hidden_state and lm_logits
const int EXT_INTERNAL_TENSORS = 4;

enum InTensorId : int {
    IN_INPUT_IDS = 0,
    IN_ATTENTION_MASK,
    IN_CACHED_K,
    IN_CACHED_V,
    IN_TOKENOFFSET,
    IN_SEQLEN,
    IN_PLACE_HOLDER,
    IN_TENSOR_MAX
    };

void FlashAttentionModel::Param::FromString(const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);
    if (paramJson.contains("layerNormEps")) {
        layerNormEps = paramJson["layerNormEps"].get<double>();
    }
    headNum = paramJson["headNum"].get<int>();
    dk = paramJson["dk"].get<int>();
    invNormFactorvarAttr = paramJson["invNormFactorvarAttr"].get<float>();

    if (paramJson.contains("rank")) {
        rank = paramJson["rank"].get<int>();
    }
    if (paramJson.contains("rankSize")) {
        rankSize = paramJson["rankSize"].get<int>();
    }
    if (paramJson.contains("layerNum")) {
        layerNum = paramJson["layerNum"].get<float>();
    }
    for (auto item : paramJson["floatLayers"]) {
        floatLayers.push_back(item.get<int>());
    }
    for (auto item : paramJson["qkvInputScale"]) {
        qkvInputScale.push_back(item.get<float>());
    }
    for (auto item : paramJson["qkvInputOffset"]) {
        qkvInputOffset.push_back(item.get<int>());
    }
    for (auto item : paramJson["denseInputScale"]) {
        denseInputScale.push_back(item.get<float>());
    }
    for (auto item : paramJson["denseInputOffset"]) {
        denseInputOffset.push_back(item.get<int>());
    }
    for (auto item : paramJson["selfLnInputScale"]) {
        selfLnInputScale.push_back(item.get<float>());
    }
    for (auto item : paramJson["selfLnInputOffset"]) {
        selfLnInputOffset.push_back(item.get<int>());
    }
    for (auto item : paramJson["ffnOutInputScale"]) {
        ffnOutInputScale.push_back(item.get<float>());
    }
    for (auto item : paramJson["ffnOutInputOffset"]) {
        ffnOutInputOffset.push_back(item.get<int>());
    }
}

FlashAttentionModel::FlashAttentionModel(const std::string &param)
    : Model("Bloom7BFlashAttentionModel", param)
{
    param_.FromString(param);
}

FlashAttentionModel::~FlashAttentionModel() {}

uint32_t FlashAttentionModel::GetInputNum() const
{
    return graph_.inTensors.size();
}

uint32_t FlashAttentionModel::GetOutputNum() const
{
    return graph_.outTensors.size();
}

atb::Status FlashAttentionModel::InferShape(
    const std::vector<atb::TensorDesc> &inTensorDescs, std::vector<atb::TensorDesc> &outTensorDescs)
{
    if (outTensorDescs.size() != GetOutputNum()) {
        return atb::ERROR_INVALID_GRAPH;
    }

    size_t vocabSize = graph_.weightTensors.back().desc.shape.dims[0];
    size_t hiddenSize = graph_.weightTensors.back().desc.shape.dims[1] * param_.rankSize;

    const atb::TensorDesc &inputIds = inTensorDescs.at(IN_INPUT_IDS);

    size_t dimAll = 3;
    outTensorDescs.at(0) = inputIds; // [batch, seq_len, headNum * 3 * dk]
    outTensorDescs.at(0).shape.dimNum = dimAll;
    outTensorDescs.at(0).shape.dims[dimAll - 1] = hiddenSize;
    outTensorDescs.at(0).dtype = inTensorDescs.at(IN_ATTENTION_MASK).dtype;

    size_t dim = 0;
    outTensorDescs.at(1) = inputIds;
    outTensorDescs.at(1).shape.dimNum = dimAll;
    outTensorDescs.at(1).shape.dims[dim++] = inputIds.shape.dims[0]; // batch_size
    outTensorDescs.at(1).shape.dims[dim++] = 1; // 长度seq_len
    outTensorDescs.at(1).shape.dims[dim++] = vocabSize;                // 长度vocab_size
    outTensorDescs.at(1).dtype = inTensorDescs.at(IN_ATTENTION_MASK).dtype;

    for (uint i = 2; i < GetOutputNum(); i++) {
        outTensorDescs.at(i) = inputIds;
        outTensorDescs.at(i).shape.dimNum = dimAll;
        outTensorDescs.at(i).shape.dims[dimAll - 1] = hiddenSize;
        outTensorDescs.at(i).dtype = inTensorDescs.at(IN_ATTENTION_MASK).dtype;
    }

    return atb::NO_ERROR;
}

int64_t FlashAttentionModel::BuildGraph()
{
    ATB_LOG(INFO) << "Build Graph Start.";

    const int weightTensorSize = (
        EMBEDDING_WEIGHT_COUNT + EMBEDDING_WEIGHT_NORM_COUNT +
        WEIGHT_COUNT_PER_LAYER * param_.layerNum + FINAL_LINEAR_WEIGHT_COUNT + FINAL_NORM_WEIGHT_COUNT
        );
    graph_.weightTensors.resize(weightTensorSize);

    graph_.inTensors.resize(IN_TENSOR_NUM + param_.layerNum);
    graph_.outTensors.resize(OUT_TENSOR_NUM);

    const int nodeSize = param_.layerNum + 6;
    graph_.nodes.resize(nodeSize);

    graph_.internalTensors.resize(param_.layerNum + EXT_INTERNAL_TENSORS);

    int nodeId = 0;
    int weightOffset = 0;
    size_t internalTensorCnt = 0;

    auto &wordEmbeddingNode = graph_.nodes.at(nodeId++);
    atb::infer::GatherParam wordEmbeddingParam;
    atb::Operation *op = nullptr;
    CREATE_OPERATION(wordEmbeddingParam, &op);
    wordEmbeddingNode.operation.reset(op);
    wordEmbeddingNode.inTensors = {&graph_.weightTensors.at(weightOffset++), &graph_.inTensors.at(0)};
    wordEmbeddingNode.outTensors = {&graph_.internalTensors.at(internalTensorCnt)};
    
    auto &firstNormNode = graph_.nodes.at(nodeId++);
    atb::infer::LayerNormParam firstNormParam;
    firstNormParam.layerType = atb::infer::LayerNormParam::LAYER_NORM_NORM;
    const int32_t beginParamsAxis = 2;
    firstNormParam.normParam.epsilon = param_.layerNormEps;
    firstNormParam.normParam.beginNormAxis = beginParamsAxis;
    firstNormParam.normParam.beginParamsAxis = 1;
    CREATE_OPERATION(firstNormParam, &op);
    firstNormNode.operation.reset(op);
    firstNormNode.inTensors = {&graph_.internalTensors.at(internalTensorCnt++),
                               &graph_.weightTensors.at(weightOffset++), &graph_.weightTensors.at(weightOffset++)};
    firstNormNode.outTensors = {&graph_.internalTensors.at(internalTensorCnt)};

    atb::Tensor *firstInTensor = &graph_.internalTensors.at(internalTensorCnt++);

    ATB_LOG(INFO) << "First InTensor Set.";

    for (int layerId = 0; layerId < param_.layerNum; ++layerId) {
        auto &layerNode = graph_.nodes.at(nodeId++);
        bool isFloatLayer = false;
        if (std::find(param_.floatLayers.begin(), param_.floatLayers.end(), layerId) != param_.floatLayers.end()) {
            isFloatLayer = true;
        }

        atb_speed::bloom_7b::Bloom7bCommonLayerParam opParam;
        opParam.layerNormEps = param_.layerNormEps;
        opParam.headNum = param_.headNum;
        opParam.dk = param_.dk;
        opParam.rank = param_.rank;
        opParam.rankSize = param_.rankSize;
        opParam.invNormFactorvarAttr = param_.invNormFactorvarAttr;
        opParam.qkvInputScale = param_.qkvInputScale[layerId];
        opParam.qkvInputOffset = param_.qkvInputOffset[layerId];
        opParam.denseInputScale = param_.denseInputScale[layerId];
        opParam.denseInputOffset = param_.denseInputOffset[layerId];
        opParam.selfLnInputScale = param_.selfLnInputScale[layerId];
        opParam.selfLnInputOffset = param_.selfLnInputOffset[layerId];
        opParam.ffnOutInputScale = param_.ffnOutInputScale[layerId];
        opParam.ffnOutInputOffset = param_.ffnOutInputOffset[layerId];
        if (isFloatLayer) {
            opParam.quantmodel = false;
        }
        atb_speed::bloom_7b::CommomLayer(opParam, &op);
        layerNode.operation.reset(op);
        layerNode.inTensors.resize(layerNode.operation->GetInputNum());
        ATB_LOG(INFO) << "layerNode Set." << layerId;
        size_t inTensorId = 0;

        size_t weightCount = WEIGHT_COUNT_PER_LAYER ;
        for (size_t weightTensorId = 0; weightTensorId < weightCount; ++weightTensorId) {
            layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(weightOffset++);
        }
        ATB_LOG(INFO) << "weightTensors Set." << layerId;

        for (int i = 0; i < IN_TENSOR_NUM + 1; i++) {
            if (i == 0) {
                layerNode.inTensors.at(inTensorId++) = firstInTensor; // IN_HIDDENSTATES
            } else if (i == IN_TENSOR_NUM) {
                layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_MAX + layerId); // IN_LAYER_ID
            } else {
                layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(i);
            }
        }

        ATB_LOG(INFO) << "inTensors Set." << layerId;
        if (layerId != param_.layerNum - 1) {
            layerNode.outTensors = {&graph_.internalTensors.at(internalTensorCnt++),
                                    &graph_.outTensors.at(2),
                                    &graph_.outTensors.at(3)};
        } else {
            layerNode.outTensors = {&graph_.outTensors.at(0), &graph_.outTensors.at(2), &graph_.outTensors.at(3)};
        }
        firstInTensor = layerNode.outTensors.at(0);
    }

    auto &finalNormNode = graph_.nodes.at(nodeId++);
    atb::infer::LayerNormParam finalNormParam;
    finalNormParam.layerType = atb::infer::LayerNormParam::LAYER_NORM_NORM;
    finalNormParam.normParam.epsilon = param_.layerNormEps;
    finalNormParam.normParam.beginNormAxis = beginParamsAxis;
    finalNormParam.normParam.beginParamsAxis = 1;
    CREATE_OPERATION(finalNormParam, &op);
    finalNormNode.operation.reset(op);

    const int finalLayerNormWeightTensorId =
        graph_.weightTensors.size() - FINAL_LINEAR_WEIGHT_COUNT - FINAL_NORM_WEIGHT_COUNT;
    finalNormNode.inTensors = {firstInTensor, &graph_.weightTensors.at(finalLayerNormWeightTensorId),
                               &graph_.weightTensors.at(finalLayerNormWeightTensorId + 1)};
    finalNormNode.outTensors = {&graph_.internalTensors.at(internalTensorCnt)};

    // only keep the last token
    auto &sliceNode = graph_.nodes.at(nodeId++);
    atb::infer::SliceParam sliceParam;
    sliceParam.offsets = {0, -1, 0};
    sliceParam.size = {-1, 1, -1};
    CREATE_OPERATION(sliceParam, &op);
    sliceNode.operation.reset(op);
    sliceNode.inTensors = {&graph_.internalTensors.at(internalTensorCnt++)};
    sliceNode.outTensors = {&graph_.internalTensors.at(internalTensorCnt)};

    const int hiddenSize = param_.headNum * param_.dk;
    auto &sliceNode2 = graph_.nodes.at(nodeId++);
    atb::infer::SliceParam sliceParam2;
    sliceParam2.offsets = {0, 0, hiddenSize * param_.rank};
    sliceParam2.size = {-1, -1, hiddenSize};
    CREATE_OPERATION(sliceParam2, &op);
    sliceNode2.operation.reset(op);
    sliceNode2.inTensors = {&graph_.internalTensors.at(internalTensorCnt++)};
    sliceNode2.outTensors = {&graph_.internalTensors.at(internalTensorCnt)};

    auto &finalLinearNode = graph_.nodes.at(nodeId++);
    atb_speed::common::ParallelParam finalLinearParam;
    finalLinearParam.rank = param_.rank;
    finalLinearParam.rankSize = param_.rankSize;
    finalLinearParam.isBias = false;
    atb_speed::common::RowParallelLinear(finalLinearParam, &op);
    finalLinearNode.operation.reset(op);
    finalLinearNode.inTensors.resize(finalLinearNode.operation->GetInputNum());
    const int finalLinearNodeWeightTensorId = graph_.weightTensors.size() - FINAL_LINEAR_WEIGHT_COUNT;
    finalLinearNode.inTensors = {&graph_.internalTensors.at(internalTensorCnt++),
                                 &graph_.weightTensors.at(finalLinearNodeWeightTensorId)};
    finalLinearNode.outTensors = {&graph_.outTensors.at(1)};
    ATB_LOG(INFO) << "Build Graph finished.";

    return atb::NO_ERROR;
}

atb::Status FlashAttentionModel::ParseParam(const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);
    tokenOffset_.clear();
    for (auto item : paramJson["tokenOffset"]) {
        tokenOffset_.push_back(item.get<int>());
    }
 
    seqLen_.clear();
    for (auto item : paramJson["seqLen"]) {
        seqLen_.push_back(item.get<int>());
    }
 
    ATB_LOG(INFO) << "FlashAttentionModel ParseParam tokenOffset set";
 
    return atb::NO_ERROR;
}
 
atb::Status FlashAttentionModel::BindParamHostTensor(uint32_t nodeId)
{
    size_t nodeBeforeLayers = 2;
    if (nodeId < nodeBeforeLayers || nodeId >= param_.layerNum + nodeBeforeLayers) {
        return atb::NO_ERROR;
    }

    const uint32_t InttokenOffsetTensorId = 20;
    const uint32_t IntseqLenTensorId = 21;

    auto &node = graph_.nodes.at(nodeId);
    node.variantPack.inTensors.at(InttokenOffsetTensorId).hostData = tokenOffset_.data();
    node.variantPack.inTensors.at(IntseqLenTensorId).hostData = seqLen_.data();
 
    return atb::NO_ERROR;
}

}  // namespace bloom_7b
}  // namespace atb_speed