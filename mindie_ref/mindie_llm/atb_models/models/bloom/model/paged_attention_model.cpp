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
#include "paged_attention_model.h"
#include "atb_speed/log.h"
#include "layers/parallel_layer.h"
#include "layers/operations/word_embedding.h"
#include "layers/operations/lmhead.h"
#include "models/bloom/layer/paged_attention_layer.h"


namespace atb_speed {
namespace bloom_7b {
const int WEIGHT_COUNT_PER_LAYER = 28;
const int EMBEDDING_WEIGHT_COUNT = 1;
const int EMBEDDING_WEIGHT_NORM_COUNT = 2;
const int FINAL_LINEAR_WEIGHT_COUNT = 1;
const int FINAL_NORM_WEIGHT_COUNT = 2;
const int EXT_INTERNAL_TENSORS = 2;


enum InTensorId : int {
    IN_INPUT_IDS = 0,
    IN_ATTENTION_MASK,
    IN_BLOCK_TABLES,
    IN_SLOTS,
    IN_INPUT_LENGTHS,
    IN_PLACE_HOLDER,
    IN_TENSOR_LOGTIS_INDICES,
    IN_TENSOR_MAX
    };

enum OutTensorId : int {
    OUT_HIDDENSTATES = 0,
    OUT_LOGITS,
    OUT_TENSOR_MAX
    };

void PagedAttentionModel::Param::FromString(const std::string &param)
{
    ATB_LOG(INFO) << "parse param begin.";
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
    if (paramJson.contains("backend")) {
        backend = paramJson["backend"].get<std::string>();
    }
    
    if (paramJson.contains("rankSize")) {
        rankSize = paramJson["rankSize"].get<int>();
    }
    if (paramJson.contains("layerNum")) {
        layerNum = paramJson["layerNum"].get<float>();
    }
    if (paramJson.contains("quantMode")) {
        quantMode = paramJson["quantMode"].get<int>();
    }
    if (paramJson.contains("isPrefill")) {
        isPrefill = paramJson["isPrefill"].get<bool>();
    }
    for (auto item : paramJson["floatLayers"]) {
        floatLayers.push_back(item.get<int>());
    }
    ATB_LOG(INFO) << "parse param end.";
}

PagedAttentionModel::PagedAttentionModel(const std::string &param)
    : Model("Bloom7BPagedAttentionModel", param)
{
    ATB_LOG(INFO) << "init model start.";
    param_.FromString(param);
}

PagedAttentionModel::~PagedAttentionModel() {}

uint32_t PagedAttentionModel::GetInputNum() const
{
    return graph_.inTensors.size();
}

uint32_t PagedAttentionModel::GetOutputNum() const
{
    return graph_.outTensors.size();
}

atb::Status PagedAttentionModel::InferShape(
    const std::vector<atb::TensorDesc> &inTensorDescs, std::vector<atb::TensorDesc> &outTensorDescs)
{
    if (outTensorDescs.size() != GetOutputNum()) {
        return atb::ERROR_INVALID_GRAPH;
    }

    size_t vocabSize = graph_.weightTensors.back().desc.shape.dims[0];
    size_t hiddenSize = graph_.weightTensors.back().desc.shape.dims[1] * param_.rankSize;

    const atb::TensorDesc &inputIds = inTensorDescs.at(IN_INPUT_IDS);  // [btokens, hiddenSize]
    size_t dimAll = inputIds.shape.dimNum + 1;
    
    outTensorDescs.at(0) = inputIds;
    outTensorDescs.at(0).shape.dimNum = dimAll;
    outTensorDescs.at(0).shape.dims[dimAll - 1] = hiddenSize;
    outTensorDescs.at(0).dtype = inTensorDescs.at(IN_ATTENTION_MASK).dtype;

    outTensorDescs.at(1) = outTensorDescs.at(0);
    outTensorDescs.at(1).shape.dims[dimAll - 1] = vocabSize;           // 长度vocab_size

    if (param_.isPrefill) {
        outTensorDescs.at(1).shape.dims[0] = inTensorDescs.at(IN_TENSOR_LOGTIS_INDICES).shape.dims[0];
    }
    return atb::NO_ERROR;
}

int64_t PagedAttentionModel::BuildGraph()
{
    ATB_LOG(INFO) << "Build Graph Start.";

    const int weightTensorSize = (
        EMBEDDING_WEIGHT_COUNT + EMBEDDING_WEIGHT_NORM_COUNT +
        WEIGHT_COUNT_PER_LAYER * param_.layerNum + FINAL_LINEAR_WEIGHT_COUNT + FINAL_NORM_WEIGHT_COUNT
        );
    graph_.weightTensors.resize(weightTensorSize);

    graph_.kCacheTensors.resize(param_.layerNum);
    graph_.vCacheTensors.resize(param_.layerNum);

    graph_.inTensors.resize(IN_TENSOR_MAX);
    graph_.outTensors.resize(OUT_TENSOR_MAX);

    const int nodeSize = param_.layerNum + 4;
    graph_.nodes.resize(nodeSize);
    graph_.internalTensors.resize(param_.layerNum + EXT_INTERNAL_TENSORS);

    int nodeId = 0;
    int weightOffset = 0;
    size_t internalTensorCnt = 0;
    atb::Operation *op = nullptr;

    auto &wordEmbeddingNode = graph_.nodes.at(nodeId++);
    atb_speed::common::WordEmbeddingParam wordEmbeddingParam;

    wordEmbeddingParam.unpadInputs = true;
    wordEmbeddingParam.tensorParallelInfo.rank= param_.rank;
    wordEmbeddingParam.tensorParallelInfo.worldSize= param_.rankSize;
    wordEmbeddingParam.tensorParallelInfo.backend= param_.backend;
    atb_speed::common::WordEmbedding(wordEmbeddingParam, &op);
    wordEmbeddingNode.operation.reset(op);
    wordEmbeddingNode.inTensors = {&graph_.weightTensors.at(weightOffset++), &graph_.inTensors.at(0)};
    wordEmbeddingNode.outTensors = {&graph_.internalTensors.at(internalTensorCnt)};

    auto &firstNormNode = graph_.nodes.at(nodeId++);
    atb::infer::LayerNormParam firstNormParam;
    firstNormParam.layerType = atb::infer::LayerNormParam::LAYER_NORM_NORM;
    const int32_t beginParamsAxis = 1;
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

        atb_speed::bloom_7b::Bloom7bPagedLayerParam opParam;
        opParam.layerNormEps = param_.layerNormEps;
        opParam.headNum = param_.headNum;
        opParam.dk = param_.dk;
        opParam.rank = param_.rank;
        opParam.backend = param_.backend;
        opParam.rankSize = param_.rankSize;
        opParam.isPrefill = param_.isPrefill;
        opParam.invNormFactorvarAttr = param_.invNormFactorvarAttr;
        if (isFloatLayer) {
            opParam.quantMode = 0;                              // 0:not quant
        } else {
            opParam.quantMode = param_.quantMode == 2 ? 2 : 1;  // 1:w8a8, 2:w8a16
        }

        atb_speed::bloom_7b::PagedLayer(opParam, &op);
        layerNode.operation.reset(op);
        layerNode.inTensors.resize(layerNode.operation->GetInputNum());
        ATB_LOG(INFO) << "layerNode Set." << layerId;
        size_t inTensorId = 0;

        size_t weightCount = WEIGHT_COUNT_PER_LAYER ;
        for (size_t weightTensorId = 0; weightTensorId < weightCount; ++weightTensorId) {
            layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(weightOffset++);
        }
        ATB_LOG(INFO) << "weightTensors Set." << layerId;

        for (int i = 0; i < IN_TENSOR_MAX + 1; i++) {
            if (i == 0) {
                layerNode.inTensors.at(inTensorId++) = firstInTensor; // IN_HIDDENSTATES
            } else if (i == IN_TENSOR_MAX - 1) {
                layerNode.inTensors.at(inTensorId++) = &graph_.kCacheTensors.at(layerId);
            } else if (i == IN_TENSOR_MAX) {
                layerNode.inTensors.at(inTensorId++) = &graph_.vCacheTensors.at(layerId);
            } else {
                layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(i);
            }
        }

        ATB_LOG(INFO) << "inTensors Set." << layerId;
        if (layerId != param_.layerNum - 1) {
            layerNode.outTensors = {&graph_.internalTensors.at(internalTensorCnt++)};
        } else {
            layerNode.outTensors = {&graph_.outTensors.at(0)};
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

    auto &finalLinearNode = graph_.nodes.at(nodeId++);
    atb_speed::common::LmHeadParam finalLinearParam;
    finalLinearParam.unpadInputs = true;
    finalLinearParam.gatherAhead = param_.isPrefill;
    finalLinearParam.hiddenSizePerAttentionHead = param_.headNum * param_.dk;
    finalLinearParam.linearParallelParam.unpadInputs = true;
    
    finalLinearParam.linearParallelParam.tensorParallelInfo.rank = param_.rank;
    finalLinearParam.linearParallelParam.tensorParallelInfo.worldSize = param_.rankSize;
    finalLinearParam.linearParallelParam.tensorParallelInfo.backend = param_.backend;

    finalLinearParam.linearParallelParam.parallelType = atb_speed::common::LinearParallelType::ROW_PARALLEL;
    LmHead(finalLinearParam, &op);
    finalLinearNode.operation.reset(op);
    const int finalLinearNodeWeightTensorId = graph_.weightTensors.size() - FINAL_LINEAR_WEIGHT_COUNT;
    if (param_.isPrefill) {
        finalLinearNode.inTensors = {&graph_.internalTensors.at(internalTensorCnt++),
                                    &graph_.weightTensors.at(finalLinearNodeWeightTensorId),
                                    &graph_.inTensors.at(IN_PLACE_HOLDER),
                                    &graph_.inTensors.at(IN_PLACE_HOLDER),
                                    &graph_.inTensors.at(IN_PLACE_HOLDER),
                                    &graph_.inTensors.at(IN_PLACE_HOLDER),
                                    &graph_.inTensors.at(IN_TENSOR_LOGTIS_INDICES)};
    } else {
        finalLinearNode.inTensors = {&graph_.internalTensors.at(internalTensorCnt++),
                                    &graph_.weightTensors.at(finalLinearNodeWeightTensorId),
                                    &graph_.inTensors.at(IN_PLACE_HOLDER),
                                    &graph_.inTensors.at(IN_PLACE_HOLDER),
                                    &graph_.inTensors.at(IN_PLACE_HOLDER),
                                    &graph_.inTensors.at(IN_PLACE_HOLDER),
                                    &graph_.inTensors.at(IN_PLACE_HOLDER)};
    }
    finalLinearNode.outTensors = {&graph_.outTensors.at(1)};
    ATB_LOG(INFO) << "Build Graph finished.";

    return atb::NO_ERROR;
}

atb::Status PagedAttentionModel::ParseParam(const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);
 
    seqLen_.clear();
    for (auto item : paramJson["seqLen"]) {
        seqLen_.push_back(item.get<int>());
    }
 
    ATB_LOG(INFO) << "PagedAttentionModel ParseParam tokenOffset set";
 
    return atb::NO_ERROR;
}
 
atb::Status PagedAttentionModel::BindParamHostTensor(uint32_t nodeId)
{
    size_t nodeBeforeLayers = 2;
    if (nodeId < nodeBeforeLayers || nodeId >= param_.layerNum + nodeBeforeLayers) {
        return atb::NO_ERROR;
    }

    const uint32_t IntseqLenTensorId = 32;

    auto &node = graph_.nodes.at(nodeId);
    node.variantPack.inTensors.at(IntseqLenTensorId).hostData = seqLen_.data();
 
    return atb::NO_ERROR;
}
}  // namespace bloom_7b
}  // namespace atb_speed