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
#include "atb/atb_infer.h"
#include "atb_speed/log.h"
#include "nlohmann/json.hpp"

#include "layers/parallel_layer_v2.h"
#include "models/starcoder/layer/flash_attention_layer.h"

#include "flash_attention_model.h"

namespace atb_speed {
namespace star_coder {
const int WEIGHT_COUNT_PER_LAYER = 12;
const int INPUT_TENSOR_COUNT_BEFORE_KEY = 2;
const int OUTPUT_TENSOR_COUNT_BEFORE_KEY = 1;
const int BEFORE_LAYER_WEIGHT_COUNT = 2;
const int OPERATION_COUNT_BEFORE_LAYER = 3;
const int INTERMEDIATETENSOR_COUNT_BEFORE_LAYER = 3;
const int OPERATION_COUNT_AFTER_LAYER = 3;
const int FINALNORMNODE_WEIGHT_COUNT = 2;
const int OUT_LM_HEAD_WEIGHT_COUNT = 1;
const int LAYER_NORM_AXIS_COUNT = 2;

enum InTensorId : int {
    IN_TENSOR_INPUT_IDS = 0,
    IN_TENSOR_POSITION_IDS,
    IN_TENSOR_ATTENTIONMASK,
    IN_TENSOR_PAST_KEY,
    IN_TENSOR_PAST_VALUE,
    IN_TENSOR_TOKENOFFSET,
    IN_TENSOR_SEQLEN,
    IN_HOLDER,
    IN_TENSOR_MAX, // 8
};

enum OutTensorId : int {
    OUT_TENSOR_HIDDENSTATES = 0,
    OUT_TENSOR_MAX,
};

void FlashAttentionModel::Param::FromString(const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);
    layerNormEps = paramJson["layerNormEps"].get<double>();
    headNum = paramJson["headNum"].get<int>();
    dk = paramJson["dk"].get<int>();
    layerNum = paramJson["layerNum"].get<int>();
    rank = paramJson["rank"].get<int>();
    rankSize = paramJson["rankSize"].get<int>();
    isEncoder = paramJson["isEncoder"].get<bool>();
    ATB_LOG(INFO) << "StarCoderModel param layerNormEps:" << layerNormEps << ", headNum:" << headNum << ", dk:" <<
        dk << ", layerNum:" << layerNum << ", rank:" << rank << ", rankSize:" << rankSize;
}

FlashAttentionModel::FlashAttentionModel(const std::string &param) : Model("FlashAttentionModel", param)
{
    param_.FromString(param);
}

FlashAttentionModel::~FlashAttentionModel() = default;

uint32_t FlashAttentionModel::GetInputNum() const
{
    return graph_.inTensors.size();
}

uint32_t FlashAttentionModel::GetOutputNum() const
{
    return graph_.outTensors.size();
}

atb::Status FlashAttentionModel::InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
    std::vector<atb::TensorDesc> &outTensorDescs)
{
    ATB_LOG(INFO) << "Enter FlashAttentionModel InferShape";
    if (outTensorDescs.size() != GetOutputNum()) {
        return atb::ERROR_INVALID_GRAPH;
    }
    const int64_t outDim = graph_.weightTensors.at(graph_.weightTensors.size() - 1).desc.shape.dims[0];
    outTensorDescs.at(0) = graph_.weightTensors.at(0).desc;
    outTensorDescs.at(0).shape.dimNum = 3;
    outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
    outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1];
    outTensorDescs.at(0).shape.dims[2] = outDim;

    ATB_LOG(INFO) << "FlashAttentionModel InferShape Success";
    return atb::NO_ERROR;
}

int64_t FlashAttentionModel::BuildGraph()
{
    ATB_LOG(INFO) << "Enter FlashAttentionModel BuildGraph";
    const int weightTensorSize = BEFORE_LAYER_WEIGHT_COUNT + WEIGHT_COUNT_PER_LAYER * param_.layerNum +
                                 FINALNORMNODE_WEIGHT_COUNT + OUT_LM_HEAD_WEIGHT_COUNT;
    graph_.weightTensors.resize(weightTensorSize);

    graph_.inTensors.resize(IN_TENSOR_MAX + param_.layerNum);
    graph_.outTensors.resize(OUT_TENSOR_MAX);

    const int nodeSize = param_.layerNum + OPERATION_COUNT_BEFORE_LAYER + OPERATION_COUNT_AFTER_LAYER;
    graph_.nodes.resize(nodeSize);

    const uint32_t internalTensorSize = graph_.nodes.size() - 1;
    graph_.internalTensors.resize(internalTensorSize);

    int nodeId = 0;
    atb::Operation *op = nullptr;

    auto &wtEmbeddingNode = graph_.nodes.at(nodeId++);
    atb::infer::GatherParam wtEmbeddingParam;
    atb::CreateOperation(wtEmbeddingParam, &op);
    wtEmbeddingNode.operation.reset(op);
    wtEmbeddingNode.inTensors = {&graph_.weightTensors.at(0), &graph_.inTensors.at(0)};
    wtEmbeddingNode.outTensors = {&graph_.internalTensors.at(0)};

    auto &wpEmbeddingNode = graph_.nodes.at(nodeId++);
    atb::infer::GatherParam wpEmbeddingParam;
    atb::CreateOperation(wpEmbeddingParam, &op);
    wpEmbeddingNode.operation.reset(op);
    wpEmbeddingNode.inTensors = {&graph_.weightTensors.at(1), &graph_.inTensors.at(1)};
    wpEmbeddingNode.outTensors = {&graph_.internalTensors.at(1)};

    auto &addNode = graph_.nodes.at(nodeId++);
    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CreateOperation(addParam, &op);
    addNode.operation.reset(op);
    addNode.inTensors = {&graph_.internalTensors.at(0), &graph_.internalTensors.at(1)};
    addNode.outTensors = {&graph_.internalTensors.at(2)};

    atb::Tensor *firstInTensor = &graph_.internalTensors.at(2);
    for (int layerId = 0; layerId < param_.layerNum; ++layerId) {
        auto &layerNode = graph_.nodes.at(nodeId++);

        atb_speed::star_coder::FlashAttentionLayerParam modelParam;
        modelParam.layerNormEps = param_.layerNormEps;
        modelParam.headNum = param_.headNum;
        modelParam.dk = param_.dk;
        modelParam.model = "star_coder";
        modelParam.rank = param_.rank;
        modelParam.rankSize = param_.rankSize;
        modelParam.isEncoder = param_.isEncoder;

        atb_speed::star_coder::FlashAttentionLayer(modelParam, &op);
        layerNode.operation.reset(op);
        layerNode.inTensors.resize(layerNode.operation->GetInputNum());

        size_t inTensorId = 0;
        layerNode.inTensors.at(inTensorId++) = firstInTensor; // hidden states
        for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
            layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(
                layerId * WEIGHT_COUNT_PER_LAYER + weightTensorId + BEFORE_LAYER_WEIGHT_COUNT);
        }
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ATTENTIONMASK);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PAST_KEY);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PAST_VALUE);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_TOKENOFFSET);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SEQLEN); // seqLen
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_HOLDER); // holder
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_MAX + layerId);
        
        layerNode.outTensors = {&graph_.internalTensors.at(INTERMEDIATETENSOR_COUNT_BEFORE_LAYER + layerId)};

        firstInTensor = layerNode.outTensors.at(0);
        ATB_LOG(INFO) << "LLaMAFlashAttentionModel BuildGraph success";
    }

    auto &finalNormNode = graph_.nodes.at(nodeId++);
    atb::infer::LayerNormParam finalNormParam;
    finalNormParam.layerType = atb::infer::LayerNormParam::LayerNormType::LAYER_NORM_NORM;
    finalNormParam.normParam.epsilon = param_.layerNormEps;
    finalNormParam.normParam.beginNormAxis = LAYER_NORM_AXIS_COUNT;
    finalNormParam.normParam.beginParamsAxis = LAYER_NORM_AXIS_COUNT;
    atb::CreateOperation(finalNormParam, &op);
    finalNormNode.operation.reset(op);
    const int finalLayerNormWeightTensorId =
        graph_.weightTensors.size() - FINALNORMNODE_WEIGHT_COUNT - OUT_LM_HEAD_WEIGHT_COUNT;
    const int finalLayerNormBiasTensorId =
        graph_.weightTensors.size() - (FINALNORMNODE_WEIGHT_COUNT - 1) - OUT_LM_HEAD_WEIGHT_COUNT;
    const int finalLayerNormOutTensorId = internalTensorSize - 2;
    finalNormNode.inTensors = {firstInTensor, &graph_.weightTensors.at(finalLayerNormWeightTensorId),
                               &graph_.weightTensors.at(finalLayerNormBiasTensorId)};
    finalNormNode.outTensors = {&graph_.internalTensors.at(finalLayerNormOutTensorId)};

    const int hiddenSize = param_.headNum * param_.dk;
    auto &qPassSliceNode = graph_.nodes.at(nodeId++);
    atb::infer::SliceParam slicePassParam;
    slicePassParam.offsets = {0, 0, hiddenSize * param_.rank};
    slicePassParam.size = {-1, -1, hiddenSize};
    CreateOperation(slicePassParam, &op);
    qPassSliceNode.operation.reset(op);
    const int qPassSliceNodeOutTensorId = internalTensorSize - 1;
    qPassSliceNode.inTensors = {&graph_.internalTensors.at(finalLayerNormOutTensorId)};
    qPassSliceNode.outTensors = {&graph_.internalTensors.at(qPassSliceNodeOutTensorId)};

    auto &outLinearNode = graph_.nodes.at(nodeId++);
    atb_speed::common::ParallelParamV2 outLinearParm;
    outLinearParm.commParam.rank = param_.rank;
    outLinearParm.commParam.rankSize = param_.rankSize;
    outLinearParm.isBias = false;
    atb_speed::common::RowParallelLinearV2(outLinearParm, &op);
    outLinearNode.operation.reset(op);
    const int finalLinearWeightTensorId = graph_.weightTensors.size() - OUT_LM_HEAD_WEIGHT_COUNT;
    outLinearNode.inTensors = {&graph_.internalTensors.at(qPassSliceNodeOutTensorId),
                               &graph_.weightTensors.at(finalLinearWeightTensorId),
                               &graph_.inTensors.at(IN_HOLDER), &graph_.inTensors.at(IN_HOLDER),
                               &graph_.inTensors.at(IN_HOLDER), &graph_.inTensors.at(IN_HOLDER),
                               &graph_.inTensors.at(IN_HOLDER)};
    outLinearNode.outTensors = {&graph_.outTensors.at(0)};
<<<<<<< HEAD
    return atb::NO_ERROR;
=======
>>>>>>> fbec655d8f922ef0965245b36ddbeeba216a86fb
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

    return atb::NO_ERROR;
}

atb::Status FlashAttentionModel::BindParamHostTensor(uint32_t nodeId)
{
    ATB_LOG(INFO) << "BindParamHostTensor";
    if (nodeId < OPERATION_COUNT_BEFORE_LAYER || nodeId >= OPERATION_COUNT_BEFORE_LAYER + param_.layerNum) {
        return atb::NO_ERROR;
    }
    auto &node = graph_.nodes.at(nodeId);

    const uint32_t tokenOffsetTensorId = 16;
    const uint32_t seqLenTensorId = 17;

    node.variantPack.inTensors.at(tokenOffsetTensorId).hostData = tokenOffset_.data();
    node.variantPack.inTensors.at(seqLenTensorId).hostData = seqLen_.data();
    ATB_LOG(INFO) << "BindParamHostTensor end";
    return atb::NO_ERROR;
}
} // namespace star_coder
} // namespace atb_speed