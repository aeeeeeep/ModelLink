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
#include "atb/atb_infer.h"
#include "atb_speed/log.h"
#include "models/falcon/7b//layers/flash_attention_layer.h"
#include "nlohmann/json.hpp"

namespace atb_speed {
namespace falcon_7b {
const int WEIGHT_COUNT_PER_LAYER = 6;
const int OPERATION_COUNT_BEFORE_LAYER = 3;
const int OPERATION_COUNT_AFTER_LAYER = 2;
const int WEIGHT_COUNT_BEFORE_LAYER = 1;
const int FINALNORMNODE_WEIGHT_COUNT = 2;
const int OUT_LM_HEAD_WEIGHT_COUNT = 1;
const int INTERMEDIATETENSOR_COUNT_BEFORE_LAYER = 3;

enum InTensorId : int {
    IN_TENSOR_INPUTIDS = 0,  // [bs, seq_len]
    IN_TENSOR_POSITIONID,    // [bs, seq_len]
    IN_TENSOR_COSTABLE,      // [max_seq_len, head_size]
    IN_TENSOR_SINTABLE,      // [max_seq_len, head_size]
    IN_TENSOR_ATTENTIONMASK, // [bs, max_seq_len, max_seq_len]
    IN_TENSOR_PAST_KEY,      // [layer_num, bs, max_seq_len, head_size]
    IN_TENSOR_PAST_VALUE,    // [layer_num, bs, max_seq_len, head_size]
    IN_TENSOR_TOKENOFFSET,   // [bs]
    IN_TENSOR_SEQLEN,        // [bs]
    IN_TENSOR_MAX,           // 9
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
    hiddenSize = paramJson["hiddenSize"].get<int>();
    kvHeadNum = paramJson["kvHeadNum"].get<int>();
    layerNum = paramJson["layerNum"].get<int>();
    preScale = paramJson["preScale"].get<float>();
    postScale = paramJson["postScale"].get<float>();

    if (paramJson.contains("rotaryCoeff")) {
        rotaryCoeff = paramJson["rotaryCoeff"].get<int>();
    }
}

FlashAttentionModel::FlashAttentionModel(const std::string &param) : Model(
    "FlashAttentionModel", param) { param_.FromString(param); }

FlashAttentionModel::~FlashAttentionModel() = default;

uint32_t FlashAttentionModel::GetInputNum() const { return graph_.inTensors.size(); }

uint32_t FlashAttentionModel::GetOutputNum() const { return graph_.outTensors.size(); }

atb::Status FlashAttentionModel::InferShape(
    const std::vector<atb::TensorDesc> &inTensorDescs, std::vector<atb::TensorDesc> &outTensorDescs)
{
    ATB_LOG(INFO) << "Enter FalconModel InferShape";
    if (outTensorDescs.size() != GetOutputNum()) {
        return atb::ERROR_INVALID_GRAPH;
    }

    const int64_t outDim = graph_.weightTensors.back().desc.shape.dims[0];
    outTensorDescs.at(0) = graph_.weightTensors.at(0).desc;
    outTensorDescs.at(0).shape.dimNum = 3;
    outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
    outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1];
    outTensorDescs.at(0).shape.dims[2] = outDim;
    return atb::NO_ERROR;
}

int64_t FlashAttentionModel::BuildGraph()
{
    ATB_LOG(INFO) << "Enter FalconModel BuildGraph";
    const int weightTensorSize = WEIGHT_COUNT_BEFORE_LAYER + WEIGHT_COUNT_PER_LAYER * param_.layerNum +
                                 FINALNORMNODE_WEIGHT_COUNT + OUT_LM_HEAD_WEIGHT_COUNT;
    graph_.weightTensors.resize(weightTensorSize);

    graph_.inTensors.resize(IN_TENSOR_MAX + param_.layerNum);
    graph_.outTensors.resize(OUT_TENSOR_MAX);

    const int nodeSize = OPERATION_COUNT_BEFORE_LAYER + param_.layerNum + OPERATION_COUNT_AFTER_LAYER;
    graph_.nodes.resize(nodeSize);

    const int internalTensorSize = graph_.nodes.size() - 1;
    graph_.internalTensors.resize(internalTensorSize);

    int nodeId = 0;
    int weightOffset = 0;
    int internalTensorCnt = 0;
    atb::Operation *op = nullptr;

    // wordEmbedding [bs, seq_len, hidden_size]
    auto &wordEmbeddingNode = graph_.nodes.at(nodeId++);
    atb::infer::GatherParam wordEmbeddingParam;
    CREATE_OPERATION(wordEmbeddingParam, &op);
    wordEmbeddingNode.operation.reset(op);
    wordEmbeddingNode.inTensors = {&graph_.weightTensors.at(weightOffset++), &graph_.inTensors.at(0)};
    wordEmbeddingNode.outTensors = {&graph_.internalTensors.at(internalTensorCnt++)};

    // cos_sin_table [bs, seq_len, head_size]
    auto &cosEmbeddingNode = graph_.nodes.at(nodeId++);
    atb::infer::GatherParam cosEmbeddingGatherParam;
    cosEmbeddingGatherParam.axis = param_.axis;
    CREATE_OPERATION(cosEmbeddingGatherParam, &op);
    cosEmbeddingNode.operation.reset(op);
    cosEmbeddingNode.inTensors = {&graph_.inTensors.at(IN_TENSOR_COSTABLE), &graph_.inTensors.at(IN_TENSOR_POSITIONID)};
    cosEmbeddingNode.outTensors = {&graph_.internalTensors.at(internalTensorCnt++)};

    auto &sinEmbeddingNode = graph_.nodes.at(nodeId++);
    atb::infer::GatherParam sinEmbeddingGatherParam;
    sinEmbeddingGatherParam.axis = param_.axis;
    CREATE_OPERATION(sinEmbeddingGatherParam, &op);
    sinEmbeddingNode.operation.reset(op);
    sinEmbeddingNode.inTensors = {&graph_.inTensors.at(IN_TENSOR_SINTABLE), &graph_.inTensors.at(IN_TENSOR_POSITIONID)};
    sinEmbeddingNode.outTensors = {&graph_.internalTensors.at(internalTensorCnt)};

    atb::Tensor *firstInTensor = &graph_.internalTensors.at(0);
    atb::Tensor *cosEmbedTensor = &graph_.internalTensors.at(1);
    atb::Tensor *sinEmbedTensor = &graph_.internalTensors.at(2);

    for (int layerId = 0; layerId < param_.layerNum; ++layerId) {
        auto &layerNode = graph_.nodes.at(nodeId++);

        atb_speed::falcon_7b::LayerFusionParam layerParam;
        layerParam.layerNormEps = param_.layerNormEps;
        layerParam.headNum = param_.headNum;
        layerParam.kvHeadNum = param_.kvHeadNum;
        layerParam.hiddenSize = param_.hiddenSize;
        layerParam.model = param_.model;
        layerParam.preScale = param_.preScale;
        layerParam.postScale = param_.postScale;
        layerParam.layerId = layerId;
        atb_speed::falcon_7b::FusionLayerOperation(layerParam, &op);
        layerNode.operation.reset(op);
        layerNode.inTensors.resize(layerNode.operation->GetInputNum());

        size_t inTensorId = 0;
        for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
            layerNode.inTensors.at(inTensorId++) =
                &graph_.weightTensors.at(layerId * WEIGHT_COUNT_PER_LAYER + weightTensorId + WEIGHT_COUNT_BEFORE_LAYER);
        }
        layerNode.inTensors.at(inTensorId++) = firstInTensor; // hidden states
        layerNode.inTensors.at(inTensorId++) = cosEmbedTensor;                                // cosTable
        layerNode.inTensors.at(inTensorId++) = sinEmbedTensor;                                // sinTable
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ATTENTIONMASK); // attentionMaskTensor
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PAST_KEY);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PAST_VALUE);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_TOKENOFFSET);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SEQLEN); // seqLen
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_MAX + layerId);
        layerNode.outTensors = {&graph_.internalTensors.at(INTERMEDIATETENSOR_COUNT_BEFORE_LAYER + layerId)};

        firstInTensor = layerNode.outTensors.at(0);
    }
    
    // layernorm
    auto &finalNormNode = graph_.nodes.at(nodeId++);
    atb::infer::LayerNormParam finalNormParam;
    finalNormParam.layerType = atb::infer::LayerNormParam::LAYER_NORM_NORM;
    finalNormParam.normParam.epsilon = param_.layerNormEps;
    finalNormParam.normParam.beginNormAxis = 2;
    finalNormParam.normParam.beginParamsAxis = 1;
    CREATE_OPERATION(finalNormParam, &op);
    finalNormNode.operation.reset(op);
    const int finalLayerNormWeightTensorId =
        graph_.weightTensors.size() - FINALNORMNODE_WEIGHT_COUNT - OUT_LM_HEAD_WEIGHT_COUNT;
    const int finalLayerNormOutTensorId = internalTensorSize - 1;
    finalNormNode.inTensors = {
        firstInTensor, &graph_.weightTensors.at(finalLayerNormWeightTensorId),
        &graph_.weightTensors.at(finalLayerNormWeightTensorId + 1)
        };
    finalNormNode.outTensors = {&graph_.internalTensors.at(finalLayerNormOutTensorId)};

    // lmhead
    auto &lmHeadNode = graph_.nodes.at(nodeId++);
    atb::infer::LinearParam lmHeadParm;
    lmHeadParm.hasBias = false;
    CREATE_OPERATION(lmHeadParm, &op);
    lmHeadNode.operation.reset(op);
    const int finalLinearWeightTensorId = graph_.weightTensors.size() - OUT_LM_HEAD_WEIGHT_COUNT;
    lmHeadNode.inTensors = {&graph_.internalTensors.at(finalLayerNormOutTensorId),
                            &graph_.weightTensors.at(finalLinearWeightTensorId)};
    lmHeadNode.outTensors = {&graph_.outTensors.at(0)};

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

    return atb::NO_ERROR;
}

atb::Status FlashAttentionModel::BindParamHostTensor(uint32_t nodeId)
{
    ATB_LOG(INFO) << "BindParamHostTensor";
    ATB_LOG(INFO) << "nodeId = " << nodeId;
    ATB_LOG(INFO) << "param_.layerNum = " << param_.layerNum;

    if (nodeId < OPERATION_COUNT_BEFORE_LAYER || nodeId >= OPERATION_COUNT_BEFORE_LAYER + param_.layerNum) {
        return atb::NO_ERROR;
    }
    auto &node = graph_.nodes.at(nodeId);
    const uint32_t tokenOffsetTensorId = 12;
    const uint32_t seqLenTensorId = 13;
    node.variantPack.inTensors.at(tokenOffsetTensorId).hostData = tokenOffset_.data();
    node.variantPack.inTensors.at(seqLenTensorId).hostData = seqLen_.data();
    ATB_LOG(INFO) << "BindParamHostTensor end";
    return atb::NO_ERROR;
}
} // namespace falcon_7b
} // namespace atb_speed