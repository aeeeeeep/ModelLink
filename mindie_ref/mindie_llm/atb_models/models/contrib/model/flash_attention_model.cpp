/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
#include "models/contrib/layer/flash_attention_layer.h"
#include "models/llama2/70b/operation/layer_embedding.h"
#include "models/llama2/70b/operation/llama_lmhead.h"
#include "nlohmann/json.hpp"

namespace atb_speed {
namespace contrib {
const int WEIGHT_COUNT_PER_LAYER = 23;
const int OPERATION_COUNT_BEFORE_LAYER = 1;
const int OPERATION_COUNT_AFTER_LAYER = 2;
const int WEIGHT_COUNT_BEFORE_LAYER = 1;
const int FINALNORMNODE_WEIGHT_COUNT = 1;
const int OUT_LM_HEAD_WEIGHT_COUNT = 1;
const int INTERMEDIATETENSOR_COUNT_BEFORE_LAYER = 3;

enum InTensorId : int {
    IN_TENSOR_INPUTIDS = 0,
    IN_TENSOR_POSITIONID,
    IN_TENSOR_COSTABLE,
    IN_TENSOR_SINTABLE,
    IN_TENSOR_ATTENTIONMASK,
    IN_TENSOR_PAST_KEY,
    IN_TENSOR_PAST_VALUE,
    IN_TENSOR_TOKENOFFSET,
    IN_TENSOR_SEQLEN,
    IN_TENSOR_MAX,
};

enum OutTensorId {
    OUT_TENSOR_HIDDENSTATES = 0,
    OUT_TENSOR_MAX,
};

void FlashAttentionModel::Param::FromString(const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);
    rmsNormEps = paramJson["rmsNormEps"].get<double>();
    headNum = paramJson["headNum"].get<int>();
    dk = paramJson["dk"].get<int>();
    layerNum = paramJson["layerNum"].get<int>();
    rank = paramJson["rank"].get<int>();
    rankSize = paramJson["rankSize"].get<int>();
    kvHeadNum = paramJson["kvHeadNum"].get<int>();
    if (paramJson.find("backend") != paramJson.end()) {
        backend = paramJson["backend"].get<std::string>();
    }
    for (auto item : paramJson["tokenOffset"]) {
        tokenOffset.push_back(item.get<int>());
    }
    for (auto item : paramJson["seqLen"]) {
        seqLen.push_back(item.get<int>());
    }
    if (paramJson.contains("rotaryCoeff")) {
        rotaryCoeff = paramJson["rotaryCoeff"].get<int>();
    }
    ATB_LOG(INFO) << "FlashAttentionModel param rmsNormEps:" << rmsNormEps << ", headNum:" << headNum << ", dk:" << dk
                  << ", layerNum:" << layerNum << ", rank:" << rank << ", rankSize:" << rankSize
                  << ", kvHeadNum:" << kvHeadNum;
}

FlashAttentionModel::FlashAttentionModel(const std::string &param) : Model("FlashAttentionModel", param)
{
    param_.FromString(param);
}

FlashAttentionModel::~FlashAttentionModel() = default;

uint32_t FlashAttentionModel::GetInputNum() const { return graph_.inTensors.size(); }

uint32_t FlashAttentionModel::GetOutputNum() const { return graph_.outTensors.size(); }

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
    outTensorDescs.at(0).shape.dims[1] = 1;
    outTensorDescs.at(0).shape.dims[2] = outDim;
    return atb::NO_ERROR;
}

int64_t FlashAttentionModel::BuildGraph()
{
    ATB_LOG(INFO) << "Enter FlashAttentionModel BuildGraph";
    const int weightTensorSize = WEIGHT_COUNT_BEFORE_LAYER + WEIGHT_COUNT_PER_LAYER * param_.layerNum +
                                 FINALNORMNODE_WEIGHT_COUNT + OUT_LM_HEAD_WEIGHT_COUNT;
    graph_.weightTensors.resize(weightTensorSize);

    graph_.inTensors.resize(IN_TENSOR_MAX + param_.layerNum);
    graph_.outTensors.resize(OUT_TENSOR_MAX);

    const int nodeSize = OPERATION_COUNT_BEFORE_LAYER + param_.layerNum + OPERATION_COUNT_AFTER_LAYER;
    graph_.nodes.resize(nodeSize);

    const int internalTensorSize = graph_.nodes.size() + 1;
    graph_.internalTensors.resize(internalTensorSize);

    int nodeId = 0;
    atb::Operation *op = nullptr;

    auto &embeddingNode = graph_.nodes.at(nodeId++);
    atb_speed::llama2_70b::LayerEmbeddingParam layerEmbeddingParam;
    layerEmbeddingParam.rank = param_.rank;
    layerEmbeddingParam.rankSize = param_.rankSize;
    layerEmbeddingParam.backend = param_.backend;
    atb_speed::llama2_70b::LayerEmbedding(layerEmbeddingParam, &op);
    embeddingNode.operation.reset(op);
    embeddingNode.inTensors = {&graph_.weightTensors.at(0),                 // [vocab_size, h]
                               &graph_.inTensors.at(IN_TENSOR_INPUTIDS),    // [b, s]
                               &graph_.inTensors.at(IN_TENSOR_COSTABLE),    // [max_seq, hd]
                               &graph_.inTensors.at(IN_TENSOR_SINTABLE),    // [max_seq, hd]
                               &graph_.inTensors.at(IN_TENSOR_POSITIONID)}; // [b, s]
    embeddingNode.outTensors = {&graph_.internalTensors.at(0),              // [b, s, h]
                                &graph_.internalTensors.at(1),              // [s, hd]
                                &graph_.internalTensors.at(2)};             // [s, hd]

    atb::Tensor *firstInTensor = &graph_.internalTensors.at(0);
    atb::Tensor *cosEmbedTensor = &graph_.internalTensors.at(1);
    atb::Tensor *sinEmbedTensor = &graph_.internalTensors.at(2);

    for (int layerId = 0; layerId < param_.layerNum; ++layerId) {
        auto &layerNode = graph_.nodes.at(nodeId++);

        atb_speed::contrib::FlashAttentionLayerParam modelParam;
        modelParam.rmsNormEps = param_.rmsNormEps;
        modelParam.headNum = param_.headNum;
        modelParam.dk = param_.dk;
        modelParam.rank = param_.rank;
        modelParam.rankSize = param_.rankSize;
        modelParam.kvHeadNum = param_.kvHeadNum;
        modelParam.layerId = layerId;
        modelParam.backend = param_.backend;
        atb_speed::contrib::FlashAttentionLayer(modelParam, &op);
        layerNode.operation.reset(op);
        layerNode.inTensors.resize(layerNode.operation->GetInputNum());

        size_t inTensorId = 0;
        layerNode.inTensors.at(inTensorId++) = firstInTensor;
        for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
            layerNode.inTensors.at(inTensorId++) =
                &graph_.weightTensors.at(layerId * WEIGHT_COUNT_PER_LAYER + weightTensorId + WEIGHT_COUNT_BEFORE_LAYER);
        }
        layerNode.inTensors.at(inTensorId++) = cosEmbedTensor;
        layerNode.inTensors.at(inTensorId++) = sinEmbedTensor;
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ATTENTIONMASK);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PAST_KEY);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PAST_VALUE);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_TOKENOFFSET);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SEQLEN);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_MAX + layerId);
        layerNode.outTensors = {&graph_.internalTensors.at(INTERMEDIATETENSOR_COUNT_BEFORE_LAYER + layerId)};

        firstInTensor = layerNode.outTensors.at(0);
    }

    // [b, s, h] * [h] -> [b, s, h]
    auto &finalNormNode = graph_.nodes.at(nodeId++);
    atb::infer::RmsNormParam finalNormParam;
    finalNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    finalNormParam.normParam.epsilon = param_.rmsNormEps;
    CREATE_OPERATION(finalNormParam, &op);
    finalNormNode.operation.reset(op);
    const int finalLayerNormWeightTensorId =
        graph_.weightTensors.size() - FINALNORMNODE_WEIGHT_COUNT - OUT_LM_HEAD_WEIGHT_COUNT;
    const int finalLayerNormOutTensorId = internalTensorSize - 1;
    finalNormNode.inTensors = {firstInTensor, &graph_.weightTensors.at(finalLayerNormWeightTensorId)};
    finalNormNode.outTensors = {&graph_.internalTensors.at(finalLayerNormOutTensorId)};

    // [b, s, h] * [vocab_size, h] T = [b, 1, vocab_size]
    auto &lmHeadNode = graph_.nodes.at(nodeId++);
    atb_speed::llama2_70b::llamaLmHeadParam lmHeadParm;
    CreateLlamaLmHead(lmHeadParm, &op);
    lmHeadNode.operation.reset(op);
    const int finalLinearWeightTensorId = graph_.weightTensors.size() - OUT_LM_HEAD_WEIGHT_COUNT;
    lmHeadNode.inTensors = {&graph_.internalTensors.at(finalLayerNormOutTensorId),
                            &graph_.weightTensors.at(finalLinearWeightTensorId)};
    lmHeadNode.outTensors = {&graph_.outTensors.at(0)};
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
    const uint32_t tokenOffsetTensorId = 15;
    const uint32_t seqLenTensorId = 16;
    node.variantPack.inTensors.at(tokenOffsetTensorId).hostData = tokenOffset_.data();
    node.variantPack.inTensors.at(seqLenTensorId).hostData = seqLen_.data();
    ATB_LOG(INFO) << "BindParamHostTensor end";
    return atb::NO_ERROR;
}
} // namespace contrib
} // namespace atb_speed