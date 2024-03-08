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
#include "quant_model.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtype-limits"
#include <nlohmann/json.hpp>
#pragma GCC diagnostic pop
#include <atb/atb_infer.h>
#include "telechat/layer/embedding_layer.h"
#include "telechat/layer/quant_layer.h"
#include "atb_speed/utils/model_factory.h"

namespace atb_speed {
namespace telechat {

REGISTER_MODEL(telechat, QuantFAModel);

const int WEIGHT_COUNT_PER_LAYER = 22;
const int WORD_EMBEDDING_WEIGHT_COUNT = 1;
const int FINAL_RMSNORM_WEIGHT_COUNT = 1;
const int OUT_LM_HEAD_WEIGHT_COUNT = 1;
const int INTERNAL_TENSOR_COUNT_BEFORE_LAYER = 3;
const int OPERATION_COUNT_BEFORE_LAYER = 1;
const int OPERATION_COUNT_AFTER_LAYER = 2;
const int OUT_TENSOR_NUM = 1;

enum InTensorId {
    IN_TENSOR_INPUT_IDS = 0,
    IN_TENSOR_POSITIONIDS,
    IN_TENSOR_COSTABLE,
    IN_TENSOR_SINTABLE,
    IN_TENSOR_ATTENTIONMASK,
    IN_TENSOR_PAST_KEY,
    IN_TENSOR_PAST_VALUE,
    IN_TOKENOFFSET,
    IN_SEQLEN,
    IN_HOLDER,
    IN_MAX_TENSOR
};

void QuantFAModel::QuantFAParam::FromString(const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);
    rmsNormEps = paramJson["rmsNormEps"].get<float>();
    headNum = paramJson["headNum"].get<int>();
    dk = paramJson["dk"].get<int>();
    layerNum = paramJson["layerNum"].get<int>();
    rank = paramJson["rank"].get<int>();
    rankSize = paramJson["rankSize"].get<int>();
    for (auto item : paramJson["float_query_layers"]) {
        float_query_layers.push_back(item.get<int>());
    }
    for (auto item : paramJson["float_kv_layers"]) {
        float_kv_layers.push_back(item.get<int>());
    }
    for (auto item : paramJson["float_down_layers"]) {
        float_down_layers.push_back(item.get<int>());
    }
    for (auto item : paramJson["inputScale_qkv"]) {
        inputScale_qkv.push_back(item.get<float>());
    }
    for (auto item : paramJson["inputOffset_qkv"]) {
        inputOffset_qkv.push_back(item.get<int>());
    }
    for (auto item : paramJson["inputScale_dense"]) {
        inputScale_dense.push_back(item.get<float>());
    }
    for (auto item : paramJson["inputOffset_dense"]) {
        inputOffset_dense.push_back(item.get<int>());
    }
    for (auto item : paramJson["inputScale_gate_up"]) {
        inputScale_gate_up.push_back(item.get<float>());
    }
    for (auto item : paramJson["inputOffset_gate_up"]) {
        inputOffset_gate_up.push_back(item.get<int>());
    }
    for (auto item : paramJson["inputScale_down_proj"]) {
        inputScale_down_proj.push_back(item.get<float>());
    }
    for (auto item : paramJson["inputOffset_down_proj"]) {
        inputOffset_down_proj.push_back(item.get<int>());
    }
    ATB_LOG(INFO) << "QuantFAModel param rmsNormEps:" << rmsNormEps << ", headNum:" << headNum << ", dk:" << dk
                  << ", layerNum:" << layerNum << ", rankSize:" << rankSize << ", rank:" << rank;
}

QuantFAModel::QuantFAModel(const std::string &param) : Model("QuantFAModel", param)
{
    param_.FromString(param);
}

QuantFAModel::~QuantFAModel() {}

uint32_t QuantFAModel::GetInputNum() const
{
    return graph_.inTensors.size();
}
uint32_t QuantFAModel::GetOutputNum() const
{
    return graph_.outTensors.size();
}

atb::Status QuantFAModel::InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
                                     std::vector<atb::TensorDesc> &outTensorDescs)
{
    if (outTensorDescs.size() != GetOutputNum()) {
        return atb::ERROR_INVALID_GRAPH;
    }

    outTensorDescs.at(0) = graph_.weightTensors.at(0).desc;
    outTensorDescs.at(0).shape.dimNum = 3;
    outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(IN_TENSOR_INPUT_IDS).shape.dims[0];
    outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(IN_TENSOR_INPUT_IDS).shape.dims[1];
    outTensorDescs.at(0).shape.dims[2] = graph_.weightTensors.at(graph_.weightTensors.size() - 1).desc.shape.dims[0];

    return atb::NO_ERROR;
}

int64_t QuantFAModel::BuildGraph()
{
    ATB_LOG(INFO) << "Enter Telechat QuantFAModel BuildGraph";

    const int weightTensorSize = WORD_EMBEDDING_WEIGHT_COUNT + WEIGHT_COUNT_PER_LAYER * param_.layerNum +
                                 FINAL_RMSNORM_WEIGHT_COUNT + OUT_LM_HEAD_WEIGHT_COUNT;
    graph_.weightTensors.resize(weightTensorSize);

    const int inTensorSize = IN_MAX_TENSOR + param_.layerNum;
    graph_.inTensors.resize(inTensorSize);

    const int outTensorSize = OUT_TENSOR_NUM;
    graph_.outTensors.resize(outTensorSize);

    const int nodeSize = OPERATION_COUNT_BEFORE_LAYER + param_.layerNum + OPERATION_COUNT_AFTER_LAYER;
    graph_.nodes.resize(nodeSize);

    const int internalTensorSize = graph_.nodes.size() + 1;
    graph_.internalTensors.resize(internalTensorSize);

    int nodeId = 0;
    atb::Operation *op = nullptr;

    auto &embeddingNode = graph_.nodes.at(nodeId++);
    EmbeddingLayerParam embeddingLayerParam;
    EmbeddingLayer(embeddingLayerParam, &op);
    embeddingNode.operation.reset(op);
    embeddingNode.inTensors = { &graph_.weightTensors.at(0), &graph_.inTensors.at(IN_TENSOR_INPUT_IDS),
                                &graph_.inTensors.at(IN_TENSOR_COSTABLE), &graph_.inTensors.at(IN_TENSOR_SINTABLE),
                                &graph_.inTensors.at(IN_TENSOR_POSITIONIDS) };
    embeddingNode.outTensors = { &graph_.internalTensors.at(0), &graph_.internalTensors.at(1),
                                 &graph_.internalTensors.at(2) };
    atb::Tensor *firstInTensor = &graph_.internalTensors.at(0);
    atb::Tensor *cosEmbedTensor = &graph_.internalTensors.at(1);
    atb::Tensor *sinEmbedTensor = &graph_.internalTensors.at(2);
    for (int layerId = 0; layerId < param_.layerNum; ++layerId) {
        auto &layerNode = graph_.nodes.at(nodeId++);

        QuantFALayerParam layerParam;

        layerParam.rmsNormEps = param_.rmsNormEps;
        layerParam.headNum = param_.headNum;
        layerParam.dk = param_.dk;
        layerParam.rank = param_.rank;
        layerParam.rankSize = param_.rankSize;
        if (std::find(param_.float_query_layers.begin(), param_.float_query_layers.end(), layerId) !=
            param_.float_query_layers.end()) {
            layerParam.isFloatQueryLayer = true;
            ATB_LOG(INFO) << "layerParam.isFloatQueryLayer" << layerParam.isFloatQueryLayer;
        }
        if (std::find(param_.float_kv_layers.begin(), param_.float_kv_layers.end(), layerId) !=
            param_.float_kv_layers.end()) {
            layerParam.isFloatKVLayer = true;
            ATB_LOG(INFO) << "layerParam.isFloatKVLayer" << layerParam.isFloatKVLayer;
        }
        if (std::find(param_.float_down_layers.begin(), param_.float_down_layers.end(), layerId) !=
            param_.float_down_layers.end()) {
            layerParam.isFloatDownLayer = true;
            ATB_LOG(INFO) << "layerParam.isFloatDownLayer" << layerParam.isFloatDownLayer;
        }
        layerParam.inputScale_qkv = param_.inputScale_qkv[layerId];
        layerParam.inputOffset_qkv = param_.inputOffset_qkv[layerId];
        layerParam.inputScale_dense = param_.inputScale_dense[layerId];
        layerParam.inputOffset_dense = param_.inputOffset_dense[layerId];
        layerParam.inputScale_gate_up = param_.inputScale_gate_up[layerId];
        layerParam.inputOffset_gate_up = param_.inputOffset_gate_up[layerId];
        layerParam.inputScale_down_proj = param_.inputScale_down_proj[layerId];
        layerParam.inputOffset_down_proj = param_.inputOffset_down_proj[layerId];

        QuantFALayer(layerParam, &op);
        layerNode.operation.reset(op);
        layerNode.inTensors.resize(layerNode.operation->GetInputNum());

        size_t inTensorId = 0;
        layerNode.inTensors.at(inTensorId++) = firstInTensor;
        for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
            layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(
                WORD_EMBEDDING_WEIGHT_COUNT + layerId * WEIGHT_COUNT_PER_LAYER + weightTensorId);
        }
        layerNode.inTensors.at(inTensorId++) = cosEmbedTensor;
        layerNode.inTensors.at(inTensorId++) = sinEmbedTensor;
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ATTENTIONMASK);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PAST_KEY);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PAST_VALUE);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TOKENOFFSET);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_SEQLEN);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_HOLDER);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_MAX_TENSOR + layerId);

        layerNode.outTensors = { &graph_.internalTensors.at(INTERNAL_TENSOR_COUNT_BEFORE_LAYER + layerId) };
        firstInTensor = layerNode.outTensors.at(0);
    }

    auto &finalRmsNormNode = graph_.nodes.at(nodeId++);
    atb::infer::RmsNormParam rmsNormParam;
    rmsNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    rmsNormParam.normParam.epsilon = param_.rmsNormEps;
    CREATE_OPERATION(rmsNormParam, &op);
    finalRmsNormNode.operation.reset(op);
    const int finalRmsNormWeightTensorId =
        graph_.weightTensors.size() - FINAL_RMSNORM_WEIGHT_COUNT - OUT_LM_HEAD_WEIGHT_COUNT;
    const int finalRmsNormOutTensorId = internalTensorSize - 1;
    finalRmsNormNode.inTensors = { firstInTensor, &graph_.weightTensors.at(finalRmsNormWeightTensorId) };
    finalRmsNormNode.outTensors = { &graph_.internalTensors.at(finalRmsNormOutTensorId) };

    auto &lmHeadNode = graph_.nodes.at(nodeId++);
    atb::infer::LinearParam linearParam;
    linearParam.hasBias = false;
    CREATE_OPERATION(linearParam, &op);
    lmHeadNode.operation.reset(op);
    const int lmHeadWeightTensorId = graph_.weightTensors.size() - OUT_LM_HEAD_WEIGHT_COUNT;
    lmHeadNode.inTensors = { &graph_.internalTensors.at(finalRmsNormOutTensorId),
                             &graph_.weightTensors.at(lmHeadWeightTensorId) };
    lmHeadNode.outTensors = { &graph_.outTensors.at(0) };
    return atb::NO_ERROR;
}

atb::Status QuantFAModel::ParseParam(const std::string &param)
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

    ATB_LOG(INFO) << "ParseParam tokenOffset:" << tokenOffset_ << ", seqLen:" << seqLen_;

    return atb::NO_ERROR;
}

atb::Status QuantFAModel::BindParamHostTensor(uint32_t nodeId)
{
    if (nodeId < OPERATION_COUNT_BEFORE_LAYER || nodeId >= static_cast<uint32_t>(OPERATION_COUNT_BEFORE_LAYER + param_.layerNum)) {
        return atb::NO_ERROR;
    }

    auto &node = graph_.nodes.at(nodeId);
    const uint32_t tokenOffsetTensorId = 28;
    const uint32_t seqLenTensorId = 29;
    node.variantPack.inTensors.at(tokenOffsetTensorId).hostData = tokenOffset_.data();
    node.variantPack.inTensors.at(seqLenTensorId).hostData = seqLen_.data();
    return atb::NO_ERROR;
}
}  // namespace telechat
}  // namespace atb_speed