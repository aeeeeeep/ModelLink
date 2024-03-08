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
#include "nlohmann/json.hpp"

#include "layers/parallel_layer_v2.h"
#include "models/vlmo/2b/layer/encoder_layer.h"
#include "models/vlmo/2b/layer/encoder_vl_layer.h"

namespace atb_speed {
namespace vlmo {
const int WEIGHT_COUNT_PER_LAYER = 22;
const int WEIGHT_COUNT_PER_VL_LAYER = 16;

const int FINALNORMNODE_WEIGHT_COUNT = 2;
const int OUT_LM_HEAD_WEIGHT_COUNT = 2;
const int INTERMEDIATETENSOR_COUNT_BEFORE_VL_LAYER = 1;
const int OPERATION_COUNT_AFTER_LAYER = 0;
const int OPERATION_COUNT_BEFORE_LAYER = 0;

enum InTensorId : int {
    IN_TENSOR_X = 0,
    //IN_TENSOR_MASK,
    IN_TENSOR_PAST_KEY,
    IN_TENSOR_PAST_VALUE,
    IN_TENSOR_TOKENOFFSET,
    IN_TENSOR_SEQLEN,
    IN_HOLDER,
    IN_TENSOR_MAX,
};

enum OutTensorId : int {
    OUT_LAYEROUT = 0,
    OUT_TENSOR_MAX,
};

void FlashAttentionModel::Param::FromString(const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);
    layerNormEps = paramJson["rmsNormEps"].get<double>();
    headNum = paramJson["headNum"].get<int>();
    dk = paramJson["dk"].get<int>();
    layerNum = paramJson["layerNum"].get<int>();
    if (paramJson.contains("rank")) {
        rank = paramJson["rank"].get<int>();
    }
    if (paramJson.contains("rankSize")) {
        rankSize = paramJson["rankSize"].get<int>();
    }
    if (paramJson.contains("backend")) {
        backend = paramJson["backend"];
    }
    if (paramJson.contains("maxTextLen")) {
        maxTextLen = paramJson["maxTextLen"];
    }
    if (paramJson.contains("vlLayerIndex")) {
        vlLayerIndex = paramJson["vlLayerIndex"];
    }
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
    if (outTensorDescs.size() != GetOutputNum()) {
        return atb::ERROR_INVALID_GRAPH;
    }
    outTensorDescs.at(0) = inTensorDescs.at(0);
    return atb::NO_ERROR;
}

int64_t FlashAttentionModel::BuildGraph()
{
    const int weightTensorSize = WEIGHT_COUNT_PER_LAYER * param_.vlLayerIndex +
                                 WEIGHT_COUNT_PER_VL_LAYER * (param_.layerNum - param_.vlLayerIndex);
    graph_.weightTensors.resize(weightTensorSize);

    graph_.inTensors.resize(IN_TENSOR_MAX + param_.layerNum * 2);
    graph_.outTensors.resize(OUT_TENSOR_MAX);

    const int nodeSize = OPERATION_COUNT_BEFORE_LAYER + param_.layerNum + OPERATION_COUNT_AFTER_LAYER;
    graph_.nodes.resize(nodeSize);

    const int internalTensorSize = (param_.layerNum - 1) * OUT_TENSOR_MAX;
    graph_.internalTensors.resize(internalTensorSize);

    int nodeId = 0;

    atb::Operation *op = nullptr;

    ATB_LOG(INFO) << __func__ << " called, layerNum: " << param_.layerNum;
    atb::Tensor *firstInTensor = &graph_.inTensors.at(0);
    int layerId = 0;
    for (; layerId < param_.vlLayerIndex; ++layerId) {
        ATB_LOG(INFO) << __func__ << " layerId " << layerId << " create node";
        auto &layerNode = graph_.nodes.at(nodeId++);

        atb_speed::vlmo::EncoderLayerParam opParam;
        opParam.layerNormEps = param_.layerNormEps;
        opParam.headNum = param_.headNum;
        opParam.dk = param_.dk;
        opParam.rank = param_.rank;
        opParam.rankSize = param_.rankSize;
        opParam.backend = param_.backend;
        opParam.maxTextLen = param_.maxTextLen;
        atb_speed::vlmo::EncoderLayer(opParam, &op);
        layerNode.operation.reset(op);
        layerNode.inTensors.resize(layerNode.operation->GetInputNum());

        size_t inTensorId = 0;
        layerNode.inTensors.at(inTensorId++) = firstInTensor;
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_MAX + layerId + param_.layerNum);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PAST_KEY);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PAST_VALUE);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_TOKENOFFSET);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SEQLEN);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_HOLDER);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_MAX + layerId);

        for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
            ATB_LOG(INFO) << __func__ << " layerId " << layerId << " weightID" << weightTensorId << " -> in weight ID"
                          << (layerId * WEIGHT_COUNT_PER_LAYER + weightTensorId);
            layerNode.inTensors.at(inTensorId++) =
                &graph_.weightTensors.at(layerId * WEIGHT_COUNT_PER_LAYER + weightTensorId);
        }

        layerNode.outTensors.resize(layerNode.operation->GetOutputNum());
        for (int i = 0; i < OUT_TENSOR_MAX; i++) {
            layerNode.outTensors.at(i) = &graph_.internalTensors.at((layerId * 1) + i);
        }
        firstInTensor = layerNode.outTensors.at(0);
    }
    for (; layerId < param_.layerNum; ++layerId) {
        ATB_LOG(INFO) << __func__ << " layerId " << layerId << " create node";
        auto &layerNode = graph_.nodes.at(nodeId++);

        atb_speed::vlmo::EncoderVllayerParam opParam;
        opParam.layerNormEps = param_.layerNormEps;
        opParam.headNum = param_.headNum;
        opParam.dk = param_.dk;
        opParam.rank = param_.rank;
        opParam.rankSize = param_.rankSize;
        opParam.backend = param_.backend;
        opParam.maxTextLen = param_.maxTextLen;
        atb_speed::vlmo::EncoderVlLayer(opParam, &op);
        layerNode.operation.reset(op);
        layerNode.inTensors.resize(layerNode.operation->GetInputNum());

        size_t inTensorId = 0;
        layerNode.inTensors.at(inTensorId++) = firstInTensor;
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_MAX + layerId + param_.layerNum);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PAST_KEY);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PAST_VALUE);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_TOKENOFFSET);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SEQLEN);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_HOLDER);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_MAX + layerId);

        for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_VL_LAYER; ++weightTensorId) {
            layerNode.inTensors.at(inTensorId++) =
                &graph_.weightTensors.at((WEIGHT_COUNT_PER_LAYER * param_.vlLayerIndex) +
                                         (layerId - param_.vlLayerIndex) * WEIGHT_COUNT_PER_VL_LAYER + weightTensorId);
        }

        layerNode.outTensors.resize(layerNode.operation->GetOutputNum());
        if (layerId + 1 == param_.layerNum) {
            for (int i = 0; i < OUT_TENSOR_MAX; i++) {
                layerNode.outTensors.at(i) = &graph_.outTensors.at(i);
            }
        } else {
            for (int i = 0; i < OUT_TENSOR_MAX; i++) {
                layerNode.outTensors.at(i) = &graph_.internalTensors.at((layerId * OUT_TENSOR_MAX) + i);
            }
            firstInTensor = layerNode.outTensors.at(0);
        }
    }
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
    if (nodeId >= param_.layerNum) {
        return atb::NO_ERROR;
    }

    auto &node = graph_.nodes.at(nodeId);
    const uint32_t tokenOffsetTensorId = 4;
    const uint32_t seqLenTensorId = 5;
    node.variantPack.inTensors.at(tokenOffsetTensorId).hostData = tokenOffset_.data();
    node.variantPack.inTensors.at(seqLenTensorId).hostData = seqLen_.data();
    ATB_LOG(INFO) << "BindParamHostTensor end";
    return atb::NO_ERROR;
}
} // namespace vlmo
} // namespace atb_speed