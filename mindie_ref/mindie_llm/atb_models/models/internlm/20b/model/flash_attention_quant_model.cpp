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
#include <algorithm>
#include <iostream>

#include "models/internlm/20b/model/flash_attention_quant_model.h"

#include "atb/atb_infer.h"
#include "nlohmann/json.hpp"

#include "models/internlm/20b/layer/flash_attention_quant_layer.h"
#include "models/internlm/20b/layer/flash_attention_rope_antioutlier_layer.h"

namespace atb_speed {
namespace internlm_20b {
enum InTensorId : int {
    IN_TENSOR_INPUT_IDS = 0,
    IN_TENSOR_COSEMBED,
    IN_TENSOR_SINEMBED,
    IN_TENSOR_ATTENTION_MASK,
    IN_TENSOR_PAST_KEY,
    IN_TENSOR_PAST_VALUE,
    IN_TENSOR_TOKEN_OFFSET,
    IN_TENSOR_SEQ_LEN,
    IN_BETA,
    IN_HOLDER,
    IN_TENSOR_MAX, // 10
};

enum OutTensorId : int {
    OUT_TENSOR_HIDDEN_STATES = 0,
    OUT_TENSOR_MAX,
};

const int WEIGHT_COUNT_PER_LAYER = 25;            // 量化层权重输入数量
const int ROLLBACK_WEIGHT_COUNT_PER_LAYER = 16;   // 回退层权重输入数量
const int WORD_EMBEDDING_NODE_WEIGHT_COUNT = 1;   // word embedding权重输入数量
const int FINAL_NORM_NODE_WEIGHT_COUNT = 2;       // model层 self.norm输入权重数量
const int OUT_LM_HEAD_WEIGHT_COUNT = 1;           // lm_head输入权重数量
const int OPERATION_COUNT_BEFORE_LAYER = 1;       // 调用layer层前调用operation数量
const int INTERNAL_TENSOR_COUNT_BEFORE_LAYER = 1; // 调用layer前用过的临时tensor数量
const int LAYER_FIRST_OUT_TENSORS = 0;            // layer计算图output index

void FlashAttentionQuantModel::Param::FromString(const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);
    rmsNormEps = paramJson["rmsNormEps"].get<double>();
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
        backend = paramJson["backend"].get<std::string>();
    }
    for (const auto &item : paramJson["qProjInputScale"]) {
        qProjInputScale.push_back(item.get<float>());
    }
    for (const auto &item : paramJson["qProjInputOffset"]) {
        qProjInputOffset.push_back(item.get<int>());
    }
    for (const auto &item : paramJson["kProjInputScale"]) {
        kProjInputScale.push_back(item.get<float>());
    }
    for (const auto &item : paramJson["kProjInputOffset"]) {
        kProjInputOffset.push_back(item.get<int>());
    }
    for (const auto &item : paramJson["vProjInputScale"]) {
        vProjInputScale.push_back(item.get<float>());
    }
    for (const auto &item : paramJson["vProjInputOffset"]) {
        vProjInputOffset.push_back(item.get<int>());
    }
    for (const auto &item : paramJson["oProjInputScale"]) {
        oProjInputScale.push_back(item.get<float>());
    }
    for (const auto &item : paramJson["oProjInputOffset"]) {
        oProjInputOffset.push_back(item.get<int>());
    }
    for (const auto &item : paramJson["gateProjInputScale"]) {
        gateProjInputScale.push_back(item.get<float>());
    }
    for (const auto &item : paramJson["gateProjInputOffset"]) {
        gateProjInputOffset.push_back(item.get<int>());
    }
    for (const auto &item : paramJson["downProjInputScale"]) {
        downProjInputScale.push_back(item.get<float>());
    }
    for (const auto &item : paramJson["downProjInputOffset"]) {
        downProjInputOffset.push_back(item.get<int>());
    }
    for (const auto &item : paramJson["float_layers"]) {
        float_layers.push_back(item.get<int>());
    }
}

FlashAttentionQuantModel::FlashAttentionQuantModel(const std::string &param) : Model("FlashAttentionQuantModel", param)
{
    param_.FromString(param);
}

FlashAttentionQuantModel::~FlashAttentionQuantModel() = default;

uint32_t FlashAttentionQuantModel::GetInputNum() const
{
    return graph_.inTensors.size();
}

uint32_t FlashAttentionQuantModel::GetOutputNum() const
{
    return graph_.outTensors.size();
}

// 计算传入权重的起始id
int CalWeightTensorStartId(const int layerId, int floatLayerNum)
{
    int weightTensorStartId = WORD_EMBEDDING_NODE_WEIGHT_COUNT;
    weightTensorStartId +=
        (floatLayerNum * ROLLBACK_WEIGHT_COUNT_PER_LAYER + (layerId - floatLayerNum) * WEIGHT_COUNT_PER_LAYER);
    return weightTensorStartId;
}

// model修改输出shape
atb::Status FlashAttentionQuantModel::InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
    std::vector<atb::TensorDesc> &outTensorDescs)
{
    if (outTensorDescs.size() != GetOutputNum()) {
        return atb::ERROR_INVALID_GRAPH;
    }
    const int64_t outDim = graph_.weightTensors.at(graph_.weightTensors.size() - 1).desc.shape.dims[0];
    outTensorDescs.at(0) = graph_.weightTensors.at(0).desc;
    outTensorDescs.at(0).shape.dimNum = 3;
    outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
    outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1];
    outTensorDescs.at(0).shape.dims[2] = outDim;
    return atb::NO_ERROR;
}

int64_t FlashAttentionQuantModel::BuildGraph()
{
    int floatLayerSize = param_.float_layers.size();
    const int weightTensorSize = WORD_EMBEDDING_NODE_WEIGHT_COUNT + ROLLBACK_WEIGHT_COUNT_PER_LAYER * floatLayerSize +
        WEIGHT_COUNT_PER_LAYER * (param_.layerNum - floatLayerSize) + FINAL_NORM_NODE_WEIGHT_COUNT +
        OUT_LM_HEAD_WEIGHT_COUNT; // 传入权重总数
    graph_.weightTensors.resize(weightTensorSize);

    graph_.inTensors.resize(IN_TENSOR_MAX + param_.layerNum); // 输入tensor数量
    graph_.outTensors.resize(OUT_TENSOR_MAX);                 // 输出tensor数量

    const int nodeSize = param_.layerNum + OPERATION_COUNT_BEFORE_LAYER + FINAL_NORM_NODE_WEIGHT_COUNT +
        OUT_LM_HEAD_WEIGHT_COUNT; // 计算图节点总数
    graph_.nodes.resize(nodeSize);

    const int internalTensorSize = graph_.nodes.size() - 1; // 临时tensor数量
    graph_.internalTensors.resize(internalTensorSize);

    // embed_tokens
    int nodeId = 0;
    auto &wordEmbeddingNode = graph_.nodes.at(nodeId++);
    atb::infer::GatherParam wordEmbeddingParam;
    atb::Operation *op = nullptr;
    CREATE_OPERATION(wordEmbeddingParam, &op);
    wordEmbeddingNode.operation.reset(op);
    wordEmbeddingNode.inTensors = { &graph_.weightTensors.at(0), &graph_.inTensors.at(IN_TENSOR_INPUT_IDS) };
    wordEmbeddingNode.outTensors = { &graph_.internalTensors.at(0) };

    atb::Tensor *firstInTensor = &graph_.internalTensors.at(0);
    int floatLayerNum = 0;
    for (int layerId = 0; layerId < param_.layerNum; ++layerId) {
        auto &layerNode = graph_.nodes.at(nodeId++);
        // 计算当前layer传入权重的起始id
        size_t weightTensorStartId = CalWeightTensorStartId(layerId, floatLayerNum);
        size_t inTensorId = 0;

        // 判断layer是量化层还是回退层
        if (std::find(std::begin(param_.float_layers), std::end(param_.float_layers), layerId) !=
            std::end(param_.float_layers)) {
            // 回退层，调用 FlashAttentionRopeAntiOutlierLayer
            atb_speed::internlm_20b::FlashAttentionRopeAntiOutlierLayerParam opParam;
            opParam.rmsNormEps = param_.rmsNormEps;
            opParam.headNum = param_.headNum;
            opParam.dk = param_.dk;
            opParam.rank = param_.rank;
            opParam.rankSize = param_.rankSize;
            opParam.backend = param_.backend;
            atb_speed::internlm_20b::FlashAttentionRopeAntiOutlierLayer(opParam, &op);
            layerNode.operation.reset(op);
            layerNode.inTensors.resize(layerNode.operation->GetInputNum());
            layerNode.inTensors.at(inTensorId++) = firstInTensor;
            for (size_t weightTensorId = 0; weightTensorId < ROLLBACK_WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
                layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(weightTensorStartId + weightTensorId);
            }
            floatLayerNum++;
        } else {
            // 量化层，调用FlashAttentionQuantLayer
            atb_speed::internlm_20b::FlashAttentionQuantLayerParam opParam;
            opParam.rmsNormEps = param_.rmsNormEps;
            opParam.headNum = param_.headNum;
            opParam.dk = param_.dk;
            opParam.rank = param_.rank;
            opParam.rankSize = param_.rankSize;
            opParam.backend = param_.backend;
            opParam.qProjInputScale = param_.qProjInputScale[layerId];
            opParam.qProjInputOffset = param_.qProjInputOffset[layerId];
            opParam.kProjInputScale = param_.kProjInputScale[layerId];
            opParam.kProjInputOffset = param_.kProjInputOffset[layerId];
            opParam.vProjInputScale = param_.vProjInputScale[layerId];
            opParam.vProjInputOffset = param_.vProjInputOffset[layerId];
            opParam.oProjInputScale = param_.oProjInputScale[layerId];
            opParam.oProjInputOffset = param_.oProjInputOffset[layerId];
            opParam.gateProjInputScale = param_.gateProjInputScale[layerId];
            opParam.gateProjInputOffset = param_.gateProjInputOffset[layerId];
            opParam.downProjInputScale = param_.downProjInputScale[layerId];
            opParam.downProjInputOffset = param_.downProjInputOffset[layerId];
            atb_speed::internlm_20b::FlashAttentionQuantLayer(opParam, &op);
            layerNode.operation.reset(op);
            layerNode.inTensors.resize(layerNode.operation->GetInputNum());

            layerNode.inTensors.at(inTensorId++) = firstInTensor; // IN_HIDDEN_STATES
            for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
                // IN_NORM_WEIGHT ->IN_SELF_OUT_NORM_WEIGHT
                layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(weightTensorStartId + weightTensorId);
            }
        }
        // inTensors -> 通过 input参数传入
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_COSEMBED);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SINEMBED);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ATTENTION_MASK);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PAST_KEY);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PAST_VALUE);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_TOKEN_OFFSET);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SEQ_LEN);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_BETA);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_HOLDER);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_MAX + layerId);

        layerNode.outTensors = { &graph_.internalTensors.at(INTERNAL_TENSOR_COUNT_BEFORE_LAYER + layerId) };

        firstInTensor = layerNode.outTensors.at(LAYER_FIRST_OUT_TENSORS);
    }

    // self.norm operation
    auto &finalNormNode = graph_.nodes.at(nodeId++);
    atb::infer::RmsNormParam finalNormParam;
    finalNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    finalNormParam.normParam.epsilon = param_.rmsNormEps;
    CREATE_OPERATION(finalNormParam, &op);
    finalNormNode.operation.reset(op);
    const int finalLayerNormWeightTensorId =
        graph_.weightTensors.size() - FINAL_NORM_NODE_WEIGHT_COUNT - OUT_LM_HEAD_WEIGHT_COUNT;
    const int finalLayerNormOutTensorId = internalTensorSize - 2;
    finalNormNode.inTensors = { firstInTensor, &graph_.weightTensors.at(finalLayerNormWeightTensorId) };
    finalNormNode.outTensors = { &graph_.internalTensors.at(finalLayerNormOutTensorId) };

    // self.norm后加上离群点抑制的bias
    auto &finalNormAddNode = graph_.nodes.at(nodeId++);
    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CREATE_OPERATION(addParam, &op);
    finalNormAddNode.operation.reset(op);
    const int finalLayerNormWeightAddBiasTensorId = finalLayerNormWeightTensorId + 1;
    const int finalLayerNormAddBiasOutTensorId = internalTensorSize - 1;
    finalNormAddNode.inTensors = { &graph_.internalTensors.at(finalLayerNormOutTensorId),
        &graph_.weightTensors.at(finalLayerNormWeightAddBiasTensorId) };
    finalNormAddNode.outTensors = { &graph_.internalTensors.at(finalLayerNormAddBiasOutTensorId) };

    // self.lm_head operation
    auto &outLinearNode = graph_.nodes.at(nodeId++);
    atb::infer::LinearParam outLinearParm = { false, false, false };
    CREATE_OPERATION(outLinearParm, &op);
    outLinearNode.operation.reset(op);
    const int finalLinearWeightTensorId = graph_.weightTensors.size() - OUT_LM_HEAD_WEIGHT_COUNT;
    outLinearNode.inTensors = { &graph_.internalTensors.at(finalLayerNormAddBiasOutTensorId),
        &graph_.weightTensors.at(finalLinearWeightTensorId) };
    outLinearNode.outTensors = { &graph_.outTensors.at(0) };
    return atb::NO_ERROR;
}

atb::Status FlashAttentionQuantModel::ParseParam(const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);
    tokenOffset_.clear();
    for (const auto &item : paramJson["tokenOffset"]) {
        tokenOffset_.push_back(item.get<int>());
    }
    seqLen_.clear();
    for (const auto &item : paramJson["seqLen"]) {
        seqLen_.push_back(item.get<int>());
    }
    return atb::NO_ERROR;
}

// layer中的tokenOffset和seqLen绑定id
atb::Status FlashAttentionQuantModel::BindParamHostTensor(uint32_t nodeId)
{
    if (nodeId < OPERATION_COUNT_BEFORE_LAYER || nodeId >= OPERATION_COUNT_BEFORE_LAYER + param_.layerNum) {
        return atb::NO_ERROR;
    }
    auto &node = graph_.nodes.at(nodeId);
    const uint32_t rollbackTokenOffsetTensorId = 22; // 使用layer里的ID（回退层）
    const uint32_t rollbackSeqLenTensorId = 23;
    const uint32_t tokenOffsetTensorId = 31; // 使用layer里的ID （量化层）
    const uint32_t seqLenTensorId = 32;

    if (std::find(std::begin(param_.float_layers), std::end(param_.float_layers), nodeId - 1) !=
        std::end(param_.float_layers)) {
        // 绑定回退层
        node.variantPack.inTensors.at(rollbackTokenOffsetTensorId).hostData = tokenOffset_.data();
        node.variantPack.inTensors.at(rollbackSeqLenTensorId).hostData = seqLen_.data();
    } else {
        // 绑定量化层
        node.variantPack.inTensors.at(tokenOffsetTensorId).hostData = tokenOffset_.data();
        node.variantPack.inTensors.at(seqLenTensorId).hostData = seqLen_.data();
    }

    ATB_LOG(INFO) << "BindParamHostTensor end";
    return atb::NO_ERROR;
}
} // namespace internlm_20b
} // namespace atb_speed