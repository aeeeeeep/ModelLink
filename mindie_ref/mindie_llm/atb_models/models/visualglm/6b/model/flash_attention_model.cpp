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
#include <nlohmann/json.hpp>
#include <atb/atb_infer.h>
#include "atb_speed/log.h"
#include "visualglm/6b/layer/flash_attention_layer.h"
#pragma GCC diagnostic ignored "-Wsign-compare"

namespace atb_speed {
namespace visualglm_6b {
const int WEIGHT_COUNT_PER_LAYER = 12;
const int WORDEMBEDDINGNODE_WEIGHT_COUNT = 1;
const int FINALNORMNODE_WEIGHT_COUNT = 2;
const int LMHEADNODE_WEIGHT_COUNT = 1;
const int OPERATION_COUNT_BEFORE_LAYER = 1;
const int OPERATION_COUNT_AFTER_LAYER = 3;

enum InTensorId {
    IN_TENSOR_HIDDENSTATES = 0,
    IN_TENSOR_POSITIONID,
    IN_TENSOR_COSTABLE,
    IN_TENSOR_SINTABLE,
    IN_TENSOR_ATTENTIONMASK,
    IN_TENSOR_PASTKEY,
    IN_TENSOR_PASTVALUE,
    IN_TENSOR_TOKENOFFSET,
    IN_TENSOR_SEQLEN,
    IN_TENSOR_MAX,
};

void FlashAttentionModel::Param::FromString(const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);
    layerNormEps = paramJson["layerNormEps"].get<double>();
    headNum = paramJson["headNum"].get<int>();
    headDim = paramJson["headDim"].get<int>();
    layerNum = paramJson["layerNum"].get<int>();
    isEncoder = paramJson["isEncoder"].get<bool>();
    operationCountBeforeLayers = isEncoder ? OPERATION_COUNT_BEFORE_LAYER : OPERATION_COUNT_BEFORE_LAYER + 1;
    for (auto item : paramJson["qScale"]) {
        qScale.push_back(item.get<float>());
    }
    for (auto item : paramJson["qkScale"]) {
        qkScale.push_back(item.get<float>());
    }
    residualAddScale = paramJson["residualAddScale"].get<float>();
    if (paramJson.contains("beginNormAxis")) {
        beginNormAxis = paramJson["beginNormAxis"].get<int>();
    }
    ATB_LOG(INFO) << "VisualGlm6BFlashAttentionModel param layerNormEps:" << layerNormEps << ", headNum:" << headNum
                  << ", headDim:" << headDim << ", layerNum:" << layerNum << ", qScale:" << qScale
                  << ", qkScale:" << qkScale << ", residualAddScale:" << residualAddScale
                  << ", beginNormAxis:" << beginNormAxis;
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

    outTensorDescs.at(0) = graph_.weightTensors.at(0).desc;
    outTensorDescs.at(0).shape.dimNum = 3;
    outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
    outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1];
    outTensorDescs.at(0).shape.dims[2] = graph_.weightTensors.back().desc.shape.dims[0];
    return atb::NO_ERROR;
}

int64_t FlashAttentionModel::BuildGraph()
{
    const int weightTensorSize =
        WORDEMBEDDINGNODE_WEIGHT_COUNT + WEIGHT_COUNT_PER_LAYER * param_.layerNum + FINALNORMNODE_WEIGHT_COUNT + LMHEADNODE_WEIGHT_COUNT;
    graph_.weightTensors.resize(weightTensorSize);

    graph_.inTensors.resize(IN_TENSOR_MAX + param_.layerNum);
    graph_.outTensors.resize(1);

    const int nodeSize = param_.layerNum + param_.operationCountBeforeLayers + OPERATION_COUNT_AFTER_LAYER;
    graph_.nodes.resize(nodeSize);

    graph_.internalTensors.resize(graph_.nodes.size() - 1);

    int nodeId = 0;
    atb::Operation *op = nullptr;

    if (!param_.isEncoder) {
        auto &wordEmbeddingNode = graph_.nodes.at(nodeId++);
        atb::infer::GatherParam wordEmbeddingParam;
        CREATE_OPERATION(wordEmbeddingParam, &op);
        wordEmbeddingNode.operation.reset(op);
        wordEmbeddingNode.inTensors = {&graph_.weightTensors.at(0), &graph_.inTensors.at(0)};
        wordEmbeddingNode.outTensors = {&graph_.internalTensors.at(0)};
    }

    auto &transposeNode = graph_.nodes.at(nodeId++);
    atb::infer::TransposeParam transposeParam = {{1, 0, 2}};
    CREATE_OPERATION(transposeParam, &op);
    transposeNode.operation.reset(op);

    transposeNode.inTensors = {param_.isEncoder?&graph_.inTensors.at(0):&graph_.internalTensors.at(0)};
    transposeNode.outTensors = {&graph_.internalTensors.at(param_.isEncoder?0:1)};

    atb::Tensor *firstInTensor = &graph_.internalTensors.at(param_.isEncoder?0:1);

    for (int layerId = 0; layerId < param_.layerNum; ++layerId) {
        auto &layerNode = graph_.nodes.at(nodeId++);

        Glm6BLayerDecoderFlashAttentionParam opParam;
        opParam.layerNormEps = param_.layerNormEps;
        opParam.headNum = param_.headNum;
        opParam.headDim = param_.headDim;
        opParam.qScale = param_.qScale.at(layerId);
        opParam.qkScale = param_.qkScale.at(layerId);
        opParam.residualAddScale = param_.residualAddScale;
        CreateGlm6BLayerDecoderFlashAttentionOperation(opParam, &op);
        layerNode.operation.reset(op);
        layerNode.inTensors.resize(layerNode.operation->GetInputNum());

        size_t inTensorId = 0;
        layerNode.inTensors.at(inTensorId++) = firstInTensor;
        for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
            layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(
                layerId * WEIGHT_COUNT_PER_LAYER + weightTensorId + WORDEMBEDDINGNODE_WEIGHT_COUNT);
        }
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_POSITIONID);    // positionIdTensor
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_COSTABLE);      // cosTable
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SINTABLE);      // sinTable
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ATTENTIONMASK); // attentionMaskTensor
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PASTKEY);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PASTVALUE);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_TOKENOFFSET);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SEQLEN); // seqLen
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_MAX + layerId);

        layerNode.outTensors = {&graph_.internalTensors.at(param_.operationCountBeforeLayers + layerId)};
        firstInTensor = layerNode.outTensors.at(0);
    }

    auto &finalNormNode = graph_.nodes.at(nodeId++);
    atb::infer::LayerNormParam finalNormParam;
    finalNormParam.layerType = atb::infer::LayerNormParam::LAYER_NORM_NORM;
    finalNormParam.normParam.epsilon = param_.layerNormEps;
    finalNormParam.normParam.beginNormAxis = 2;
    finalNormParam.normParam.beginParamsAxis = 1;
    CREATE_OPERATION(finalNormParam, &op);
    finalNormNode.operation.reset(op);
    const int finalLayerNormWeightTensorId = graph_.weightTensors.size() - FINALNORMNODE_WEIGHT_COUNT - LMHEADNODE_WEIGHT_COUNT;
    const int finalLayerNormBiasTensorId = finalLayerNormWeightTensorId + 1;
    finalNormNode.inTensors = {firstInTensor, &graph_.weightTensors.at(finalLayerNormWeightTensorId),
                               &graph_.weightTensors.at(finalLayerNormBiasTensorId)};
    finalNormNode.outTensors = {&graph_.internalTensors.at(graph_.internalTensors.size() - 2)};
    firstInTensor = finalNormNode.outTensors.at(0);

    auto &linearNode = graph_.nodes.at(nodeId++);
    atb::infer::LinearParam linearParam;
    linearParam.hasBias = false;
    CREATE_OPERATION(linearParam, &op);
    linearNode.operation.reset(op);
    const int finalLmheadWeightTensorId = graph_.weightTensors.size() - LMHEADNODE_WEIGHT_COUNT;
    linearNode.inTensors = {firstInTensor, &graph_.weightTensors.at(finalLmheadWeightTensorId)};
    linearNode.outTensors = {&graph_.internalTensors.at(graph_.internalTensors.size() - 1)};
    firstInTensor = linearNode.outTensors.at(0);

    auto &transpose2Node = graph_.nodes.at(nodeId++);
    CREATE_OPERATION(transposeParam, &op);
    transpose2Node.operation.reset(op);
    transpose2Node.inTensors = {firstInTensor};
    transpose2Node.outTensors = {&graph_.outTensors.at(0)};

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

    ATB_LOG(INFO) << "VisualGlm6BFlashAttentionModel ParseParam tokenOffset:" << tokenOffset_ << ", seqLen:" << seqLen_;

    return atb::NO_ERROR;
}

atb::Status FlashAttentionModel::BindParamHostTensor(uint32_t nodeId)
{
    if (nodeId < param_.operationCountBeforeLayers || nodeId >= param_.operationCountBeforeLayers + param_.layerNum) {
        return atb::NO_ERROR;
    }

    auto &node = graph_.nodes.at(nodeId);

    const uint32_t tokenOffsetTensorId = 19;
    const uint32_t seqLenTensorId = 20;
    node.variantPack.inTensors.at(tokenOffsetTensorId).hostData = tokenOffset_.data();
    node.variantPack.inTensors.at(seqLenTensorId).hostData = seqLen_.data();

    return atb::NO_ERROR;
}
} // namespace visualglm_6b
} // namespace atb_speed
