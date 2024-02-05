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
#include <nlohmann/json.hpp>
#include <atb/atb_infer.h>
#include "atb_speed/log.h"
#include "glm/130b/layer/fusion_parallel_layer.h"
#include "glm/130b/operation/lmhead.h"
#include "fusion_parallel_model.h"

namespace atb_speed {
namespace glm130b {
static const uint64_t WEIGHT_COUNT_PER_LAYER = 12;
static const uint64_t WORDEMBEDDINGNODE_WEIGHT_COUNT = 0;
static const uint64_t FINALNORMNODE_WEIGHT_COUNT = 3; // change to 3 includes final forward weight
static const uint64_t OPERATION_COUNT_BEFORE_LAYER = 2;
static const uint64_t OPERATION_COUNT_AFTER_LAYER = 2; // final norm and lm head
static const uint64_t DEFAULT_BEGIN_NORM_AXIS = 2;

enum InTensorId : int {
    IN_TENSOR_HIDDENSTATES = 0,
    IN_TENSOR_POSITIONID,
    IN_TENSOR_COSTABLE,
    IN_TENSOR_SINTABLE,
    IN_TENSOR_ATTENTIONMASK,
    IN_TENSOR_PASTKEY,
    IN_TENSOR_PASTVALUE,
    IN_TENSOR_TOKENOFFSET,
    IN_TENSOR_SEQLEN,
    IN_TENSOR_LAYERID_BASE,
};

enum InternelTensorId : int {
    INTERNEL_TENSOR_COS = 0,
    INTERNEL_TENSOR_SIN,
    INTERNEL_TENSOR_LAYEROUT_BASE,
};

void FusionParallelModel::Param::FromString(const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);
    layerNum = paramJson["layerNum"].get<int>();
    headNum = paramJson["headNum"].get<int>();
    headDim = paramJson["headDim"].get<int>();
    rank = paramJson["rank"].get<int>();
    rankSize = paramJson["rankSize"].get<int>();
    rankRoot = paramJson["rankRoot"].get<int>();
    residualAddScale = paramJson["residualAddScale"].get<float>();
    layerNormEps = paramJson["layerNormEps"].get<double>();
    backend = paramJson["backend"].get<std::string>();
    coderType = paramJson["coderType"].get<int>();

    for (auto item : paramJson["perm"]) {
        perm.push_back(item.get<int>());
    }
    for (auto item : paramJson["qScale"]) {
        qScale.push_back(item.get<float>());
    }
    for (auto item : paramJson["qkScale"]) {
        qkScale.push_back(item.get<float>());
    }

    ATB_LOG(INFO) << "Glm130BFusionParallelModel param layerNum:" << layerNum << ", headNum:" << headNum
                  << ", headDim:" << headDim << ", rank:" << rank << ", rankSize:" << rankSize
                  << ", rankRoot:" << rankRoot << ", qScale:" << qScale << ", qkScale:" << qkScale
                  << ", residualAddScale:" << residualAddScale << ", layerNormEps:" << layerNormEps
                  << ", backend:" << backend << ", perm:" << perm << ", coderType:" << coderType;
}

FusionParallelModel::FusionParallelModel(const std::string &param) : Model("FusionParallelModel", param)
{
    param_.FromString(param);
}

FusionParallelModel::~FusionParallelModel() {}

uint32_t FusionParallelModel::GetInputNum() const { return graph_.inTensors.size(); }

uint32_t FusionParallelModel::GetOutputNum() const { return graph_.outTensors.size(); }

atb::Status FusionParallelModel::InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
                                            std::vector<atb::TensorDesc> &outTensorDescs)
{
    if (outTensorDescs.size() != GetOutputNum()) {
        return atb::ERROR_INVALID_GRAPH;
    }

    outTensorDescs.at(0) = inTensorDescs.at(0);
    outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
    outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1];
    outTensorDescs.at(0).shape.dims[2] = // 2: 设置第一个张量第三维长度
        graph_.weightTensors.at(graph_.weightTensors.size() - 1).desc.shape.dims[0] * param_.rankSize;
    return atb::NO_ERROR;
}

int64_t FusionParallelModel::BuildGraph()
{
    const int weightTensorSize =
        WORDEMBEDDINGNODE_WEIGHT_COUNT + WEIGHT_COUNT_PER_LAYER * param_.layerNum + FINALNORMNODE_WEIGHT_COUNT;
    graph_.weightTensors.resize(weightTensorSize);

    graph_.inTensors.resize(IN_TENSOR_LAYERID_BASE + param_.layerNum);
    graph_.outTensors.resize(1);

    const int nodeSize = param_.layerNum + OPERATION_COUNT_BEFORE_LAYER + OPERATION_COUNT_AFTER_LAYER;
    graph_.nodes.resize(nodeSize);

    const int internalTensorSize = INTERNEL_TENSOR_LAYEROUT_BASE + param_.layerNum;
    graph_.internalTensors.resize(internalTensorSize);

    int nodeId = 0;
    auto &cosEmbeddingNode = graph_.nodes.at(nodeId++);
    atb::infer::GatherParam ropeCosParam;
    atb::Operation *op = nullptr;
    atb::CreateOperation(ropeCosParam, &op);
    cosEmbeddingNode.operation.reset(op);
    cosEmbeddingNode.inTensors = {&graph_.inTensors.at(IN_TENSOR_COSTABLE), &graph_.inTensors.at(IN_TENSOR_POSITIONID)};
    cosEmbeddingNode.outTensors = {&graph_.internalTensors.at(INTERNEL_TENSOR_COS)};

    auto &sinEmbeddingNode = graph_.nodes.at(nodeId++);
    atb::infer::GatherParam ropeSinParam;
    atb::CreateOperation(ropeSinParam, &op);
    sinEmbeddingNode.operation.reset(op);
    sinEmbeddingNode.inTensors = {&graph_.inTensors.at(IN_TENSOR_SINTABLE), &graph_.inTensors.at(IN_TENSOR_POSITIONID)};
    sinEmbeddingNode.outTensors = {&graph_.internalTensors.at(INTERNEL_TENSOR_SIN)};

    atb::Tensor *firstInTensor = &graph_.inTensors.at(IN_TENSOR_HIDDENSTATES);

    for (int layerId = 0; layerId < param_.layerNum; ++layerId) {
        auto &layerNode = graph_.nodes.at(nodeId++);
        FusionParallelLayerParam opParam;
        opParam.headNum = param_.headNum;
        opParam.headDim = param_.headDim;
        opParam.rank = param_.rank;
        opParam.rankSize = param_.rankSize;
        opParam.rankRoot = param_.rankRoot;
        opParam.residualAddScale = param_.residualAddScale;
        opParam.layerNormEps = param_.layerNormEps;
        opParam.backend = param_.backend;
        opParam.qScale = param_.qScale.at(layerId);
        opParam.qkScale = param_.qkScale.at(layerId);
        opParam.coderType = param_.coderType;
        CreateFusionParallelLayer(opParam, &op);
        layerNode.operation.reset(op);
        layerNode.inTensors.resize(layerNode.operation->GetInputNum());

        size_t inTensorId = 0;
        layerNode.inTensors.at(inTensorId++) = firstInTensor;
        for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
            layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(
                layerId * WEIGHT_COUNT_PER_LAYER + weightTensorId + WORDEMBEDDINGNODE_WEIGHT_COUNT);
        }
        layerNode.inTensors.at(inTensorId++) = &graph_.internalTensors.at(INTERNEL_TENSOR_COS);
        layerNode.inTensors.at(inTensorId++) = &graph_.internalTensors.at(INTERNEL_TENSOR_SIN);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ATTENTIONMASK);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PASTKEY);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PASTVALUE);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SEQLEN);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_TOKENOFFSET);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_LAYERID_BASE + layerId);
        layerNode.outTensors = {&graph_.internalTensors.at(INTERNEL_TENSOR_LAYEROUT_BASE + layerId)};
        firstInTensor = layerNode.outTensors.at(0);
    }

    auto &finalNormNode = graph_.nodes.at(nodeId++);
    atb::infer::LayerNormParam finalNormParam;
    finalNormParam.layerType = atb::infer::LayerNormParam::LAYER_NORM_NORM;
    finalNormParam.normParam.epsilon = param_.layerNormEps;
    finalNormParam.normParam.beginNormAxis = DEFAULT_BEGIN_NORM_AXIS;
    finalNormParam.normParam.beginParamsAxis = 1;
    atb::CreateOperation(finalNormParam, &op);
    finalNormNode.operation.reset(op);
    const int finalLayerNormWeightTensorId = graph_.weightTensors.size() - FINALNORMNODE_WEIGHT_COUNT;
    const int finalLayerNormBiasTensorId = finalLayerNormWeightTensorId + 1;
    finalNormNode.inTensors = {firstInTensor, &graph_.weightTensors.at(finalLayerNormWeightTensorId),
                               &graph_.weightTensors.at(finalLayerNormBiasTensorId)};
    finalNormNode.outTensors = {&graph_.internalTensors.at(internalTensorSize - 1)};

    auto &lmHeadNode = graph_.nodes.at(nodeId++);
    LmHeadParam lmHeadParam;
    lmHeadParam.rank = param_.rank;
    lmHeadParam.rankSize = param_.rankSize;
    lmHeadParam.rankRoot = param_.rankRoot;
    lmHeadParam.backend = param_.backend;
    lmHeadParam.perm = param_.perm;
    CreateLmHead(lmHeadParam, &op);
    lmHeadNode.operation.reset(op);
    const int finalForwardWeightTensorId = graph_.weightTensors.size() - 1;
    lmHeadNode.inTensors = {&graph_.internalTensors.at(internalTensorSize - 1),
                            &graph_.weightTensors.at(finalForwardWeightTensorId)};
    lmHeadNode.outTensors = {&graph_.outTensors.at(0)};

    return atb::NO_ERROR;
}

atb::Status FusionParallelModel::ParseParam(const std::string &param)
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

    ATB_LOG(INFO) << "Glm130BFusionParallelModel ParseParam tokenOffset:" << tokenOffset_ << ", seqLen:" << seqLen_;

    return atb::NO_ERROR;
}

atb::Status FusionParallelModel::BindParamHostTensor(uint32_t nodeId)
{
    if (nodeId < OPERATION_COUNT_BEFORE_LAYER || nodeId >= OPERATION_COUNT_BEFORE_LAYER + param_.layerNum) {
        return atb::NO_ERROR;
    }

    auto &node = graph_.nodes.at(nodeId);

    const uint32_t seqLenTensorId = FusionParallelLayerTensorId::IN_SEQLEN_ID;
    const uint32_t tokenOffsetTensorId = FusionParallelLayerTensorId::IN_TOKENOFFSET_ID;
    node.variantPack.inTensors.at(seqLenTensorId).hostData = seqLen_.data();
    node.variantPack.inTensors.at(tokenOffsetTensorId).hostData = tokenOffset_.data();

    return atb::NO_ERROR;
}
} // namespace glm130b
} // namespace atb_speed