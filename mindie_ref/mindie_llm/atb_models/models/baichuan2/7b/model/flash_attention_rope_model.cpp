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
#include "flash_attention_rope_model.h"

#include "atb/atb_infer.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtype-limits"
#include "nlohmann/json.hpp"
#pragma GCC diagnostic pop

#include "atb_speed/utils/operation_util.h"
#include "models/baichuan2/7b/layer/flash_attention_rope_layer.h"
#include "atb_speed/utils/model_factory.h"
#include "operations/lmhead.h"

namespace atb_speed {
namespace baichuan2_7b {

REGISTER_MODEL(baichuan2_7b, FlashAttentionRopeModel);

const int WEIGHT_COUNT_PER_LAYER = 7;
const int WORDEMBEDDINGNODE_WEIGHT_COUNT = 1;
const int FINALNORMNODE_WEIGHT_COUNT = 1;
const int OUT_LM_HEAD_WEIGHT_COUNT = 1;
const int OPERATION_COUNT_BEFORE_LAYER = 1;
const int INTERMEDIATETENSOR_COUNT_BEFORE_LAYER = 1;
const int OPERATION_COUNT_AFTER_LAYER = 2;

enum InTensorId : int {
    IN_TENSOR_INPUTIDS = 0,
    IN_TENSOR_COSEMBED,
    IN_TENSOR_SINEMBED,
    IN_TENSOR_ATTENTIONMASK,
    IN_TENSOR_PAST_KEY,
    IN_TENSOR_PAST_VALUE,
    IN_TENSOR_TOKENOFFSET,
    IN_TENSOR_SEQLEN,
    IN_HOLDER,
    IN_FINAL_NORM_SLICE_OFFSET,
    IN_TENSOR_MAX, // 10
};

enum OutTensorId : int {
    OUT_TENSOR_HIDDENSTATES = 0,
    OUT_TENSOR_MAX,
};

void FlashAttentionRopeModel::Param::FromString(const std::string &param)
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
        backend = paramJson["backend"];
    }
}

FlashAttentionRopeModel::FlashAttentionRopeModel(const std::string &param) : Model("FlashAttentionModel", param)
{
    param_.FromString(param);
}

FlashAttentionRopeModel::~FlashAttentionRopeModel() = default;

uint32_t FlashAttentionRopeModel::GetInputNum() const { return graph_.inTensors.size(); }

uint32_t FlashAttentionRopeModel::GetOutputNum() const { return graph_.outTensors.size(); }

atb::Status FlashAttentionRopeModel::InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
                                                std::vector<atb::TensorDesc> &outTensorDescs)
{
    if (outTensorDescs.size() != GetOutputNum()) {
        return atb::ERROR_INVALID_GRAPH;
    }
    const int64_t outDim = graph_.weightTensors.at(graph_.weightTensors.size() - 1).desc.shape.dims[0];
    outTensorDescs.at(0) = graph_.weightTensors.at(0).desc;
    outTensorDescs.at(0).shape.dimNum = 3;
    outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
    outTensorDescs.at(0).shape.dims[1] = 1; // inTensorDescs.at(0).shape.dims[1];
    outTensorDescs.at(0).shape.dims[2] = outDim * param_.rankSize;

    return atb::NO_ERROR;
}

int64_t FlashAttentionRopeModel::BuildGraph()
{
    const int weightTensorSize = WORDEMBEDDINGNODE_WEIGHT_COUNT + WEIGHT_COUNT_PER_LAYER * param_.layerNum +
                                 FINALNORMNODE_WEIGHT_COUNT + OUT_LM_HEAD_WEIGHT_COUNT;
    graph_.weightTensors.resize(weightTensorSize);

    graph_.inTensors.resize(IN_TENSOR_MAX + param_.layerNum);
    graph_.outTensors.resize(OUT_TENSOR_MAX);

    const int nodeSize = param_.layerNum + OPERATION_COUNT_BEFORE_LAYER + OPERATION_COUNT_AFTER_LAYER;
    graph_.nodes.resize(nodeSize);

    const int internalTensorSize = graph_.nodes.size() - 1;
    graph_.internalTensors.resize(internalTensorSize);

    int nodeId = 0;
    auto &wordEmbeddingNode = graph_.nodes.at(nodeId++);
    atb::Operation *op = nullptr;

    atb::infer::GatherParam wordEmbeddingParam;
    CREATE_OPERATION(wordEmbeddingParam, &op);
    wordEmbeddingNode.operation.reset(op);
    wordEmbeddingNode.inTensors = {&graph_.weightTensors.at(0), &graph_.inTensors.at(IN_TENSOR_INPUTIDS)};
    wordEmbeddingNode.outTensors = {&graph_.internalTensors.at(0)};

    atb::Tensor *firstInTensor = &graph_.internalTensors.at(0);

    for (int layerId = 0; layerId < param_.layerNum; ++layerId) {
        auto &layerNode = graph_.nodes.at(nodeId++);

        atb_speed::baichuan2_7b::FlashAttentionRopeLayerParam opParam;
        opParam.rmsNormEps = param_.rmsNormEps;
        opParam.headNum = param_.headNum;
        opParam.dk = param_.dk;
        opParam.rank = param_.rank;
        opParam.rankSize = param_.rankSize;
        opParam.backend = param_.backend;
        atb_speed::baichuan2_7b::FlashAttentionRopeLayer(opParam, &op);
        layerNode.operation.reset(op);
        layerNode.inTensors.resize(layerNode.operation->GetInputNum()); // .at 需要resize，直接赋值不需要

        size_t inTensorId = 0;
        layerNode.inTensors.at(inTensorId++) = firstInTensor;
        for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
            layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(
                layerId * WEIGHT_COUNT_PER_LAYER + weightTensorId + WORDEMBEDDINGNODE_WEIGHT_COUNT);
        }
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_COSEMBED);      // cosEmbed
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SINEMBED);      // sinEmbed
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ATTENTIONMASK); // attentionMaskTensor
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PAST_KEY);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PAST_VALUE);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_TOKENOFFSET);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SEQLEN); // seqLen
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_HOLDER);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_MAX + layerId);

        layerNode.outTensors = {&graph_.internalTensors.at(INTERMEDIATETENSOR_COUNT_BEFORE_LAYER + layerId)};

        firstInTensor = layerNode.outTensors.at(0);
    }

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

    auto &lmHeadNode = graph_.nodes.at(nodeId++);
    atb_speed::common::LmHeadParam lmHeadParam;
    lmHeadParam.unpadInputs = !param_.isFA;
    lmHeadParam.gatherAhead = param_.isPrefill;
    lmHeadParam.linearParallelParam.parallelType = atb_speed::common::COLUMN_PARALLEL;
    lmHeadParam.linearParallelParam.tensorParallelInfo.rank = param_.rank;
    lmHeadParam.linearParallelParam.tensorParallelInfo.worldSize = param_.rankSize;
    lmHeadParam.linearParallelParam.tensorParallelInfo.backend = param_.backend;
    LmHead(lmHeadParam, &op);
    lmHeadNode.operation.reset(op);
    const int finalLinearWeightTensorId = graph_.weightTensors.size() - OUT_LM_HEAD_WEIGHT_COUNT;
    lmHeadNode.inTensors = {&graph_.internalTensors.at(finalLayerNormOutTensorId),
                            // shape: [vocabSizePerRank, hiddenSize]
                            &graph_.weightTensors.at(finalLinearWeightTensorId),
                            // LmHead未接入量化，量化权重使用placeholder代替
                            &graph_.inTensors.at(IN_HOLDER), &graph_.inTensors.at(IN_HOLDER),
                            &graph_.inTensors.at(IN_HOLDER), &graph_.inTensors.at(IN_HOLDER),
                            &graph_.inTensors.at(IN_FINAL_NORM_SLICE_OFFSET)};
    // shape: FA: [batchSize, seqLen, vocabSize] PA: [seqLen, vocabSize]
    lmHeadNode.outTensors = {&graph_.outTensors.at(OUT_TENSOR_HIDDENSTATES)};

    return atb::NO_ERROR;
}

atb::Status FlashAttentionRopeModel::ParseParam(const std::string &param)
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

atb::Status FlashAttentionRopeModel::BindParamHostTensor(uint32_t nodeId)
{
    if (nodeId < OPERATION_COUNT_BEFORE_LAYER ||
        nodeId >= static_cast<uint32_t>(OPERATION_COUNT_BEFORE_LAYER + param_.layerNum)) {
        return atb::NO_ERROR;
    }
    auto &node = graph_.nodes.at(nodeId);

    const uint32_t tokenOffsetTensorId = 13;
    const uint32_t seqLenTensorId = 14;

    node.variantPack.inTensors.at(tokenOffsetTensorId).hostData = tokenOffset_.data();
    node.variantPack.inTensors.at(seqLenTensorId).hostData = seqLen_.data();
    ATB_LOG(INFO) << "BindParamHostTensor end";
    return atb::NO_ERROR;
}
} // namespace baichuan2_7b
} // namespace atb_speed