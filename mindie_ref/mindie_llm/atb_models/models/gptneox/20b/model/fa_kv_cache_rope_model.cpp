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
#include "models/gptneox/20b/model/fa_kv_cache_rope_model.h"
#include "atb/atb_infer.h"
#include "atb_speed/log.h"
#include "models/gptneox/20b/layer/flashattention_kvcache_rope_layer.h"
#include "models/gptneox/20b/layer/embedding_layer.h"
#include "models/gptneox/20b/layer/flashattention_kvcache_layer.h"
#include "layers/parallel_layer_v2.h"
#include "nlohmann/json.hpp"
#include "operations/lmhead.h"
#include "atb_speed/utils/model_factory.h"
#include "operations/lmhead.h"

namespace atb_speed {
namespace gptneox_20b {

REGISTER_MODEL(gptneox_20b, FaKvCacheRopeModel);

const int WEIGHT_COUNT_PER_LAYER = 12;
const int WORDEMBEDDINGNODE_WEIGHT_COUNT = 1;
const int FINALNORMNODE_WEIGHT_COUNT = 2;
const int OUT_LM_HEAD_WEIGHT_COUNT = 1;
const int OPERATION_COUNT_BEFORE_LAYER = 1;
const int INTERMEDIATETENSOR_COUNT_BEFORE_LAYER = 3;
const int OPERATION_COUNT_AFTER_LAYER = 2;
const int OUT_TENSOR_DIM_NUM = 3;
const int LAYER_NORM_AXIS_NUM = 2;

enum InTensorId : int {
    IN_TENSOR_INPUTIDS = 0,
    IN_TENSOR_POSITIONID,
    IN_TENSOR_COSTABLE,
    IN_TENSOR_SINTABLE,
    IN_TENSOR_ATTENTIONMASK,
    IN_TENSOR_KEYCACHE,
    IN_TENSOR_VALUECACHE,
    IN_TENSOR_TOKENOFFSET,
    IN_TENSOR_SEQLEN,
    IN_HOLDER,
    IN_FINAL_NORM_SLICE_OFFSET,
    IN_TENSOR_MAX
};

enum OutTensorId : int {
    OUT_TENSOR_HIDDENSTATES = 0,
    OUT_TENSOR_MAX,
};

void FaKvCacheRopeModel::Param::FromString(const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);
    layerNormEps = paramJson["layerNormEps"].get<float>();
    headNum = paramJson["headNum"].get<int>();
    dk = paramJson["dk"].get<int>();
    layerNum = paramJson["layerNum"].get<int>();
    rotaryPct = paramJson["rotaryPct"].get<float>();
    qScale = paramJson["qScale"].get<float>();

    if (paramJson.contains("isPrefill")) {
        isPrefill = paramJson["isPrefill"].get<bool>();
    }
    if (paramJson.contains("rank")) {
        rank = paramJson["rank"].get<int>();
    }
    if (paramJson.contains("rankSize")) {
        rankSize = paramJson["rankSize"].get<int>();
    }
    if (paramJson.contains("qkScale")) {
        qkScale = paramJson["qkScale"].get<float>();
    }
    if (paramJson.contains("backend")) {
        backend = paramJson["backend"];
    }

    ATB_LOG(INFO) << "GptNeox20BModel param layerNormEps:" << layerNormEps << ", headNum:" << headNum << ", dk:" <<
        dk << ", layerNum:" << layerNum << ", rotaryPct:" << rotaryPct;
}

FaKvCacheRopeModel::FaKvCacheRopeModel(const std::string &param) : Model("GptNeoX_20B_MODEL", param)
{
    param_.FromString(param);
}

FaKvCacheRopeModel::~FaKvCacheRopeModel() {}

uint32_t FaKvCacheRopeModel::GetInputNum() const
{
    return graph_.inTensors.size();
}

uint32_t FaKvCacheRopeModel::GetOutputNum() const
{
    return graph_.outTensors.size();
}

atb::Status FaKvCacheRopeModel::InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
    std::vector<atb::TensorDesc> &outTensorDescs)
{
    if (outTensorDescs.size() != GetOutputNum()) {
        return atb::ERROR_INVALID_GRAPH;
    }

    const int64_t outTensorLastDimSize = graph_.weightTensors.at(graph_.weightTensors.size() - 1).desc.shape.dims[0];
    outTensorDescs.at(0) = graph_.weightTensors.at(0).desc;
    outTensorDescs.at(0).shape.dimNum = OUT_TENSOR_DIM_NUM;
    outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
    outTensorDescs.at(0).shape.dims[1] = 1;
    outTensorDescs.at(0).shape.dims[2] = outTensorLastDimSize * param_.rankSize;

    return atb::NO_ERROR;
}

int64_t FaKvCacheRopeModel::BuildGraph()
{
    const int weightTensorSize = WORDEMBEDDINGNODE_WEIGHT_COUNT + WEIGHT_COUNT_PER_LAYER * param_.layerNum +
        FINALNORMNODE_WEIGHT_COUNT + OUT_LM_HEAD_WEIGHT_COUNT;
    graph_.weightTensors.resize(weightTensorSize);

    graph_.inTensors.resize(IN_TENSOR_MAX + param_.layerNum);
    graph_.outTensors.resize(OUT_TENSOR_MAX);

    const int nodeSize = param_.layerNum + OPERATION_COUNT_BEFORE_LAYER + OPERATION_COUNT_AFTER_LAYER;
    ATB_LOG(INFO) << "GptNeoX_20B_MODEL nodeSize is " << nodeSize;
    graph_.nodes.resize(nodeSize);

    const uint32_t internalTensorSize = (INTERMEDIATETENSOR_COUNT_BEFORE_LAYER - 1) + graph_.nodes.size() - 1;
    graph_.internalTensors.resize(internalTensorSize);

    int nodeId = 0;
    auto &embeddingNode = graph_.nodes.at(nodeId++);
    atb_speed::gptneox_20b::EmbeddingLayerParam embeddingLayerParam;
    atb::Operation *op = nullptr;
    atb_speed::gptneox_20b::EmbeddingLayer(embeddingLayerParam, &op);
    embeddingNode.operation.reset(op);
    embeddingNode.inTensors = { &graph_.weightTensors.at(0), &graph_.inTensors.at(IN_TENSOR_INPUTIDS),
        &graph_.inTensors.at(IN_TENSOR_COSTABLE), &graph_.inTensors.at(IN_TENSOR_SINTABLE),
        &graph_.inTensors.at(IN_TENSOR_POSITIONID) };
    embeddingNode.outTensors = { &graph_.internalTensors.at(0), &graph_.internalTensors.at(1),
        &graph_.internalTensors.at(2) };

    atb::Tensor *firstInTensor = &graph_.internalTensors.at(0);
    atb::Tensor *cosEmbedTensor = &graph_.internalTensors.at(1);
    atb::Tensor *sinEmbedTensor = &graph_.internalTensors.at(2);

    for (int layerId = 0; layerId < param_.layerNum; ++layerId) {
        auto &layerNode = graph_.nodes.at(nodeId++);

        atb_speed::gptneox_20b::FlashAttentionKvCacheRopeParam opParam;
        opParam.layerNormEps = param_.layerNormEps;
        opParam.headNum = param_.headNum;
        opParam.dk = param_.dk;
        opParam.rotaryPct = param_.rotaryPct;
        opParam.model = "gptneox_20b";
        opParam.isPrefill = param_.isPrefill;
        opParam.qScale = param_.qScale;
        opParam.qkScale = param_.qkScale;
        opParam.rank = param_.rank;
        opParam.backend = param_.backend;
        opParam.rankSize = param_.rankSize;
        atb_speed::gptneox_20b::FlashAttentionKvCacheRopeLayer(opParam, &op);
        layerNode.operation.reset(op);
        layerNode.inTensors.resize(layerNode.operation->GetInputNum());

        size_t inTensorId = 0;
        layerNode.inTensors.at(inTensorId++) = firstInTensor;
        for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
            layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(layerId * WEIGHT_COUNT_PER_LAYER +
                weightTensorId + WORDEMBEDDINGNODE_WEIGHT_COUNT);
        }
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_POSITIONID);    // positionIdTensor
        layerNode.inTensors.at(inTensorId++) = cosEmbedTensor;                                // cosEmbed
        layerNode.inTensors.at(inTensorId++) = sinEmbedTensor;                                // sinEmbed
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ATTENTIONMASK); // attentionMaskTensor
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_KEYCACHE);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_VALUECACHE);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_TOKENOFFSET);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SEQLEN);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_HOLDER);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_MAX + layerId);

        layerNode.outTensors = { &graph_.internalTensors.at(INTERMEDIATETENSOR_COUNT_BEFORE_LAYER + layerId) };

        firstInTensor = layerNode.outTensors.at(0);
    }

    auto &finalNormNode = graph_.nodes.at(nodeId++);
    atb::infer::LayerNormParam finalNormParam;
    finalNormParam.layerType = atb::infer::LayerNormParam::LayerNormType::LAYER_NORM_NORM;
    finalNormParam.normParam.epsilon = param_.layerNormEps;
    finalNormParam.normParam.beginNormAxis = LAYER_NORM_AXIS_NUM;
    finalNormParam.normParam.beginParamsAxis = LAYER_NORM_AXIS_NUM;
    CREATE_OPERATION(finalNormParam, &op);
    finalNormNode.operation.reset(op);
    const int finalLayerNormWeightTensorId =
        graph_.weightTensors.size() - FINALNORMNODE_WEIGHT_COUNT - OUT_LM_HEAD_WEIGHT_COUNT;
    const int finalLayerNormBiasTensorId =
        graph_.weightTensors.size() - (FINALNORMNODE_WEIGHT_COUNT - 1) - OUT_LM_HEAD_WEIGHT_COUNT;
    const int finalLayerNormOutTensorId = internalTensorSize - 1;
    finalNormNode.inTensors = { firstInTensor, &graph_.weightTensors.at(finalLayerNormWeightTensorId),
        &graph_.weightTensors.at(finalLayerNormBiasTensorId) };
    finalNormNode.outTensors = { &graph_.internalTensors.at(finalLayerNormOutTensorId) };

    auto &lmHeadNode = graph_.nodes.at(nodeId++);
    atb_speed::common::LmHeadParam lmHeadParam;
    lmHeadParam.unpadInputs = !param_.isFA;
    lmHeadParam.gatherAhead = param_.isPrefill;
    if (param_.rankSize > 1) {
        lmHeadParam.linearParallelParam.parallelType = atb_speed::common::COLUMN_PARALLEL;
        lmHeadParam.linearParallelParam.tensorParallelInfo.rank = param_.rank;
        lmHeadParam.linearParallelParam.tensorParallelInfo.worldSize = param_.rankSize;
        lmHeadParam.linearParallelParam.tensorParallelInfo.backend = param_.backend;
    }
    LmHead(lmHeadParam, &op);
    lmHeadNode.operation.reset(op);
    const int finalLinearWeightTensorId = graph_.weightTensors.size() - OUT_LM_HEAD_WEIGHT_COUNT;
    lmHeadNode.inTensors = {&graph_.internalTensors.at(finalLayerNormOutTensorId),
                            // shape: [vocabSizePerRank, hiddenSize]
                            &graph_.weightTensors.at(finalLinearWeightTensorId),
                            // LmHead未接入量化，量化权重使用placeholder代替
                            &graph_.inTensors.at(IN_HOLDER), &graph_.inTensors.at(IN_HOLDER),
                            &graph_.inTensors.at(IN_HOLDER), &graph_.inTensors.at(IN_FINAL_NORM_SLICE_OFFSET)};
    // shape: FA: [batchSize, seqLen, vocabSize] PA: [seqLen, vocabSize]
    lmHeadNode.outTensors = {&graph_.outTensors.at(OUT_TENSOR_HIDDENSTATES)};

    return atb::NO_ERROR;
}

atb::Status FaKvCacheRopeModel::ParseParam(const std::string &param)
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

    ATB_LOG(INFO) << "GptNeox_20B FaKvCacheModel ParseParam tokenOffset:" << tokenOffset_ << ", seqLen:" << seqLen_;

    return atb::NO_ERROR;
}

atb::Status FaKvCacheRopeModel::BindParamHostTensor(uint32_t nodeId)
{
    if (nodeId < OPERATION_COUNT_BEFORE_LAYER || nodeId >= static_cast<uint32_t>(OPERATION_COUNT_BEFORE_LAYER + param_.layerNum)) {
        return atb::NO_ERROR;
    }

    auto &node = graph_.nodes.at(nodeId);
    const uint32_t tokenOffsetTensorId = 19;
    const uint32_t seqLenTensorId = 20;
    node.variantPack.inTensors.at(tokenOffsetTensorId).hostData = tokenOffset_.data();
    node.variantPack.inTensors.at(seqLenTensorId).hostData = seqLen_.data();

    return atb::NO_ERROR;
}
} // namespace gptneox_20b
} // namespace atb_speed