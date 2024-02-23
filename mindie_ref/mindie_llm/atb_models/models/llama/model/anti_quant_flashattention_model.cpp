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
#include "nlohmann/json.hpp"
#include "atb/atb_infer.h"
#include "models/llama/operation/layer_embedding.h"
#include "models/llama/layer/anti_float_layer.h"
#include "models/llama/layer/anti_quant_layer.h"
#include "anti_quant_flashattention_model.h"
#pragma GCC diagnostic ignored "-Wsign-compare"

namespace atb_speed {
namespace llama {
const int WEIGHT_COUNT_PER_LAYER = 25;
const int ROLLBACK_WEIGHT_COUNT_PER_LAYER = 16;
const int OUTPUT_TENSOR_COUNT_BEFORE_KEY = 1;
const int WORDEMBEDDINGNODE_WEIGHT_COUNT = 1;
const int FINALNORMNODE_WEIGHT_COUNT = 1;
const int OUT_LM_HEAD_WEIGHT_COUNT = 1;
const int OPERATION_COUNT_BEFORE_LAYER = 2;
const int INTERMEDIATETENSOR_COUNT_BEFORE_LAYER = 3;
const int OPERATION_COUNT_AFTER_LAYER = 2;
const int MODEL_OUT_DIM_NUM = 3;
const int MODEL_OUT_DIM0 = 0;
const int MODEL_OUT_DIM1 = 1;
const int MODEL_OUT_DIM2 = 2;

enum InTensorId : int {
    IN_TENSOR_HIDDENSTATES = 0,
    IN_TENSOR_POSITIONID,
    IN_TENSOR_COSTABLE,
    IN_TENSOR_SINTABLE,
    IN_TENSOR_ATTENTIONMASK,
    IN_TENSOR_PAST_KEY,
    IN_TENSOR_PAST_VALUE,
    IN_TENSOR_TOKENOFFSET,
    IN_TENSOR_SEQLEN,
    IN_HOLDER,
    IN_TENSOR_MAX, // 9
};

enum OutTensorId : int {
    OUT_TENSOR_HIDDENSTATES = 0,
    OUT_TENSOR_MAX,
};

void AntiQuantFlashAttentionModel::Param::FromString(const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);
    rmsNormEps = paramJson["rmsNormEps"].get<double>();
    headNum = paramJson["headNum"].get<int>();
    dk = paramJson["dk"].get<int>();
    layerNum = paramJson["layerNum"].get<int>();
    rank = paramJson["rank"].get<int>();
    rankSize = paramJson["rankSize"].get<int>();

    for (auto item : paramJson["qkvInputScale"]) {
        qkvInputScale.push_back(item.get<float>());
    }
    for (auto item : paramJson["qkvInputOffset"]) {
        qkvInputOffset.push_back(item.get<int>());
    }
    for (auto item : paramJson["denseInputScale"]) {
        denseInputScale.push_back(item.get<float>());
    }
    for (auto item : paramJson["denseInputOffset"]) {
        denseInputOffset.push_back(item.get<int>());
    }
    for (auto item : paramJson["selfLnInputScale"]) {
        selfLnInputScale.push_back(item.get<float>());
    }
    for (auto item : paramJson["selfLnInputOffset"]) {
        selfLnInputOffset.push_back(item.get<int>());
    }
    for (auto item : paramJson["ffnOutInputScale"]) {
        ffnOutInputScale.push_back(item.get<float>());
    }
    for (auto item : paramJson["ffnOutInputOffset"]) {
        ffnOutInputOffset.push_back(item.get<int>());
    }
    for (auto item : paramJson["floatLayers"]) {
        floatLayers.push_back(item.get<int>());
    }

    ATB_LOG(INFO) << "LLaMA AntiQuantFlashAttentionModel param rmsNormEps:" << rmsNormEps << ", headNum:" << headNum
                  << ", dk:" << dk << ", layerNum:" << layerNum << ", rank:" << rank << ", rankSize:" << rankSize;
}

AntiQuantFlashAttentionModel::AntiQuantFlashAttentionModel(const std::string &param) : Model("AntiQuantFlashAttentionModel", param)
{
    param_.FromString(param);
}

AntiQuantFlashAttentionModel::~AntiQuantFlashAttentionModel() {}

uint32_t AntiQuantFlashAttentionModel::GetInputNum() const { return graph_.inTensors.size(); }

uint32_t AntiQuantFlashAttentionModel::GetOutputNum() const { return graph_.outTensors.size(); }

atb::Status AntiQuantFlashAttentionModel::InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
    std::vector<atb::TensorDesc> &outTensorDescs)
{
    ATB_LOG(INFO) << "Enter LLaMA AntiQuantFlashAttentionModel InferShape";
    if (outTensorDescs.size() != GetOutputNum()) {
        return atb::ERROR_INVALID_GRAPH;
    }
    const int64_t outDim = graph_.weightTensors.at(graph_.weightTensors.size() - 1).desc.shape.dims[0];
    outTensorDescs.at(MODEL_OUT_DIM0) = graph_.weightTensors.at(MODEL_OUT_DIM0).desc;
    outTensorDescs.at(MODEL_OUT_DIM0).shape.dimNum = MODEL_OUT_DIM_NUM;
    outTensorDescs.at(MODEL_OUT_DIM0).shape.dims[MODEL_OUT_DIM0] =
        inTensorDescs.at(MODEL_OUT_DIM0).shape.dims[MODEL_OUT_DIM0];
    outTensorDescs.at(MODEL_OUT_DIM0).shape.dims[MODEL_OUT_DIM1] =
        inTensorDescs.at(MODEL_OUT_DIM0).shape.dims[MODEL_OUT_DIM1];
    outTensorDescs.at(MODEL_OUT_DIM0).shape.dims[MODEL_OUT_DIM2] = outDim;

    ATB_LOG(INFO) << "LLaMA AntiQuantFlashAttentionModel InferShape Success";
    return atb::NO_ERROR;
}

int64_t AntiQuantFlashAttentionModel::BuildGraph()
{
    ATB_LOG(INFO) << "Enter LLaMA AntiQuantFlashAttentionModel BuildGraph";

    const int floatLayerCnt = param_.floatLayers.size();
    const int weightTensorSize = WORDEMBEDDINGNODE_WEIGHT_COUNT +
                                 ROLLBACK_WEIGHT_COUNT_PER_LAYER * floatLayerCnt +
                                 WEIGHT_COUNT_PER_LAYER * (param_.layerNum - floatLayerCnt) +
                                 FINALNORMNODE_WEIGHT_COUNT + OUT_LM_HEAD_WEIGHT_COUNT;
    graph_.weightTensors.resize(weightTensorSize);

    graph_.inTensors.resize(IN_TENSOR_MAX + param_.layerNum);
    graph_.outTensors.resize(OUT_TENSOR_MAX);

    const int nodeSize = param_.layerNum + OPERATION_COUNT_BEFORE_LAYER + OPERATION_COUNT_AFTER_LAYER;
    graph_.nodes.resize(nodeSize);

    const int internalTensorSize = graph_.nodes.size() - 1;
    graph_.internalTensors.resize(internalTensorSize);

    int nodeId = 0;
    int weightOffset = 0;

    auto &wordEmbeddingNode = graph_.nodes.at(nodeId++);
    atb::infer::GatherParam wordEmbeddingParam;
    atb::Operation *op = nullptr;
    CREATE_OPERATION(wordEmbeddingParam, &op);
    wordEmbeddingNode.operation.reset(op);
    wordEmbeddingNode.inTensors = {&graph_.weightTensors.at(weightOffset++), &graph_.inTensors.at(0)};
    wordEmbeddingNode.outTensors = {&graph_.internalTensors.at(0)};

    auto &embeddingNode = graph_.nodes.at(nodeId++);
    atb_speed::llama::LayerEmbeddingParam layerEmbeddingParam;
    atb_speed::llama::LayerEmbedding(layerEmbeddingParam, &op);
    embeddingNode.operation.reset(op);
    embeddingNode.inTensors = {&graph_.inTensors.at(IN_TENSOR_COSTABLE),
                                &graph_.inTensors.at(IN_TENSOR_SINTABLE),
                                &graph_.inTensors.at(IN_TENSOR_POSITIONID)};
    embeddingNode.outTensors = {&graph_.internalTensors.at(1),
                                &graph_.internalTensors.at(2)};

    atb::Tensor *firstInTensor = &graph_.internalTensors.at(0);
    atb::Tensor *cosEmbedInTensor = &graph_.internalTensors.at(1);
    atb::Tensor *sinEmbedInTensor = &graph_.internalTensors.at(2);

    for (int layerId = 0; layerId < param_.layerNum; ++layerId) {
        auto &layerNode = graph_.nodes.at(nodeId++);
        bool isFloatLayer = false;
        if (std::find(param_.floatLayers.begin(), param_.floatLayers.end(), layerId) != param_.floatLayers.end()) {
            isFloatLayer = true;
        }
        if (isFloatLayer) {
            atb_speed::llama::AntiFloatLayerParam modelParamRollback;
            modelParamRollback.rmsNormEps = param_.rmsNormEps;
            modelParamRollback.headNum = param_.headNum;
            modelParamRollback.dk = param_.dk;
            modelParamRollback.model = "llama13b";
            modelParamRollback.rank = param_.rank;
            modelParamRollback.rankSize = param_.rankSize;

            atb_speed::llama::AntiFloatLayer(modelParamRollback, &op);
            layerNode.operation.reset(op);
            layerNode.inTensors.resize(layerNode.operation->GetInputNum());
            layerNode.outTensors.resize(layerNode.operation->GetOutputNum());

            size_t inTensorId = 0;
            layerNode.inTensors.at(inTensorId++) = firstInTensor;
            for (size_t weightTensorId = 0; weightTensorId < ROLLBACK_WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
                layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(weightOffset++);
            }
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_POSITIONID);
            layerNode.inTensors.at(inTensorId++) = cosEmbedInTensor;
            layerNode.inTensors.at(inTensorId++) = sinEmbedInTensor;
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ATTENTIONMASK);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PAST_KEY);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PAST_VALUE);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_TOKENOFFSET);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SEQLEN); // seqLen
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_MAX + layerId);

            layerNode.outTensors = {&graph_.internalTensors.at(INTERMEDIATETENSOR_COUNT_BEFORE_LAYER + layerId)};

            firstInTensor = layerNode.outTensors.at(0);
        } else {
            atb_speed::llama::AntiQuantLayerParam modelParam;
            modelParam.rmsNormEps = param_.rmsNormEps;
            modelParam.headNum = param_.headNum;
            modelParam.dk = param_.dk;
            modelParam.model = "llama13b";
            modelParam.rank = param_.rank;
            modelParam.rankSize = param_.rankSize;
            // 量化适配
            modelParam.qkvInputScale = param_.qkvInputScale[layerId];
            modelParam.qkvInputOffset = param_.qkvInputOffset[layerId];
            modelParam.denseInputScale = param_.denseInputScale[layerId];
            modelParam.denseInputOffset = param_.denseInputOffset[layerId];
            modelParam.selfLnInputScale = param_.selfLnInputScale[layerId];
            modelParam.selfLnInputOffset = param_.selfLnInputOffset[layerId];
            modelParam.ffnOutInputScale = param_.ffnOutInputScale[layerId];
            modelParam.ffnOutInputOffset = param_.ffnOutInputOffset[layerId];

            atb_speed::llama::AntiQuantLayer(modelParam, &op);
            layerNode.operation.reset(op);
            layerNode.inTensors.resize(layerNode.operation->GetInputNum());
            layerNode.outTensors.resize(layerNode.operation->GetOutputNum());

            size_t inTensorId = 0;
            layerNode.inTensors.at(inTensorId++) = firstInTensor;
            for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
                layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(weightOffset++);
            }
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_POSITIONID);
            layerNode.inTensors.at(inTensorId++) = cosEmbedInTensor;
            layerNode.inTensors.at(inTensorId++) = sinEmbedInTensor;
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ATTENTIONMASK);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PAST_KEY);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PAST_VALUE);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_TOKENOFFSET);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SEQLEN); // seqLen
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_HOLDER);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_MAX + layerId);

            layerNode.outTensors = {&graph_.internalTensors.at(INTERMEDIATETENSOR_COUNT_BEFORE_LAYER + layerId)};

            firstInTensor = layerNode.outTensors.at(0);
        }
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

    auto &outLinearNode = graph_.nodes.at(nodeId++);
    atb::infer::LinearParam outLinearParm = {false, false, false};
    CREATE_OPERATION(outLinearParm, &op);
    outLinearNode.operation.reset(op);
    const int finalLinearWeightTensorId = graph_.weightTensors.size() - OUT_LM_HEAD_WEIGHT_COUNT;
    outLinearNode.inTensors = {&graph_.internalTensors.at(finalLayerNormOutTensorId),
                               &graph_.weightTensors.at(finalLinearWeightTensorId)};
    outLinearNode.outTensors = {&graph_.outTensors.at(0)};

    ATB_LOG(INFO) << "LLaMA QuantFlashAttentionModel BuildGraph success";
    return atb::NO_ERROR;
}

atb::Status AntiQuantFlashAttentionModel::ParseParam(const std::string &param)
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

atb::Status AntiQuantFlashAttentionModel::BindParamHostTensor(uint32_t nodeId)
{
    ATB_LOG(INFO) << "BindParamHostTensor";
    if (nodeId < OPERATION_COUNT_BEFORE_LAYER || nodeId >= OPERATION_COUNT_BEFORE_LAYER + param_.layerNum) {
        return atb::NO_ERROR;
    }

    bool isFloatLayer = false;
    size_t layerId = nodeId - OPERATION_COUNT_BEFORE_LAYER;
    if (std::find(param_.floatLayers.begin(), param_.floatLayers.end(), layerId) != param_.floatLayers.end()) {
        isFloatLayer = true;
    }

    const uint32_t floatTokenOffsetTensorId = 23;
    const uint32_t floatSeqLenTensorId = 24;
    const uint32_t quantTokenOffsetTensorId = 32;
    const uint32_t quantSeqLenTensorId = 33;

    auto &node = graph_.nodes.at(nodeId);
    if (isFloatLayer) {
        node.variantPack.inTensors.at(floatTokenOffsetTensorId).hostData = tokenOffset_.data();
        node.variantPack.inTensors.at(floatSeqLenTensorId).hostData = seqLen_.data();
    } else {
        node.variantPack.inTensors.at(quantTokenOffsetTensorId).hostData = tokenOffset_.data();
        node.variantPack.inTensors.at(quantSeqLenTensorId).hostData = seqLen_.data();
    }

    ATB_LOG(INFO) << "BindParamHostTensor end";
    return atb::NO_ERROR;
}

} // namespace llama
} // namespace atb_speed