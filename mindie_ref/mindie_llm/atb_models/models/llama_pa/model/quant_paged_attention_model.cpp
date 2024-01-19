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
#include "quant_paged_attention_model.h"

#include "atb/atb_infer.h"
#include "nlohmann/json.hpp"

#include "parallel_lmhead.h"
#include "models/llama_pa/layer/anti_quant_paged_attention_layer.h"
#include "models/llama_pa/layer/anti_float_paged_attention_layer.h"

namespace atb_speed {
namespace llama_pa {
const int QUANT_WEIGHT_COUNT_PER_LAYER = 25;
const int FLOAT_WEIGHT_COUNT_PER_LAYER = 16;
const int WORDEMBEDDINGNODE_WEIGHT_COUNT = 1;
const int FINALNORMNODE_WEIGHT_COUNT = 1;
const int OUT_LM_HEAD_WEIGHT_COUNT = 1;
const int OPERATION_COUNT_BEFORE_LAYER = 1;
const int INTERMEDIATETENSOR_COUNT_BEFORE_LAYER = 1;
const int OPERATION_COUNT_AFTER_LAYER = 2;

enum QuantPAModelInTensorId : int {
    IN_TENSOR_INPUTIDS = 0,
    IN_TENSOR_POSITIONID,
    IN_TENSOR_COSEMBED,
    IN_TENSOR_SINEMBED,
    IN_TENSOR_ATTENTIONMASK,
    IN_TENSOR_BLOCK_TABLES,
    IN_TENSOR_SLOTS,
    IN_TENSOR_INPUT_LENGTHS,
    IN_TENSOR_LOGTIS_INDICES,
    IN_HOLDER,
    IN_TENSOR_MAX,
};

enum QuantPAModelOutTensorId : int {
    OUT_TENSOR_HIDDENSTATES = 0,
    OUT_TENSOR_MAX,
};

int64_t QuantPAModel::Param::FromString(const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);
    rmsNormEps = paramJson["rmsNormEps"].get<double>();
    headNum = paramJson["headNum"].get<int>();
    dk = paramJson["dk"].get<int>();
    layerNum = paramJson["layerNum"].get<int>();
    if (paramJson.contains("transposedWeight")) {
        transposedWeight = paramJson["transposedWeight"].get<bool>();
    }
    if (paramJson.contains("isPrefill")) {
        isPrefill = paramJson["isPrefill"].get<bool>();
    }
    if (paramJson.contains("rank")) {
        rank = paramJson["rank"].get<int>();
    }
    if (paramJson.contains("rankSize")) {
        rankSize = paramJson["rankSize"].get<int>();
    }
    if (paramJson.contains("backend")) {
        backend = paramJson["backend"];
    }
    if (paramJson.contains("isLmHeadParallel")) {
        isLmHeadParallel = paramJson["isLmHeadParallel"].get<bool>();
    }
    if (paramJson.contains("isBF16")) {
        isBF16 = paramJson["isBF16"].get<bool>();
    }
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
    if (headNum == 0) {
        ATB_LOG(ERROR) << "param.headNum is 0, please input a correct value";
        return atb::ERROR_INVALID_PARAM;
    }
    if (dk == 0) {
        ATB_LOG(ERROR) << "param.dk is 0, please input a correct value";
        return atb::ERROR_INVALID_PARAM;
    }
    ATB_LOG(INFO) << "Llama QuantPAModel param rmsNormEps:" << rmsNormEps << ", headNum:" << headNum << ", dk:" << dk
                  << ", layerNum:" << layerNum << ", transposedWeight:" << transposedWeight << ", rank:" << rank
                  << ", rankSize:" << rankSize << ", backend: " << backend << ", isLmHeadParallel:" << isLmHeadParallel
                  << ", isBF16:" << isBF16;
    return atb::NO_ERROR;
}

QuantPAModel::QuantPAModel(const std::string &param) : Model("LlamaQuantPAModel", param)
{
    param_.FromString(param);
    modelName_ += param_.isPrefill ? "_Prefill" : "_Decoder";
}

QuantPAModel::~QuantPAModel() {}

uint32_t QuantPAModel::GetInputNum() const { return graph_.inTensors.size(); }

uint32_t QuantPAModel::GetOutputNum() const { return graph_.outTensors.size(); }

atb::Status QuantPAModel::InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
                                     std::vector<atb::TensorDesc> &outTensorDescs)
{
    if (outTensorDescs.size() != GetOutputNum()) {
        return atb::ERROR_INVALID_GRAPH;
    }

    const int64_t outDim = graph_.weightTensors.at(graph_.weightTensors.size() - 1).desc.shape.dims[0];
    outTensorDescs.at(0) = graph_.weightTensors.at(0).desc;
    auto outDimNum = inTensorDescs.at(0).shape.dimNum + 1;
    for (int i = 0; i < outDimNum - 1; i++) {
        outTensorDescs.at(0).shape.dims[i] = inTensorDescs.at(0).shape.dims[i];
    }
    if (param_.isLmHeadParallel) {
        outTensorDescs.at(0).shape.dims[outDimNum - 1] = outDim * param_.rankSize;
    } else {
        outTensorDescs.at(0).shape.dims[outDimNum - 1] = outDim;
    }

    if (param_.isPrefill) {
        outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(IN_TENSOR_LOGTIS_INDICES).shape.dims[0];
    }

    return atb::NO_ERROR;
}

int64_t QuantPAModel::BuildGraph()
{
    const int floatLayerCnt = param_.floatLayers.size();
    const int weightTensorSize = WORDEMBEDDINGNODE_WEIGHT_COUNT +
                        FLOAT_WEIGHT_COUNT_PER_LAYER * floatLayerCnt +
                        QUANT_WEIGHT_COUNT_PER_LAYER * (param_.layerNum - floatLayerCnt) +
                        FINALNORMNODE_WEIGHT_COUNT + OUT_LM_HEAD_WEIGHT_COUNT;
    ATB_LOG(INFO) << "weightTensorSize is: " << weightTensorSize;
    
    graph_.weightTensors.resize(weightTensorSize);
    graph_.kCacheTensors.resize(param_.layerNum);
    graph_.vCacheTensors.resize(param_.layerNum);
    graph_.inTensors.resize(IN_TENSOR_MAX);
    graph_.outTensors.resize(OUT_TENSOR_MAX);

    const int nodeSize = param_.layerNum + OPERATION_COUNT_BEFORE_LAYER + OPERATION_COUNT_AFTER_LAYER;
    graph_.nodes.resize(nodeSize);
    ATB_LOG(INFO) << "LlamaQuantPAModel nodeSize is " << nodeSize;

    const int internalTensorSize = graph_.nodes.size() - 1;
    graph_.internalTensors.resize(internalTensorSize);

    int nodeId = 0;
    int weightOffset = 0;
    auto &wordEmbeddingNode = graph_.nodes.at(nodeId++);
    atb::infer::GatherParam wordEmbeddingParam;
    atb::Operation *op = nullptr;
    CREATE_OPERATION(wordEmbeddingParam, &op);
    wordEmbeddingNode.operation.reset(op);
    wordEmbeddingNode.inTensors = {&graph_.weightTensors.at(weightOffset++), &graph_.inTensors.at(IN_TENSOR_INPUTIDS)};
    wordEmbeddingNode.outTensors = {&graph_.internalTensors.at(0)};

    atb::Tensor *firstInTensor = &graph_.internalTensors.at(0);

    for (int layerId = 0; layerId < param_.layerNum; ++layerId) {
        auto &layerNode = graph_.nodes.at(nodeId++);
        bool isFloatLayer = false;
        if (std::find(param_.floatLayers.begin(), param_.floatLayers.end(), layerId) != param_.floatLayers.end()) {
            isFloatLayer = true;
        }
        if (isFloatLayer) {
            ATB_LOG(INFO) << "Enter Float Layer, LayerId is: " << layerId;
            AntiPALayerParam opParam;
            opParam.rmsNormEps = param_.rmsNormEps;
            opParam.headNum = param_.headNum;
            opParam.dk = param_.dk;
            opParam.transposedWeight = param_.transposedWeight;
            opParam.model = "llama";
            opParam.isPrefill = param_.isPrefill;
            opParam.rank = param_.rank;
            opParam.rankSize = param_.rankSize;
            opParam.backend = param_.backend;
            opParam.isBF16 = param_.isBF16;
            AntiPALayer(opParam, &op);
            layerNode.operation.reset(op);
            layerNode.inTensors.resize(layerNode.operation->GetInputNum());

            size_t inTensorId = 0;
            layerNode.inTensors.at(inTensorId++) = firstInTensor;
            for (size_t weightTensorId = 0; weightTensorId < FLOAT_WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
                layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(weightOffset++);
            }
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_POSITIONID);    // positionIdTensor
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_COSEMBED);      // cosEmbed
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SINEMBED);      // sinEmbed
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ATTENTIONMASK); // attentionMaskTensor
            layerNode.inTensors.at(inTensorId++) = &graph_.kCacheTensors.at(layerId);
            layerNode.inTensors.at(inTensorId++) = &graph_.vCacheTensors.at(layerId);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_BLOCK_TABLES);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SLOTS);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_INPUT_LENGTHS);

            layerNode.outTensors = {&graph_.internalTensors.at(INTERMEDIATETENSOR_COUNT_BEFORE_LAYER + layerId)};

            firstInTensor = layerNode.outTensors.at(0);
        } else {
            ATB_LOG(INFO) << "Enter Quant Layer, LayerId: " << layerId;
            QuantPALayerParam opParam;
            opParam.rmsNormEps = param_.rmsNormEps;
            opParam.headNum = param_.headNum;
            opParam.dk = param_.dk;
            opParam.transposedWeight = param_.transposedWeight;
            opParam.model = "llama";
            opParam.isPrefill = param_.isPrefill;
            opParam.rank = param_.rank;
            opParam.rankSize = param_.rankSize;
            opParam.backend = param_.backend;
            opParam.isBF16 = param_.isBF16;

            opParam.qkvInputScale = param_.qkvInputScale[layerId];
            opParam.qkvInputOffset = param_.qkvInputOffset[layerId];
            opParam.denseInputScale = param_.denseInputScale[layerId];
            opParam.denseInputOffset = param_.denseInputOffset[layerId];
            opParam.selfLnInputScale = param_.selfLnInputScale[layerId];
            opParam.selfLnInputOffset = param_.selfLnInputOffset[layerId];
            opParam.ffnOutInputScale = param_.ffnOutInputScale[layerId];
            opParam.ffnOutInputOffset = param_.ffnOutInputOffset[layerId];

            QuantPALayer(opParam, &op);
            layerNode.operation.reset(op);
            layerNode.inTensors.resize(layerNode.operation->GetInputNum());
            layerNode.outTensors.resize(layerNode.operation->GetOutputNum());

            size_t inTensorId = 0;
            layerNode.inTensors.at(inTensorId++) = firstInTensor;
            for (size_t weightTensorId = 0; weightTensorId < QUANT_WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
                layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(weightOffset++);
            }
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_POSITIONID);    // positionIdTensor
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_COSEMBED);      // cosEmbed
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SINEMBED);      // sinEmbed
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ATTENTIONMASK); // attentionMaskTensor
            layerNode.inTensors.at(inTensorId++) = &graph_.kCacheTensors.at(layerId);
            layerNode.inTensors.at(inTensorId++) = &graph_.vCacheTensors.at(layerId);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_BLOCK_TABLES);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SLOTS);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_INPUT_LENGTHS);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_HOLDER);

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

    auto &lmHeadNode = graph_.nodes.at(nodeId++);
    atb_speed::common::ParallelLmHeadParam lmHeadParam;
    if (param_.isLmHeadParallel) {
        lmHeadParam.rank = param_.rank;
        lmHeadParam.rankSize = param_.rankSize;
    }
    lmHeadParam.unpadInputs = true;
    lmHeadParam.gatherAhead = param_.isPrefill;
    lmHeadParam.backend = param_.backend;
    ParallelLmHead(lmHeadParam, &op);
    lmHeadNode.operation.reset(op);
    const int finalLinearWeightTensorId = graph_.weightTensors.size() - OUT_LM_HEAD_WEIGHT_COUNT;
    if (param_.isPrefill) {
        lmHeadNode.inTensors = {&graph_.internalTensors.at(finalLayerNormOutTensorId),
                                &graph_.weightTensors.at(finalLinearWeightTensorId),
                                &graph_.inTensors.at(IN_TENSOR_LOGTIS_INDICES)};
    } else {
        lmHeadNode.inTensors = {&graph_.internalTensors.at(finalLayerNormOutTensorId),
                                &graph_.weightTensors.at(finalLinearWeightTensorId)};
    }
    lmHeadNode.outTensors = {&graph_.outTensors.at(0)};
    return atb::NO_ERROR;
}

atb::Status QuantPAModel::ParseParam(const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);

    seqLen_.clear();
    for (auto item : paramJson["seqLen"]) {
        seqLen_.push_back(item.get<int>());
    }

    return atb::NO_ERROR;
}

atb::Status QuantPAModel::BindParamHostTensor(uint32_t nodeId)
{
    if (nodeId < OPERATION_COUNT_BEFORE_LAYER || nodeId >= OPERATION_COUNT_BEFORE_LAYER + param_.layerNum) {
        return atb::NO_ERROR;
    }
    
    bool isFloatLayer = false;
    size_t layerId = nodeId - OPERATION_COUNT_BEFORE_LAYER;
    if (std::find(param_.floatLayers.begin(), param_.floatLayers.end(), layerId) != param_.floatLayers.end()) {
        isFloatLayer = true;
    }

    auto &node = graph_.nodes.at(nodeId);
    const uint32_t floatSeqLenTensorId = 25;
    const uint32_t quantSeqLenTensorId = 34;

    if (isFloatLayer) {
        node.variantPack.inTensors.at(floatSeqLenTensorId).hostData = seqLen_.data();
    } else {
        node.variantPack.inTensors.at(quantSeqLenTensorId).hostData = seqLen_.data();
    }
    ATB_LOG(INFO) << "BindParamHostTensor end";
    return atb::NO_ERROR;
}
} // namespace llama_pa
} // namespace atb_speed