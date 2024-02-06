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
#include "fusion_pa_model_w8a8.h"
#include "atb/atb_infer.h"
#include "atb_speed/log.h"
#include "models/llama2/70b/layer/fusion_pa_layer_w8a8.h"
#include "models/llama2/70b/operation/llama_lmhead.h"
#include "models/llama2/70b/operation/pa_layer_embedding.h"
#include "nlohmann/json.hpp"
#include "parallel_lmhead.h"

namespace atb_speed {
namespace llama2_70b {
const int WEIGHT_COUNT_PER_LAYER = 30;
const int OPERATION_COUNT_BEFORE_LAYER = 1;
const int OPERATION_COUNT_AFTER_LAYER = 2;
const int WEIGHT_COUNT_BEFORE_LAYER = 1;
const int FINALNORMNODE_WEIGHT_COUNT = 1;
const int OUT_LM_HEAD_WEIGHT_COUNT = 1;
const int INTERMEDIATETENSOR_COUNT_BEFORE_LAYER = 3;

enum InTensorId : int {
    IN_TENSOR_INPUTIDS = 0,  // [1, 20, 8192]
    IN_TENSOR_POSITIONID,    // [1, 20]
    IN_TENSOR_COSTABLE,      // [4096, 128]
    IN_TENSOR_SINTABLE,      // [4096, 128]
    IN_TENSOR_ATTENTIONMASK, // [4096, 4096]
    IN_TENSOR_BLOCK_TABLES,
    IN_TENSOR_SLOTS,
    IN_TENSOR_SEQLEN,
    IN_TENSOR_LOGTIS_INDICES,
    IN_TENSOR_MAX,
};

enum OutTensorId : int {
    OUT_TENSOR_HIDDENSTATES = 0,
    OUT_TENSOR_MAX,
};

void FusionPAModelW8A8::Param::FromString(const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);
    rmsNormEps = paramJson["rmsNormEps"].get<double>();
    headNum = paramJson["headNum"].get<int>();
    dk = paramJson["dk"].get<int>();
    layerNum = paramJson["layerNum"].get<int>();
    rank = paramJson["rank"].get<int>();
    rankSize = paramJson["rankSize"].get<int>();
    numHeadsPerPartition = paramJson["numHeadsPerPartition"].get<int>();
    if (paramJson.contains("isPrefill")) {
        isPrefill = paramJson["isPrefill"].get<bool>();
    }
    if (paramJson.contains("transposedWeight")) {
        transposedWeight = paramJson["transposedWeight"].get<bool>();
    }
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
    ATB_LOG(INFO) << "Llama2_70B_FusionPAModelW8A8 param rmsNormEps:" << rmsNormEps << ", headNum:" << headNum
                  << ", dk:" << dk << ", layerNum:" << layerNum << ", transposedWeight:" << transposedWeight
                  << ", rank:" << rank << ", rankSize:" << rankSize
                  << ", numHeadsPerPartition:" << numHeadsPerPartition;
}

FusionPAModelW8A8::FusionPAModelW8A8(const std::string &param) : Model("FusionPAModelW8A8", param)
{
    param_.FromString(param);
    modelName_ += param_.isPrefill ? "_Prefill" : "_Decoder";
}

FusionPAModelW8A8::~FusionPAModelW8A8() = default;

uint32_t FusionPAModelW8A8::GetInputNum() const { return graph_.inTensors.size(); }

uint32_t FusionPAModelW8A8::GetOutputNum() const { return graph_.outTensors.size(); }

atb::Status FusionPAModelW8A8::InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
                                          std::vector<atb::TensorDesc> &outTensorDescs)
{
    ATB_LOG(INFO) << "Enter LlamaFusionPAModelW8A8 InferShape";
    if (outTensorDescs.size() != GetOutputNum()) {
        return atb::ERROR_INVALID_GRAPH;
    }

    const int64_t outDim = graph_.weightTensors.at(graph_.weightTensors.size() - 1).desc.shape.dims[0];
    outTensorDescs.at(0) = graph_.weightTensors.at(0).desc;
    auto outDimNum = inTensorDescs.at(0).shape.dimNum + 1;
    for (int i = 0; i < outDimNum - 1; i++) {
        outTensorDescs.at(0).shape.dims[i] = inTensorDescs.at(0).shape.dims[i];
    }
    outTensorDescs.at(0).shape.dims[outDimNum - 1] = outDim * param_.rankSize;

    // change first dim
    if (param_.isPrefill) {
        outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(IN_TENSOR_LOGTIS_INDICES).shape.dims[0];
    }

    return atb::NO_ERROR;
}

int64_t FusionPAModelW8A8::BuildGraph()
{
    ATB_LOG(INFO) << "Enter EncoderModel BuildGraph";
    const int weightTensorSize = WEIGHT_COUNT_BEFORE_LAYER + WEIGHT_COUNT_PER_LAYER * param_.layerNum +
                                 FINALNORMNODE_WEIGHT_COUNT + OUT_LM_HEAD_WEIGHT_COUNT;
    graph_.weightTensors.resize(weightTensorSize);

    graph_.kCacheTensors.resize(param_.layerNum);
    graph_.vCacheTensors.resize(param_.layerNum);

    graph_.inTensors.resize(IN_TENSOR_MAX);
    graph_.outTensors.resize(OUT_TENSOR_MAX);

    const int nodeSize = OPERATION_COUNT_BEFORE_LAYER + param_.layerNum + OPERATION_COUNT_AFTER_LAYER;
    ATB_LOG(INFO) << "FusionPAModelW8A8 nodeSize is " << nodeSize;
    graph_.nodes.resize(nodeSize);

    const int internalTensorSize = graph_.nodes.size() + 1;
    graph_.internalTensors.resize(internalTensorSize);

    int nodeId = 0;
    atb::Operation *op = nullptr;

    auto &embeddingNode = graph_.nodes.at(nodeId++);
    atb_speed::llama2_70b::PALayerEmbeddingParam PALayerEmbeddingParam;
    PALayerEmbeddingParam.rank = param_.rank;
    PALayerEmbeddingParam.rankSize = param_.rankSize;
    PALayerEmbeddingParam.backend = param_.backend;
    atb_speed::llama2_70b::PALayerEmbedding(PALayerEmbeddingParam, &op);
    embeddingNode.operation.reset(op);
    embeddingNode.inTensors = {&graph_.weightTensors.at(0), &graph_.inTensors.at(IN_TENSOR_INPUTIDS),
                               &graph_.inTensors.at(IN_TENSOR_COSTABLE), &graph_.inTensors.at(IN_TENSOR_SINTABLE),
                               &graph_.inTensors.at(IN_TENSOR_POSITIONID)};
    embeddingNode.outTensors = {&graph_.internalTensors.at(0), &graph_.internalTensors.at(1),
                                &graph_.internalTensors.at(2)};

    atb::Tensor *firstInTensor = &graph_.internalTensors.at(0);
    atb::Tensor *cosEmbedTensor = &graph_.internalTensors.at(1);
    atb::Tensor *sinEmbedTensor = &graph_.internalTensors.at(2);

    for (int layerId = 0; layerId < param_.layerNum; ++layerId) {
        auto &layerNode = graph_.nodes.at(nodeId++);

        atb_speed::llama2_70b::FusionPALayerW8A8Param layerParam;
        layerParam.rmsNormEps = param_.rmsNormEps;
        layerParam.headNum = param_.headNum;
        layerParam.dk = param_.dk;
        layerParam.transposedWeight = param_.transposedWeight;
        layerParam.model = "llama2_70b_w8a8";
        layerParam.rank = param_.rank;
        layerParam.rankSize = param_.rankSize;
        layerParam.numHeadsPerPartition = param_.numHeadsPerPartition;
        layerParam.isPrefill = param_.isPrefill;
        layerParam.backend = param_.backend;

        atb_speed::llama2_70b::FusionPALayerW8A8(layerParam, &op);
        layerNode.operation.reset(op);
        layerNode.inTensors.resize(layerNode.operation->GetInputNum());

        size_t inTensorId = 0;
        layerNode.inTensors.at(inTensorId++) = firstInTensor; // hidden states
        for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
            layerNode.inTensors.at(inTensorId++) =
                &graph_.weightTensors.at(layerId * WEIGHT_COUNT_PER_LAYER + weightTensorId + WEIGHT_COUNT_BEFORE_LAYER);
        }
        layerNode.inTensors.at(inTensorId++) = cosEmbedTensor;                                // cosTable
        layerNode.inTensors.at(inTensorId++) = sinEmbedTensor;                                // sinTable
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ATTENTIONMASK); // attentionMaskTensor
        layerNode.inTensors.at(inTensorId++) = &graph_.kCacheTensors.at(layerId);
        layerNode.inTensors.at(inTensorId++) = &graph_.vCacheTensors.at(layerId);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_BLOCK_TABLES);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SLOTS);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SEQLEN); // seqLen

        layerNode.outTensors = {&graph_.internalTensors.at(INTERMEDIATETENSOR_COUNT_BEFORE_LAYER + layerId)};

        firstInTensor = layerNode.outTensors.at(0);
    }
    auto &finalNormNode = graph_.nodes.at(nodeId++);
    atb::infer::RmsNormParam finalNormParam;
    finalNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    finalNormParam.normParam.epsilon = param_.rmsNormEps;
    atb::CreateOperation(finalNormParam, &op);
    finalNormNode.operation.reset(op);
    const int finalLayerNormWeightTensorId =
        graph_.weightTensors.size() - FINALNORMNODE_WEIGHT_COUNT - OUT_LM_HEAD_WEIGHT_COUNT;
    const int finalLayerNormOutTensorId = internalTensorSize - 1;
    finalNormNode.inTensors = {firstInTensor, &graph_.weightTensors.at(finalLayerNormWeightTensorId)};
    finalNormNode.outTensors = {&graph_.internalTensors.at(finalLayerNormOutTensorId)};

    auto &lmHeadNode = graph_.nodes.at(nodeId++);
    atb_speed::common::ParallelLmHeadParam lmHeadParam;
    lmHeadParam.rank = param_.rank;
    lmHeadParam.rankSize = param_.rankSize;
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
}

atb::Status FusionPAModelW8A8::ParseParam(const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);
    seqLen_.clear();
    for (auto item : paramJson["seqLen"]) {
        seqLen_.push_back(item.get<int>());
    }
    ATB_LOG(INFO) << "FusionPAModelW8A8 ParseParam seqLen: " << seqLen_.capacity();
    return atb::NO_ERROR;
}

atb::Status FusionPAModelW8A8::BindParamHostTensor(uint32_t nodeId)
{
    ATB_LOG(INFO) << "BindParamHostTensor";
    ATB_LOG(INFO) << "nodeId = " << nodeId;
    ATB_LOG(INFO) << "param_.layerNum = " << param_.layerNum;

    if (nodeId < OPERATION_COUNT_BEFORE_LAYER || nodeId >= OPERATION_COUNT_BEFORE_LAYER + param_.layerNum) {
        return atb::NO_ERROR;
    }
    auto &node = graph_.nodes.at(nodeId);
    const uint32_t seqLenTensorId = atb_speed::llama2_70b::FusionPALayerW8A8TensorId::IN_SEQLEN;
    node.variantPack.inTensors.at(seqLenTensorId).hostData = seqLen_.data();
    ATB_LOG(INFO) << "BindParamHostTensor end";
    return atb::NO_ERROR;
}
} // namespace llama2_70b
} // namespace atb_speed
