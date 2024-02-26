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
#include "atb/atb_infer.h"
#include "atb_speed/log.h"
#include "nlohmann/json.hpp"
#pragma GCC diagnostic ignored "-Wsign-compare"
#include "models/starcoder/layer/paged_attention_layer.h"

#include "paged_attention_model.h"

namespace atb_speed {
namespace star_coder {
const int WEIGHT_COUNT_PER_LAYER = 12;
const int BEFORE_LAYER_WEIGHT_COUNT = 2;
const int OPERATION_COUNT_BEFORE_LAYER = 3;
const int INTERMEDIATETENSOR_COUNT_BEFORE_LAYER = 3;
const int OPERATION_COUNT_AFTER_LAYER = 2;
const int FINALNORMNODE_WEIGHT_COUNT = 2;
const int OUT_LM_HEAD_WEIGHT_COUNT = 1;
const int LAYER_NORM_AXIS_COUNT = 2;

enum InTensorId : int {
    IN_TENSOR_INPUT_IDS = 0,
    IN_TENSOR_POSITION_IDS,
    IN_TENSOR_ATTENTIONMASK,
    IN_TENSOR_BLOCK_TABLES,
    IN_TENSOR_SLOTS,
    IN_TENSOR_INPUT_LENGTHS,
    IN_TENSOR_LOGTIS_INDICES,
    IN_HOLDER,
    IN_TENSOR_MAX, // 8
};

enum OutTensorId : int {
    OUT_TENSOR_HIDDENSTATES = 0,
    OUT_TENSOR_MAX,
};

void PAModel::Param::FromString(const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);
    layerNormEps = paramJson["layerNormEps"].get<double>();
    headNum = paramJson["headNum"].get<int>();
    dk = paramJson["dk"].get<int>();
    kvHead = paramJson["kvHead"].get<int>();
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
    ATB_LOG(INFO) << "StarCoderPAModel param layerNormEps:" << layerNormEps << ", headNum:" << headNum << ", dk:"
                  << dk << ", layerNum:" << layerNum << ", transposedWeight:" << transposedWeight << ", rank:"
                  << rank << ", rankSize:" << rankSize << ", backend: " << backend;
}

PAModel::PAModel(const std::string &param) : Model("PAModel", param)
{
    param_.FromString(param);
    ATB_LOG(INFO) << "check from string success";
}

PAModel::~PAModel() = default;

uint32_t PAModel::GetInputNum() const
{
    return graph_.inTensors.size();
}

uint32_t PAModel::GetOutputNum() const
{
    return graph_.outTensors.size();
}

atb::Status PAModel::InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
    std::vector<atb::TensorDesc> &outTensorDescs)
{
    ATB_LOG(INFO) << "Enter PAModel InferShape";
    if (outTensorDescs.size() != GetOutputNum()) {
        return atb::ERROR_INVALID_GRAPH;
    }
    const int64_t outDim = graph_.weightTensors.at(graph_.weightTensors.size() - 1).desc.shape.dims[0];
    outTensorDescs.at(0) = graph_.weightTensors.at(0).desc;
    outTensorDescs.at(0).shape.dimNum = 3;
    outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
    outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1];
    outTensorDescs.at(0).shape.dims[2] = outDim;

    ATB_LOG(INFO) << "PAModel InferShape Success";
    return atb::NO_ERROR;
}

int64_t PAModel::BuildGraph()
{
    ATB_LOG(INFO) << "Enter PAModel BuildGraph";
    const int weightTensorSize = BEFORE_LAYER_WEIGHT_COUNT + WEIGHT_COUNT_PER_LAYER * param_.layerNum +
                                 FINALNORMNODE_WEIGHT_COUNT + OUT_LM_HEAD_WEIGHT_COUNT;
    graph_.weightTensors.resize(weightTensorSize);

    graph_.inTensors.resize(IN_TENSOR_MAX + param_.layerNum * 2);
    graph_.outTensors.resize(OUT_TENSOR_MAX);

    const int nodeSize = param_.layerNum + OPERATION_COUNT_BEFORE_LAYER + OPERATION_COUNT_AFTER_LAYER;
    graph_.nodes.resize(nodeSize);

    const uint32_t internalTensorSize = graph_.nodes.size() - 1;
    graph_.internalTensors.resize(internalTensorSize);

    int nodeId = 0;
    atb::Operation *op = nullptr;

    auto &wtEmbeddingNode = graph_.nodes.at(nodeId++);
    atb::infer::GatherParam wtEmbeddingParam;
    CREATE_OPERATION(wtEmbeddingParam, &op);
    wtEmbeddingNode.operation.reset(op);
    wtEmbeddingNode.inTensors = {&graph_.weightTensors.at(0), &graph_.inTensors.at(IN_TENSOR_INPUT_IDS)};
    wtEmbeddingNode.outTensors = {&graph_.internalTensors.at(0)};

    auto &wpEmbeddingNode = graph_.nodes.at(nodeId++);
    atb::infer::GatherParam wpEmbeddingParam;
    CREATE_OPERATION(wpEmbeddingParam, &op);
    wpEmbeddingNode.operation.reset(op);
    wpEmbeddingNode.inTensors = {&graph_.weightTensors.at(1), &graph_.inTensors.at(IN_TENSOR_POSITION_IDS)};
    wpEmbeddingNode.outTensors = {&graph_.internalTensors.at(1)};

    auto &addNode = graph_.nodes.at(nodeId++);
    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CREATE_OPERATION(addParam, &op);
    addNode.operation.reset(op);
    addNode.inTensors = {&graph_.internalTensors.at(0), &graph_.internalTensors.at(1)};
    addNode.outTensors = {&graph_.internalTensors.at(2)};

    atb::Tensor *firstInTensor = &graph_.internalTensors.at(2);
    for (int layerId = 0; layerId < param_.layerNum; ++layerId) {
        auto &layerNode = graph_.nodes.at(nodeId++);

        atb_speed::star_coder::PALayerParam modelParam;
        modelParam.layerNormEps = param_.layerNormEps;
        modelParam.headNum = param_.headNum;
        modelParam.dk = param_.dk;
        modelParam.kvHead = param_.kvHead;
        modelParam.rank = param_.rank;
        modelParam.rankSize = param_.rankSize;
        modelParam.backend = param_.backend;
        modelParam.isPrefill = param_.isPrefill;
        modelParam.model = "star_coder";

        atb_speed::star_coder::PALayer(modelParam, &op);
        layerNode.operation.reset(op);
        layerNode.inTensors.resize(layerNode.operation->GetInputNum());

        size_t inTensorId = 0;
        layerNode.inTensors.at(inTensorId++) = firstInTensor; // hidden states
        for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
            layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(
                layerId * WEIGHT_COUNT_PER_LAYER + weightTensorId + BEFORE_LAYER_WEIGHT_COUNT);
        }
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ATTENTIONMASK);   // atten_mask
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_BLOCK_TABLES);    // block
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SLOTS);           // slots
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_INPUT_LENGTHS);   // seqLen
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_HOLDER);                 // holder
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_MAX + layerId);   // pastk
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_MAX + layerId + param_.layerNum); // pastv
        
        layerNode.outTensors = {&graph_.internalTensors.at(INTERMEDIATETENSOR_COUNT_BEFORE_LAYER + layerId)};

        firstInTensor = layerNode.outTensors.at(0);
        ATB_LOG(INFO) << "StarCoderPAModel BuildGraph success";
    }

    auto &finalNormNode = graph_.nodes.at(nodeId++);
    atb::infer::LayerNormParam finalNormParam;
    finalNormParam.layerType = atb::infer::LayerNormParam::LayerNormType::LAYER_NORM_NORM;
    finalNormParam.normParam.epsilon = param_.layerNormEps;
    finalNormParam.normParam.beginNormAxis = LAYER_NORM_AXIS_COUNT;
    finalNormParam.normParam.beginParamsAxis = LAYER_NORM_AXIS_COUNT;
    CREATE_OPERATION(finalNormParam, &op);
    finalNormNode.operation.reset(op);
    const int finalLayerNormWeightTensorId =
        graph_.weightTensors.size() - FINALNORMNODE_WEIGHT_COUNT - OUT_LM_HEAD_WEIGHT_COUNT;
    const int finalLayerNormBiasTensorId =
        graph_.weightTensors.size() - (FINALNORMNODE_WEIGHT_COUNT - 1) - OUT_LM_HEAD_WEIGHT_COUNT;
    const int finalLayerNormOutTensorId = internalTensorSize - 1;
    finalNormNode.inTensors = {firstInTensor, &graph_.weightTensors.at(finalLayerNormWeightTensorId),
                               &graph_.weightTensors.at(finalLayerNormBiasTensorId)};
    finalNormNode.outTensors = {&graph_.internalTensors.at(finalLayerNormOutTensorId)};

    auto &outLinearNode = graph_.nodes.at(nodeId++);
    atb::infer::LinearParam outLinearParm;
    outLinearParm.hasBias = false;
    CREATE_OPERATION(outLinearParm, &op);
    outLinearNode.operation.reset(op);
    const int finalLinearWeightTensorId = graph_.weightTensors.size() - OUT_LM_HEAD_WEIGHT_COUNT;
    outLinearNode.inTensors = {&graph_.internalTensors.at(finalLayerNormOutTensorId),
                               &graph_.weightTensors.at(finalLinearWeightTensorId)};
    outLinearNode.outTensors = {&graph_.outTensors.at(0)};
    return atb::NO_ERROR;
}

atb::Status PAModel::ParseParam(const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);

    seqLen_.clear();
    for (auto item : paramJson["seqLen"]) {
        seqLen_.push_back(item.get<int>());
    }
    
    ATB_LOG(INFO) << "PAModel ParseParam seqLen: " << seqLen_.capacity();
    return atb::NO_ERROR;
}

atb::Status PAModel::BindParamHostTensor(uint32_t nodeId)
{
    ATB_LOG(INFO) << "BindParamHostTensor";
    if (nodeId < OPERATION_COUNT_BEFORE_LAYER || nodeId >= OPERATION_COUNT_BEFORE_LAYER + param_.layerNum) {
        return atb::NO_ERROR;
    }
    auto &node = graph_.nodes.at(nodeId);
    const uint32_t seqLenTensorId = 16; // IN_INPUT_LENGTHS of decoder layer
    node.variantPack.inTensors.at(seqLenTensorId).hostData = seqLen_.data();

    ATB_LOG(INFO) << "BindParamHostTensor end";
    return atb::NO_ERROR;
}
} // namespace star_coder
} // namespace atb_speed