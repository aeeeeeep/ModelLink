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
#include "paged_attention_model.h"

#include <atb/atb_infer.h>
#include <nlohmann/json.hpp>

#include "atb_speed/utils/operation_util.h"
#include "models/baichuan2/13b/layer/paged_attention_layer.h"
#include "parallel_lmhead.h"
#include "atb_speed/utils/model_factory.h"

namespace atb_speed {
namespace baichuan2_13b {

REGISTER_MODEL(baichuan2_13b, PagedAttentionModel);

const int WEIGHT_COUNT_PER_LAYER = 6;
const int WORD_EMBEDDING_NODE_WEIGHT_COUNT = 1;
const int FINAL_NORM_NODE_WEIGHT_COUNT = 1;
const int OUT_LM_HEAD_WEIGHT_COUNT = 1;
const int OPERATION_COUNT_BEFORE_LAYER = 1;
const int INTERNAL_TENSOR_COUNT_BEFORE_LAYER = 1;
const int OPERATION_COUNT_AFTER_LAYER = 2;

enum InTensorId : int {
    IN_TENSOR_INPUT_IDS = 0,
    IN_TENSOR_ATTENTION_MASK,
    IN_TENSOR_BLOCK_TABLES,
    IN_TENSOR_SLOTS,
    IN_TENSOR_INPUT_LENGTHS,
    IN_TENSOR_LOGTIS_INDICES,
    IN_HOLDER,
    IN_TENSOR_MAX,
};

enum OutTensorId : int {
    OUT_TENSOR_HIDDEN_STATES = 0,
    OUT_TENSOR_MAX,
};

void PagedAttentionModel::Param::FromString(const std::string &param)
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

    ATB_LOG(INFO) << "Baichuan2_13BPAModel param rmsNormEps:" << rmsNormEps << ", headNum:" << headNum << ", dk:" << dk
                  << ", layerNum:" << layerNum << ", transposedWeight:" << transposedWeight << ", rank:" << rank
                  << ", rankSize:" << rankSize << ", backend: " << backend << ", isLmHeadParallel:" << isLmHeadParallel;
}

PagedAttentionModel::PagedAttentionModel(const std::string &param) : Model("Baichuan2_13BPAModel", param)
{
    param_.FromString(param);
    modelName_ += param_.isPrefill ? "_Prefill" : "_Decoder";
}

PagedAttentionModel::~PagedAttentionModel() = default;

uint32_t PagedAttentionModel::GetInputNum() const { return graph_.inTensors.size(); }

uint32_t PagedAttentionModel::GetOutputNum() const { return graph_.outTensors.size(); }

atb::Status PagedAttentionModel::InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
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

    // change first dim
    if (param_.isPrefill) {
        outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(IN_TENSOR_LOGTIS_INDICES).shape.dims[0];
    }

    return atb::NO_ERROR;
}

int64_t PagedAttentionModel::BuildGraph()
{
    const int weightTensorSize = WORD_EMBEDDING_NODE_WEIGHT_COUNT + WEIGHT_COUNT_PER_LAYER * param_.layerNum +
                                 FINAL_NORM_NODE_WEIGHT_COUNT + OUT_LM_HEAD_WEIGHT_COUNT;
    graph_.weightTensors.resize(weightTensorSize);

    graph_.kCacheTensors.resize(param_.layerNum);
    graph_.vCacheTensors.resize(param_.layerNum);

    graph_.inTensors.resize(IN_TENSOR_MAX);
    graph_.outTensors.resize(OUT_TENSOR_MAX);

    const int nodeSize = param_.layerNum + OPERATION_COUNT_BEFORE_LAYER + OPERATION_COUNT_AFTER_LAYER;
    ATB_LOG(INFO) << "Baichuan2_13BPAModel nodeSize is " << nodeSize;
    graph_.nodes.resize(nodeSize);

    const int internalTensorSize = graph_.nodes.size() - 1;
    graph_.internalTensors.resize(internalTensorSize);

    int nodeId = 0;
    auto &wordEmbeddingNode = graph_.nodes.at(nodeId++);
    atb::infer::GatherParam wordEmbeddingParam;
    atb::Operation *op = nullptr;
    CREATE_OPERATION(wordEmbeddingParam, &op);
    wordEmbeddingNode.operation.reset(op);
    wordEmbeddingNode.inTensors = {&graph_.weightTensors.at(0), &graph_.inTensors.at(IN_TENSOR_INPUT_IDS)};
    wordEmbeddingNode.outTensors = {&graph_.internalTensors.at(0)};

    atb::Tensor *firstInTensor = &graph_.internalTensors.at(0);

    for (int layerId = 0; layerId < param_.layerNum; ++layerId) {
        auto &layerNode = graph_.nodes.at(nodeId++);

        atb_speed::baichuan2_13b::PALayerParam opParam;
        opParam.rmsNormEps = param_.rmsNormEps;
        opParam.headNum = param_.headNum;
        opParam.dk = param_.dk;
        opParam.transposedWeight = param_.transposedWeight;
        opParam.isPrefill = param_.isPrefill;
        opParam.rank = param_.rank;
        opParam.rankSize = param_.rankSize;
        opParam.backend = param_.backend;
        atb_speed::baichuan2_13b::PALayer(opParam, &op);
        layerNode.operation.reset(op);
        layerNode.inTensors.resize(layerNode.operation->GetInputNum());

        size_t inTensorId = 0;
        layerNode.inTensors.at(inTensorId++) = firstInTensor;
        for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
            layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(
                layerId * WEIGHT_COUNT_PER_LAYER + weightTensorId + WORD_EMBEDDING_NODE_WEIGHT_COUNT);
        }
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ATTENTION_MASK); // attentionMaskTensor
        layerNode.inTensors.at(inTensorId++) = &graph_.kCacheTensors.at(layerId);
        layerNode.inTensors.at(inTensorId++) = &graph_.vCacheTensors.at(layerId);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_BLOCK_TABLES);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SLOTS);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_INPUT_LENGTHS);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_HOLDER);

        layerNode.outTensors = {&graph_.internalTensors.at(INTERNAL_TENSOR_COUNT_BEFORE_LAYER + layerId)};

        firstInTensor = layerNode.outTensors.at(0);
    }

    auto &finalNormNode = graph_.nodes.at(nodeId++);
    atb::infer::RmsNormParam finalNormParam;
    finalNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    finalNormParam.normParam.epsilon = param_.rmsNormEps;
    CREATE_OPERATION(finalNormParam, &op);
    finalNormNode.operation.reset(op);
    const int finalLayerNormWeightTensorId =
        graph_.weightTensors.size() - FINAL_NORM_NODE_WEIGHT_COUNT - OUT_LM_HEAD_WEIGHT_COUNT;
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

atb::Status PagedAttentionModel::ParseParam(const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);

    seqLen_.clear();
    for (const auto &item : paramJson["seqLen"]) {
        seqLen_.push_back(item.get<int>());
    }

    ATB_LOG(INFO) << "PagedAttentionModel ParseParam seqLen: " << seqLen_.capacity();

    return atb::NO_ERROR;
}

atb::Status PagedAttentionModel::BindParamHostTensor(uint32_t nodeId)
{
    if (nodeId < OPERATION_COUNT_BEFORE_LAYER || nodeId >= OPERATION_COUNT_BEFORE_LAYER + param_.layerNum) {
        return atb::NO_ERROR;
    }

    auto &node = graph_.nodes.at(nodeId);
    const uint32_t seqLenTensorId = 12; // atb_speed::baichuan2_13b::LayerPATensorId::IN_INPUT_LENGTHS
    node.variantPack.inTensors.at(seqLenTensorId).hostData = seqLen_.data();

    return atb::NO_ERROR;
}
} // namespace baichuan2_13b
} // namespace atb_speed