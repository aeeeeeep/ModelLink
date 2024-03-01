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
#include "models/aquila/7b/model/paged_attention_model.h"

#include "atb/atb_infer.h"
#include "nlohmann/json.hpp"

#include "models/aquila/7b/layer/paged_attention_layer.h"
#include "parallel_lmhead.h"

namespace atb_speed {
namespace aquila_7b {
enum InTensorId : int {
    IN_TENSOR_INPUT_IDS = 0,
    IN_TENSOR_POSITION_ID,
    IN_TENSOR_COS_EMBED,
    IN_TENSOR_SIN_EMBED,
    IN_TENSOR_ATTENTION_MASK,
    IN_TENSOR_BLOCK_TABLES,
    IN_TENSOR_SLOTS,
    IN_TENSOR_INPUT_LENGTHS,
    IN_TENSOR_LOGTIS_INDICES,
    IN_HOLDER,
    IN_TENSOR_MAX, // 10
};

enum OutTensorId : int {
    OUT_TENSOR_HIDDENSTATES = 0,
    OUT_TENSOR_MAX,
};

const int WEIGHT_COUNT_PER_LAYER = 9;
const int WORDEMBEDDINGNODE_WEIGHT_COUNT = 1;
const int FINALNORMNODE_WEIGHT_COUNT = 1;
const int OUT_LM_HEAD_WEIGHT_COUNT = 1;
const int OPERATION_COUNT_BEFORE_LAYER = 1;
const int INTERMEDIATETENSOR_COUNT_BEFORE_LAYER = 1;
const int OPERATION_COUNT_AFTER_LAYER = 2;
const int OUT_TENSOR_HIDDENSTATES_ID = 0;
const int IN_TENSOR_INPUTIDS_ID = 0;
const int WORDEMBEDDINGNODE_WEIGHT_ID = 0;
const int FIRST_INTERNAL_TENSORS = 0;
const int LAYER_FIRST_OUT_TENSORS = 0;
const int FA_ROPE_LAYER_IN_SEQLEN_ID = 16;
const int OUT_TENSOR_HIDDENSTATES_ID_DIM_NUM = 3;

void PagedAttentionRopeModel::Param::FromString(const std::string &param)
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
    if (paramJson.contains("transposedWeight")) {
        transposedWeight = paramJson["transposedWeight"].get<bool>();
    }
    if (paramJson.contains("isPrefill")) {
        isPrefill = paramJson["isPrefill"].get<bool>();
    }
    if (paramJson.contains("isLmHeadParallel")) {
        isLmHeadParallel = paramJson["isLmHeadParallel"].get<bool>();
    }

    ATB_LOG(INFO) << "Aquila_7B Paged Attention Model params: \n"
                  << "  rmsNormEps:      " << rmsNormEps << ", \n"
                  << "  headNum:         " << headNum << ", \n"
                  << "  dk:              " << dk << ", \n"
                  << "  layerNum:        " << layerNum << ", \n"
                  << "  rank:            " << rank << ", \n"
                  << "  rankSize:        " << rankSize << ", \n"
                  << "  backend:         " << backend << ", \n"
                  << "  transposedWeight:" << transposedWeight << ", \n"
                  << "  isPrefill:       " << isPrefill << ", \n"
                  << "  isLmHeadParallel:" << isLmHeadParallel;
}

PagedAttentionRopeModel::PagedAttentionRopeModel(const std::string &param) : Model("PagedAttentionModel", param)
{
    param_.FromString(param);
    modelName_ += param_.isPrefill ? "_Prefill" : "_Decoder";
    ATB_LOG(INFO) << "==========================PagedAttentionRopeModel::modelName_=" << modelName_;
}

PagedAttentionRopeModel::~PagedAttentionRopeModel() = default;

uint32_t PagedAttentionRopeModel::GetInputNum() const { return graph_.inTensors.size(); }

uint32_t PagedAttentionRopeModel::GetOutputNum() const { return graph_.outTensors.size(); }

atb::Status PagedAttentionRopeModel::InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
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

    return atb::NO_ERROR;}

int64_t PagedAttentionRopeModel::BuildGraph()
{
    const int weightTensorSize = WORDEMBEDDINGNODE_WEIGHT_COUNT + WEIGHT_COUNT_PER_LAYER * param_.layerNum +
                                 FINALNORMNODE_WEIGHT_COUNT + OUT_LM_HEAD_WEIGHT_COUNT;
    graph_.weightTensors.resize(weightTensorSize);

    graph_.kCacheTensors.resize(param_.layerNum);
    graph_.vCacheTensors.resize(param_.layerNum);

    graph_.inTensors.resize(IN_TENSOR_MAX);
    graph_.outTensors.resize(OUT_TENSOR_MAX);

    const int nodeSize = param_.layerNum + OPERATION_COUNT_BEFORE_LAYER + OPERATION_COUNT_AFTER_LAYER;
    graph_.nodes.resize(nodeSize);

    const int internalTensorSize = graph_.nodes.size() - 1;
    graph_.internalTensors.resize(internalTensorSize);

    int nodeId = 0;
    auto &wordEmbeddingNode = graph_.nodes.at(nodeId++);
    atb::infer::GatherParam wordEmbeddingParam;
    atb::Operation *op = nullptr;
    CREATE_OPERATION(wordEmbeddingParam, &op);
    wordEmbeddingNode.operation.reset(op);
    wordEmbeddingNode.inTensors = {&graph_.weightTensors.at(WORDEMBEDDINGNODE_WEIGHT_ID),
                                   &graph_.inTensors.at(IN_TENSOR_INPUT_IDS)};
    wordEmbeddingNode.outTensors = {&graph_.internalTensors.at(FIRST_INTERNAL_TENSORS)};

    atb::Tensor *firstInTensor = &graph_.internalTensors.at(FIRST_INTERNAL_TENSORS);

    for (int layerId = 0; layerId < param_.layerNum; ++layerId) {
        auto &layerNode = graph_.nodes.at(nodeId++);

        atb_speed::aquila_7b::PagedAttentionRopeLayerParam opParam;
        opParam.rmsNormEps = param_.rmsNormEps;
        opParam.headNum = param_.headNum;
        opParam.dk = param_.dk;
        opParam.rank = param_.rank;
        opParam.rankSize = param_.rankSize;
        opParam.backend = param_.backend;
        opParam.transposedWeight = param_.transposedWeight;
        opParam.isPrefill = param_.isPrefill;
        atb_speed::aquila_7b::PagedAttentionRopeLayer(opParam, &op);
        layerNode.operation.reset(op);
        layerNode.inTensors.resize(layerNode.operation->GetInputNum());

        size_t inTensorId = 0;
        layerNode.inTensors.at(inTensorId++) = firstInTensor;
        for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
            layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(
                layerId * WEIGHT_COUNT_PER_LAYER + weightTensorId + WORDEMBEDDINGNODE_WEIGHT_COUNT);
        }
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_POSITION_ID);    // positionIdTensor
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_COS_EMBED);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SIN_EMBED);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ATTENTION_MASK);
        layerNode.inTensors.at(inTensorId++) = &graph_.kCacheTensors.at(layerId);
        layerNode.inTensors.at(inTensorId++) = &graph_.vCacheTensors.at(layerId);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_BLOCK_TABLES);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SLOTS);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_INPUT_LENGTHS);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_HOLDER);

        layerNode.outTensors = {&graph_.internalTensors.at(INTERMEDIATETENSOR_COUNT_BEFORE_LAYER + layerId)};

        firstInTensor = layerNode.outTensors.at(LAYER_FIRST_OUT_TENSORS);
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
    ATB_LOG(INFO) << "===================== Model lmHead done, graph_.nodes=" << graph_.nodes.size();

    return atb::NO_ERROR;
}

atb::Status PagedAttentionRopeModel::ParseParam(const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);
    seqLen_.clear();
    for (auto item : paramJson["seqLen"]) {
        seqLen_.push_back(item.get<int>());
    }

    return atb::NO_ERROR;
}

atb::Status PagedAttentionRopeModel::BindParamHostTensor(uint32_t nodeId)
{
    if (nodeId < OPERATION_COUNT_BEFORE_LAYER || nodeId >= OPERATION_COUNT_BEFORE_LAYER + param_.layerNum) {
        return atb::NO_ERROR;
    }
    auto &node = graph_.nodes.at(nodeId);

    const uint32_t seqLenTensorId = FA_ROPE_LAYER_IN_SEQLEN_ID;
    node.variantPack.inTensors.at(seqLenTensorId).hostData = seqLen_.data();
    ATB_LOG(INFO) << "BindParamHostTensor end";
    return atb::NO_ERROR;
}
} // namespace aquila_7b
} // namespace atb_speed