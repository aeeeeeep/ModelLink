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
#include "flash_attention_quant_model.h"

#include "atb/atb_infer.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtype-limits"
#include "nlohmann/json.hpp"
#pragma GCC diagnostic pop

#include "layers/parallel_layer_v2.h"
#include "models/baichuan2/7b/layer/flash_attention_quant_layer.h"
#include "models/baichuan2/7b/layer/flash_attention_rope_layer.h"
#include "atb_speed/utils/model_factory.h"

namespace atb_speed {
namespace baichuan2_7b {

REGISTER_MODEL(baichuan2_7b, FlashAttentionQuantModel);

const int WEIGHT_COUNT_PER_LAYER = 17;
const int WORD_EMBEDDING_NODE_WEIGHT_COUNT = 1;
const int FINAL_NORM_NODE_WEIGHT_COUNT = 1;
const int OUT_LM_HEAD_WEIGHT_COUNT = 1;
const int OPERATION_COUNT_BEFORE_LAYER = 1;
const int INTERNAL_TENSOR_COUNT_BEFORE_LAYER = 1;
const int OPERATION_COUNT_AFTER_LAYER = 3;
const int ROLLBACK_WEIGHT_COUNT_PER_LAYER = 7;

enum InTensorId : int {
    IN_TENSOR_INPUTIDS = 0,
    IN_TENSOR_COSEMBED,
    IN_TENSOR_SINEMBED,
    IN_TENSOR_ATTENTION_MASK,
    IN_TENSOR_PAST_KEY,
    IN_TENSOR_PAST_VALUE,
    IN_TENSOR_TOKEN_OFFSET,
    IN_TENSOR_SEQ_LEN,
    IN_BETA,
    IN_HOLDER,
    IN_TENSOR_MAX, // 10
};

enum OutTensorId : int {
    OUT_TENSOR_HIDDENSTATES = 0,
    OUT_TENSOR_MAX,
};

void FlashAttentionQuantModel::Param::FromString(const std::string &param)
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
    for (auto item : paramJson["w_packInputScale"]) {
        w_packInputScale.push_back(item.get<float>());
    }
    for (auto item : paramJson["w_packInputOffset"]) {
        w_packInputOffset.push_back(item.get<int>());
    }
    for (auto item : paramJson["o_projInputScale"]) {
        o_projInputScale.push_back(item.get<float>());
    }
    for (auto item : paramJson["o_projInputOffset"]) {
        o_projInputOffset.push_back(item.get<int>());
    }
    for (auto item : paramJson["gate_projInputScale"]) {
        gate_projInputScale.push_back(item.get<float>());
    }
    for (auto item : paramJson["gate_projInputOffset"]) {
        gate_projInputOffset.push_back(item.get<int>());
    }
    for (auto item : paramJson["down_projInputScale"]) {
        down_projInputScale.push_back(item.get<float>());
    }
    for (auto item : paramJson["down_projInputOffset"]) {
        down_projInputOffset.push_back(item.get<int>());
    }
    for (auto item : paramJson["up_projInputScale"]) {
        up_projInputScale.push_back(item.get<float>());
    }
    for (auto item : paramJson["up_projInputOffset"]) {
        up_projInputOffset.push_back(item.get<int>());
    }
    for (auto item : paramJson["roll_back_layer"]) {
        roll_back_layer.push_back(item.get<int>());
    }
}

FlashAttentionQuantModel::FlashAttentionQuantModel(const std::string &param) : Model("FlashAttentionQuantModel", param)
{
    param_.FromString(param);
}

FlashAttentionQuantModel::~FlashAttentionQuantModel() = default;

uint32_t FlashAttentionQuantModel::GetInputNum() const { return graph_.inTensors.size(); }

uint32_t FlashAttentionQuantModel::GetOutputNum() const { return graph_.outTensors.size(); }

atb::Status FlashAttentionQuantModel::InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
                                                 std::vector<atb::TensorDesc> &outTensorDescs)
{
    if (outTensorDescs.size() != GetOutputNum()) {
        return atb::ERROR_INVALID_GRAPH;
    }
    const int64_t outDim = graph_.weightTensors.at(graph_.weightTensors.size() - 1).desc.shape.dims[0];
    outTensorDescs.at(0) = graph_.weightTensors.at(0).desc;
    outTensorDescs.at(0).shape.dimNum = 3;
    outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
    outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1];
    outTensorDescs.at(0).shape.dims[2] = outDim;

    return atb::NO_ERROR;
}

int64_t FlashAttentionQuantModel::BuildGraph()
{
    int rollbackLayerLength = param_.roll_back_layer.size();
    const int weightTensorSize = WORD_EMBEDDING_NODE_WEIGHT_COUNT +
                                 ROLLBACK_WEIGHT_COUNT_PER_LAYER * rollbackLayerLength +
                                 WEIGHT_COUNT_PER_LAYER * (param_.layerNum - rollbackLayerLength) +
                                 FINAL_NORM_NODE_WEIGHT_COUNT + OUT_LM_HEAD_WEIGHT_COUNT;

    ATB_LOG(ERROR) << __func__ << " weightTensorSize: " << weightTensorSize;
    graph_.weightTensors.resize(weightTensorSize);

    graph_.inTensors.resize(IN_TENSOR_MAX + param_.layerNum);
    graph_.outTensors.resize(OUT_TENSOR_MAX);

    const int nodeSize = param_.layerNum + OPERATION_COUNT_BEFORE_LAYER + OPERATION_COUNT_AFTER_LAYER;
    graph_.nodes.resize(nodeSize);

    const int internalTensorSize = graph_.nodes.size() - 1;
    graph_.internalTensors.resize(internalTensorSize);

    int nodeId = 0;
    auto &wordEmbeddingNode = graph_.nodes.at(nodeId++);
    atb::infer::GatherParam wordEmbeddingParam;
    atb::Operation *op = nullptr;
    atb::CreateOperation(wordEmbeddingParam, &op);
    wordEmbeddingNode.operation.reset(op);
    wordEmbeddingNode.inTensors = {&graph_.weightTensors.at(0), &graph_.inTensors.at(IN_TENSOR_INPUTIDS)};
    wordEmbeddingNode.outTensors = {&graph_.internalTensors.at(0)};

    atb::Tensor *firstInTensor = &graph_.internalTensors.at(0);
    int layerTmpId = 0;
    for (int layerId = 0; layerId < param_.layerNum; ++layerId) {
        auto &layerNode = graph_.nodes.at(nodeId++);
        if (std::find(std::begin(param_.roll_back_layer), std::end(param_.roll_back_layer), layerId) !=
            std::end(param_.roll_back_layer)) {
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
            for (size_t weightTensorId = 0; weightTensorId < ROLLBACK_WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
                layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(
                    (layerId - layerTmpId) * WEIGHT_COUNT_PER_LAYER + layerTmpId * ROLLBACK_WEIGHT_COUNT_PER_LAYER +
                    weightTensorId + WORD_EMBEDDING_NODE_WEIGHT_COUNT);
            }
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_COSEMBED); // cosEmbed
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SINEMBED); // sinEmbed
            layerNode.inTensors.at(inTensorId++) =
                &graph_.inTensors.at(IN_TENSOR_ATTENTION_MASK); // attentionMaskTensor
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PAST_KEY);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PAST_VALUE);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_TOKEN_OFFSET);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SEQ_LEN); // seqLen
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_HOLDER);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_MAX + layerId);

            layerNode.outTensors = {&graph_.internalTensors.at(INTERNAL_TENSOR_COUNT_BEFORE_LAYER + layerId)};
            firstInTensor = layerNode.outTensors.at(0);
            layerTmpId += 1;
        } else {
            atb_speed::baichuan2_7b::FlashAttentionQuantLayerParam opParam;
            // param_ -> 通过param传入
            opParam.rmsNormEps = param_.rmsNormEps;
            opParam.headNum = param_.headNum;
            opParam.dk = param_.dk;
            opParam.rank = param_.rank;
            opParam.rankSize = param_.rankSize;
            opParam.backend = param_.backend;
            opParam.w_packInputScale = param_.w_packInputScale[layerId];
            opParam.w_packInputOffset = param_.w_packInputOffset[layerId];
            opParam.o_projInputScale = param_.o_projInputScale[layerId];
            opParam.o_projInputOffset = param_.o_projInputOffset[layerId];
            opParam.gate_projInputScale = param_.gate_projInputScale[layerId];
            opParam.gate_projInputOffset = param_.gate_projInputOffset[layerId];
            opParam.down_projInputScale = param_.down_projInputScale[layerId];
            opParam.down_projInputOffset = param_.down_projInputOffset[layerId];
            atb_speed::baichuan2_7b::FlashAttentionQuantLayer(opParam, &op);
            layerNode.operation.reset(op);
            layerNode.inTensors.resize(layerNode.operation->GetInputNum()); // .at 需要resize，直接赋值不需要???

            // weightTensors -> 通过setweight传入
            size_t inTensorId = 0;
            layerNode.inTensors.at(inTensorId++) = firstInTensor; // IN_HIDDEN_STATES
            for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER;
                 ++weightTensorId) { // IN_NORM_WEIGHT ->IN_SELF_OUT_NORM_WEIGHT
                layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(
                    (layerId - layerTmpId) * WEIGHT_COUNT_PER_LAYER + layerTmpId * ROLLBACK_WEIGHT_COUNT_PER_LAYER +
                    weightTensorId + WORD_EMBEDDING_NODE_WEIGHT_COUNT);
            }
            // inTensors -> 通过 input参数传入
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_COSEMBED); // cosEmbed
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SINEMBED); // sinEmbed
            layerNode.inTensors.at(inTensorId++) =
                &graph_.inTensors.at(IN_TENSOR_ATTENTION_MASK); // attentionMaskTensor
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PAST_KEY);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PAST_VALUE);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_TOKEN_OFFSET);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SEQ_LEN); // seqLen
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_BETA);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_HOLDER);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_MAX + layerId);

            layerNode.outTensors = {&graph_.internalTensors.at(INTERNAL_TENSOR_COUNT_BEFORE_LAYER + layerId)}; //
            firstInTensor = layerNode.outTensors.at(0);
        }
    }

    auto &finalNormNode = graph_.nodes.at(nodeId++);
    atb::infer::RmsNormParam finalNormParam;
    finalNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    finalNormParam.normParam.epsilon = param_.rmsNormEps;
    atb::CreateOperation(finalNormParam, &op);
    finalNormNode.operation.reset(op);
    const int finalLayerNormWeightTensorId =
        graph_.weightTensors.size() - FINAL_NORM_NODE_WEIGHT_COUNT - OUT_LM_HEAD_WEIGHT_COUNT;
    const int finalLayerNormOutTensorId = internalTensorSize - 2;
    finalNormNode.inTensors = {firstInTensor, &graph_.weightTensors.at(finalLayerNormWeightTensorId)};
    finalNormNode.outTensors = {&graph_.internalTensors.at(finalLayerNormOutTensorId)};

    const int hiddenSize = param_.headNum * param_.dk;
    auto &qPassSliceNode = graph_.nodes.at(nodeId++);
    atb::infer::SliceParam slicePassParam;
    slicePassParam.offsets = {0, 0, hiddenSize * param_.rank};
    slicePassParam.size = {-1, -1, hiddenSize};
    CreateOperation(slicePassParam, &op);
    qPassSliceNode.operation.reset(op);
    const int qPassSliceNodeOutTensorId = internalTensorSize - 1;
    qPassSliceNode.inTensors = {&graph_.internalTensors.at(finalLayerNormOutTensorId)};
    qPassSliceNode.outTensors = {&graph_.internalTensors.at(qPassSliceNodeOutTensorId)};

    auto &outLinearNode = graph_.nodes.at(nodeId++);
    atb_speed::common::ParallelParamV2 outLinearParm;
    outLinearParm.commParam.rank = param_.rank;
    outLinearParm.commParam.rankSize = param_.rankSize;
    outLinearParm.isBias = false;
    outLinearParm.commParam.backend = param_.backend;
    atb_speed::common::RowParallelLinearV2(outLinearParm, &op);
    outLinearNode.operation.reset(op);
    const int finalLinearWeightTensorId = graph_.weightTensors.size() - OUT_LM_HEAD_WEIGHT_COUNT;
    outLinearNode.inTensors = {&graph_.internalTensors.at(qPassSliceNodeOutTensorId),
                               &graph_.weightTensors.at(finalLinearWeightTensorId),
                               &graph_.internalTensors.at(IN_HOLDER),
                               &graph_.internalTensors.at(IN_HOLDER),
                               &graph_.internalTensors.at(IN_HOLDER),
                               &graph_.internalTensors.at(IN_HOLDER),
                               &graph_.internalTensors.at(IN_HOLDER)};
    outLinearNode.outTensors = {&graph_.outTensors.at(0)};

    return atb::NO_ERROR;
}

atb::Status FlashAttentionQuantModel::ParseParam(const std::string &param)
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

atb::Status FlashAttentionQuantModel::BindParamHostTensor(uint32_t nodeId)
{
    if (nodeId < OPERATION_COUNT_BEFORE_LAYER || nodeId >= static_cast<uint32_t>(OPERATION_COUNT_BEFORE_LAYER + param_.layerNum)) {
        return atb::NO_ERROR;
    }
    auto &node = graph_.nodes.at(nodeId);
    const uint32_t floatTokenOffsetTensorId = 13;
    const uint32_t floatSeqLenTensorId = 14;
    const uint32_t tokenOffsetTensorId = 23; // 使用layer里的ID
    const uint32_t seqLenTensorId = 24;
    if (std::find(std::begin(param_.roll_back_layer), std::end(param_.roll_back_layer), nodeId - 1) !=
        std::end(param_.roll_back_layer)) {
        node.variantPack.inTensors.at(floatTokenOffsetTensorId).hostData = tokenOffset_.data();
        node.variantPack.inTensors.at(floatSeqLenTensorId).hostData = seqLen_.data();
    } else {
        node.variantPack.inTensors.at(tokenOffsetTensorId).hostData = tokenOffset_.data();
        node.variantPack.inTensors.at(seqLenTensorId).hostData = seqLen_.data();
    }
    ATB_LOG(INFO) << "BindParamHostTensor end";
    return atb::NO_ERROR;
}
} // namespace baichuan2_7b
} // namespace atb_speed