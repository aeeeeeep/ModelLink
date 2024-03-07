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
#include "nlohmann/json.hpp"

#include "atb_speed/utils/operation_util.h"
#include "layers/parallel_layer_v2.h"
#include "models/baichuan2/13b/layer/flash_attention_layer.h"
#include "models/baichuan2/13b/layer/flash_attention_quant_layer.h"
#include "models/baichuan2/13b/layer/flash_attention_quant_oper_layer.h"

namespace atb_speed {
namespace baichuan2_13b {
const int WEIGHT_COUNT_PER_LAYER = 17;
const int WORD_EMBEDDING_NODE_WEIGHT_COUNT = 1;
const int FINAL_NORM_NODE_WEIGHT_COUNT = 1;
const int OUT_LM_HEAD_WEIGHT_COUNT = 1;
const int OPERATION_COUNT_BEFORE_LAYER = 1;
const int INTERNAL_TENSOR_COUNT_BEFORE_LAYER = 1;
const int OPERATION_COUNT_AFTER_LAYER = 4;
const int ROLLBACK_WEIGHT_COUNT_PER_LAYER = 7;

enum InTensorId : int {
    IN_TENSOR_INPUT_IDS = 0,
    IN_TENSOR_ATTENTION_MASK,
    IN_TENSOR_PAST_KEY,
    IN_TENSOR_PAST_VALUE,
    IN_TENSOR_TOKEN_OFFSET,
    IN_TENSOR_SEQ_LEN,
    IN_BETA,
    IN_HOLDER,
    IN_FINAL_NORM_SLICE_OFFSET,
    IN_TENSOR_MAX, // 9
};

enum OutTensorId : int {
    OUT_TENSOR_HIDDEN_STATES = 0,
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
    if (paramJson.contains("backend")) {
        backend = paramJson["backend"];
    }
    if (paramJson.contains("rankSize")) {
        rankSize = paramJson["rankSize"].get<int>();
    }
    if (paramJson.contains("isOpera")) {
        isOpera = paramJson["isOpera"].get<bool>();
    }
    for (const auto &item : paramJson["w_packInputScale"]) {
        wPackInputScale.push_back(item.get<float>());
    }
    for (const auto &item : paramJson["w_packInputOffset"]) {
        wPackInputOffset.push_back(item.get<int>());
    }
    for (const auto &item : paramJson["o_projInputScale"]) {
        oProjInputScale.push_back(item.get<float>());
    }
    for (const auto &item : paramJson["o_projInputOffset"]) {
        oProjInputOffset.push_back(item.get<int>());
    }
    for (const auto &item : paramJson["gate_projInputScale"]) {
        gateProjInputScale.push_back(item.get<float>());
    }
    for (const auto &item : paramJson["gate_projInputOffset"]) {
        gateProjInputOffset.push_back(item.get<int>());
    }
    for (const auto &item : paramJson["down_projInputScale"]) {
        downProjInputScale.push_back(item.get<float>());
    }
    for (const auto &item : paramJson["down_projInputOffset"]) {
        downProjInputOffset.push_back(item.get<int>());
    }
    for (const auto &item : paramJson["up_projInputScale"]) {
        upProjInputScale.push_back(item.get<float>());
    }
    for (const auto &item : paramJson["up_projInputOffset"]) {
        upProjInputOffset.push_back(item.get<int>());
    }
    for (const auto &item : paramJson["roll_back_layer"]) {
        rollBackLayer.push_back(item.get<int>());
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
    outTensorDescs.at(0).shape.dims[1] = 1; // [batchSize, 1, logits]
    outTensorDescs.at(0).shape.dims[2] = outDim;

    return atb::NO_ERROR;
}

int64_t FlashAttentionQuantModel::BuildGraph()
{
    int rollbackLayerLength = param_.rollBackLayer.size();
    const int weightTensorSize = WORD_EMBEDDING_NODE_WEIGHT_COUNT +
                                 ROLLBACK_WEIGHT_COUNT_PER_LAYER * rollbackLayerLength +
                                 WEIGHT_COUNT_PER_LAYER * (param_.layerNum - rollbackLayerLength) +
                                 FINAL_NORM_NODE_WEIGHT_COUNT + OUT_LM_HEAD_WEIGHT_COUNT; // 1 +17*layerNum + 1 +1
    graph_.weightTensors.resize(weightTensorSize);

    graph_.inTensors.resize(IN_TENSOR_MAX + param_.layerNum);
    graph_.outTensors.resize(OUT_TENSOR_MAX);

    const int nodeSize = param_.layerNum + OPERATION_COUNT_BEFORE_LAYER + OPERATION_COUNT_AFTER_LAYER; // 20+ 1 +3
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
    int layerTmpId = 0;
    for (int layerId = 0; layerId < param_.layerNum; ++layerId) {
        auto &layerNode = graph_.nodes.at(nodeId++);
        if (std::find(std::begin(param_.rollBackLayer), std::end(param_.rollBackLayer), layerId) !=
            std::end(param_.rollBackLayer)) {
            atb_speed::baichuan2_13b::FlashAttentionLayerParam modelParamRollback;
            modelParamRollback.rmsNormEps = param_.rmsNormEps;
            modelParamRollback.headNum = param_.headNum;
            modelParamRollback.dk = param_.dk;
            modelParamRollback.rank = param_.rank;
            modelParamRollback.rankSize = param_.rankSize;
            modelParamRollback.backend = param_.backend;

            atb_speed::baichuan2_13b::FlashAttentionLayer(modelParamRollback, &op);
            layerNode.operation.reset(op);
            layerNode.inTensors.resize(layerNode.operation->GetInputNum());
            layerNode.outTensors.resize(layerNode.operation->GetOutputNum());

            size_t inTensorId = 0;
            layerNode.inTensors.at(inTensorId++) = firstInTensor;
            for (size_t weightTensorId = 0; weightTensorId < ROLLBACK_WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
                layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(
                    (layerId - layerTmpId) * WEIGHT_COUNT_PER_LAYER + layerTmpId * ROLLBACK_WEIGHT_COUNT_PER_LAYER +
                    weightTensorId + WORD_EMBEDDING_NODE_WEIGHT_COUNT);
            }

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
            if (param_.isOpera) {
                atb_speed::baichuan2_13b::FlashAttentionQuantOperLayerParam opParam;
                // param_ -> 通过param传入
                opParam.rmsNormEps = param_.rmsNormEps;
                opParam.headNum = param_.headNum;
                opParam.dk = param_.dk;
                opParam.rank = param_.rank;
                opParam.rankSize = param_.rankSize;
                opParam.backend = param_.backend;
                opParam.wPackInputScale = param_.wPackInputScale[layerId];
                opParam.wPackInputOffset = param_.wPackInputOffset[layerId];
                opParam.oProjInputScale = param_.oProjInputScale[layerId];
                opParam.oProjInputOffset = param_.oProjInputOffset[layerId];
                opParam.gateProjInputScale = param_.gateProjInputScale[layerId];
                opParam.gateProjInputOffset = param_.gateProjInputOffset[layerId];
                opParam.downProjInputScale = param_.downProjInputScale[layerId];
                opParam.downProjInputOffset = param_.downProjInputOffset[layerId];
                atb_speed::baichuan2_13b::FlashAttentionQuantOperLayer(opParam, &op);
            } else {
                atb_speed::baichuan2_13b::FlashAttentionQuantLayerParam opParam;
                // param_ -> 通过param传入
                opParam.rmsNormEps = param_.rmsNormEps;
                opParam.headNum = param_.headNum;
                opParam.dk = param_.dk;
                opParam.rank = param_.rank;
                opParam.rankSize = param_.rankSize;
                opParam.backend = param_.backend;
                opParam.wPackInputScale = param_.wPackInputScale[layerId];
                opParam.wPackInputOffset = param_.wPackInputOffset[layerId];
                opParam.oProjInputScale = param_.oProjInputScale[layerId];
                opParam.oProjInputOffset = param_.oProjInputOffset[layerId];
                opParam.gateProjInputScale = param_.gateProjInputScale[layerId];
                opParam.gateProjInputOffset = param_.gateProjInputOffset[layerId];
                opParam.downProjInputScale = param_.downProjInputScale[layerId];
                opParam.downProjInputOffset = param_.downProjInputOffset[layerId];
                atb_speed::baichuan2_13b::FlashAttentionQuantLayer(opParam, &op);
            }

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
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ATTENTION_MASK);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PAST_KEY);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PAST_VALUE);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_TOKEN_OFFSET);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SEQ_LEN);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_BETA);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_HOLDER);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_MAX + layerId);

            layerNode.outTensors = {&graph_.internalTensors.at(INTERNAL_TENSOR_COUNT_BEFORE_LAYER + layerId)}; //

            firstInTensor = layerNode.outTensors.at(0);
        }
    }

    // float operation
    auto &finalNormNode = graph_.nodes.at(nodeId++);
    atb::infer::RmsNormParam finalNormParam;
    finalNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    finalNormParam.normParam.epsilon = param_.rmsNormEps;
    CREATE_OPERATION(finalNormParam, &op);
    finalNormNode.operation.reset(op);
    const int finalLayerNormWeightTensorId =
        graph_.weightTensors.size() - FINAL_NORM_NODE_WEIGHT_COUNT - OUT_LM_HEAD_WEIGHT_COUNT;
    const int finalLayerNormOutTensorId = internalTensorSize - 3;
    finalNormNode.inTensors = {firstInTensor, &graph_.weightTensors.at(finalLayerNormWeightTensorId)};
    finalNormNode.outTensors = {&graph_.internalTensors.at(finalLayerNormOutTensorId)};

    auto &gatherNode = graph_.nodes.at(nodeId++);
    atb::infer::GatherParam gatherParam;
    gatherParam.axis = 1;
    CREATE_OPERATION(gatherParam, &op);
    gatherNode.operation.reset(op);
    const int gatherOutTensorId = internalTensorSize - 2;
    gatherNode.inTensors = {&graph_.internalTensors.at(finalLayerNormOutTensorId),
                            &graph_.inTensors.at(IN_FINAL_NORM_SLICE_OFFSET)};
    gatherNode.outTensors = {&graph_.internalTensors.at(gatherOutTensorId)};

    const int hiddenSize = param_.headNum * param_.dk;
    auto &qPassSliceNode = graph_.nodes.at(nodeId++);
    atb::infer::SliceParam slicePassParam;
    slicePassParam.offsets = {0, 0, hiddenSize * param_.rank};
    slicePassParam.size = {-1, -1, hiddenSize};
    CREATE_OPERATION(slicePassParam, &op);
    qPassSliceNode.operation.reset(op);
    const int qPassSliceNodeOutTensorId = internalTensorSize - 1;
    qPassSliceNode.inTensors = {&graph_.internalTensors.at(gatherOutTensorId)};
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
    outLinearNode.outTensors = {&graph_.outTensors.at(OUT_TENSOR_HIDDEN_STATES)};
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
    const uint32_t floatTokenOffsetTensorId = 11;
    const uint32_t floatSeqLenTensorId = 12;
    const uint32_t tokenOffsetTensorId = 21; // 使用layer里的ID
    const uint32_t seqLenTensorId = 22;
    if (std::find(std::begin(param_.rollBackLayer), std::end(param_.rollBackLayer), nodeId - 1) !=
        std::end(param_.rollBackLayer)) {
        node.variantPack.inTensors.at(floatTokenOffsetTensorId).hostData = tokenOffset_.data();
        node.variantPack.inTensors.at(floatSeqLenTensorId).hostData = seqLen_.data();
    } else {
        node.variantPack.inTensors.at(tokenOffsetTensorId).hostData = tokenOffset_.data();
        node.variantPack.inTensors.at(seqLenTensorId).hostData = seqLen_.data();
    }
    ATB_LOG(INFO) << "BindParamHostTensor end";
    return atb::NO_ERROR;
}
} // namespace baichuan2_13b
} // namespace atb_speed