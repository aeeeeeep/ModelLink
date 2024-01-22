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
#include "models/chatglm2/6b/layer/paged_attention_layer.h"
#include "nlohmann/json.hpp"

namespace atb_speed {
namespace chatglm2_6b {
const int WEIGHT_COUNT_PER_LAYER = 7;
const int FINALNORMNODE_WEIGHT_COUNT = 1;
const int OPERATION_COUNT_AFTER_LAYER = 2;
const int INTERMEDIATETENSOR_COUNT_BEFORE_LAYER = 1;
const int OPERATION_COUNT_BEFORE_LAYER = 1;
const int OUT_LM_HEAD_WEIGHT_COUNT = 1;
const int WORDEMBEDDINGNODE_WEIGHT_COUNT = 1;

enum InTensorId : int {
    IN_TENSOR_HIDDENSTATES = 0,
    IN_TENSOR_ROPE,
    IN_ATTENTION_MASK,
    IN_TENSOR_BLOCK_TABLES,
    IN_TENSOR_SLOTS,
    IN_TENSOR_INPUT_LENGTHS,
    IN_TENSOR_MAX_SEQLEN,
    IN_TENSOR_LOGITS_INDICES,
    IN_TENSOR_MAX
};

enum OutTensorId : int {
    OUT_TENSOR_HIDDENSTATES = 0,
    OUT_TENSOR_MAX,
};

void PAModel::Param::FromString(const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);
    rmsNormEps = paramJson["rmsNormEps"].get<float>();
    headNum = paramJson["headNum"].get<int>();
    dk = paramJson["dk"].get<int>();
    numHeadsPerPartition = paramJson["numHeadsPerPartition"].get<int>();
    hiddenSizePerHead = paramJson["hiddenSizePerHead"].get<int>();
    numGroupsPerPartition = paramJson["numGroupsPerPartition"].get<int>();
    transKey = paramJson["transKey"].get<bool>();
    layerNum = paramJson["layerNum"].get<int>();
    rank = paramJson["rank"].get<int>();
    rankSize = paramJson["rankSize"].get<int>();
    residualAddScale = paramJson["residualAddScale"].get<float>();
    isPrefill = paramJson["isPrefill"].get<bool>();

    ATB_LOG(INFO) << "Chatglm2_6BPAModel param rmsNormEps:" << rmsNormEps << ", headNum:" << headNum
                  << ", dk:" << dk << ", layerNum:" << layerNum << ", isPrefill" << isPrefill;
}

PAModel::PAModel(const std::string &param) : Model("PAModel", param)
{
    ATB_LOG(INFO) << "check from string";
    param_.FromString(param);
    ATB_LOG(INFO) << "check from string";
}

PAModel::~PAModel() {}

uint32_t PAModel::GetInputNum() const { return graph_.inTensors.size(); }

uint32_t PAModel::GetOutputNum() const { return graph_.outTensors.size(); }

atb::Status PAModel::InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
                                std::vector<atb::TensorDesc> &outTensorDescs)
{
    if (outTensorDescs.size() != GetOutputNum()) {
        return atb::ERROR_INVALID_GRAPH;
    }

    const int64_t outDim = graph_.weightTensors.at(graph_.weightTensors.size() - 1).desc.shape.dims[0];
    ATB_LOG(INFO) << "check outdim" << outDim;
    outTensorDescs.at(0) = graph_.weightTensors.at(0).desc;
    outTensorDescs.at(0).shape.dimNum = 3;
    outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
    outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1];
    // ===========================================
    outTensorDescs.at(0).shape.dims[2] = outDim;
    // ===========================================
    return atb::NO_ERROR;
}

void PAModel::BuildGraph()
{
    ATB_LOG(INFO) << "START build Graph";
    const int weightTensorSize = WORDEMBEDDINGNODE_WEIGHT_COUNT + WEIGHT_COUNT_PER_LAYER * param_.layerNum +
                                 FINALNORMNODE_WEIGHT_COUNT + OUT_LM_HEAD_WEIGHT_COUNT;
    graph_.weightTensors.resize(weightTensorSize);
    graph_.inTensors.resize(IN_TENSOR_MAX + param_.layerNum * 2);
    graph_.outTensors.resize(OUT_TENSOR_MAX);
    const int nodeSize = param_.layerNum + OPERATION_COUNT_AFTER_LAYER + OPERATION_COUNT_BEFORE_LAYER;
    graph_.nodes.resize(nodeSize);

    const int internalTensorSize = graph_.nodes.size() - 1;
    graph_.internalTensors.resize(internalTensorSize);

    int nodeId = 0;
    auto &wordEmbeddingNode = graph_.nodes.at(nodeId++);
    atb::infer::GatherParam wordEmbeddingParam;
    atb::Operation *op = nullptr;
    atb::CreateOperation(wordEmbeddingParam, &op);
    wordEmbeddingNode.operation.reset(op);
    wordEmbeddingNode.inTensors = {&graph_.weightTensors.at(0), &graph_.inTensors.at(IN_TENSOR_HIDDENSTATES)};
    wordEmbeddingNode.outTensors = {&graph_.internalTensors.at(0)};

    atb::Tensor *firstInTensor = &graph_.internalTensors.at(0);
    for (int layerId = 0; layerId < param_.layerNum; ++layerId) {
        auto &layerNode = graph_.nodes.at(nodeId++);
        LayerParamPa opParam;
        opParam.rmsNormEps = param_.rmsNormEps;
        opParam.dk = param_.dk;
        opParam.headNum = param_.headNum;
        opParam.isPrefill = param_.isPrefill;
        opParam.numHeadsPerPartition = param_.numHeadsPerPartition;
        opParam.hiddenSizePerHead = param_.hiddenSizePerHead;
        opParam.numGroupsPerPartition = param_.numGroupsPerPartition;
        opParam.transKey = param_.transKey;
        opParam.layerId = layerId;
        opParam.residualAddScale = param_.residualAddScale;
        opParam.preScale = layerId + 1;
        opParam.postScale = layerId + 1;
        opParam.rank = param_.rank;
        opParam.rankSize = param_.rankSize;

        atb_speed::chatglm2_6b::DecoderPALayer(opParam, &op);
        ATB_LOG(INFO) << "FINISH init PA layer";
        layerNode.operation.reset(op);
        layerNode.inTensors.resize(layerNode.operation->GetInputNum());
        
        ATB_LOG(INFO) << "START loop";
        size_t inTensorId = 0;
        layerNode.inTensors.at(inTensorId++) = firstInTensor;
        for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
            layerNode.inTensors.at(inTensorId++) =
                &graph_.weightTensors.at(layerId * WEIGHT_COUNT_PER_LAYER + weightTensorId
                                        + WORDEMBEDDINGNODE_WEIGHT_COUNT);
        }
        ATB_LOG(INFO) << "check size" << layerNode.inTensors.size();
        ATB_LOG(INFO) << "check size 1 " <<  graph_.inTensors.size();
        
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ROPE);
        ATB_LOG(INFO) << "END ROPE";
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_ATTENTION_MASK);
        ATB_LOG(INFO) << "END ATTN MASK";
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_BLOCK_TABLES);
        ATB_LOG(INFO) << "END BLOCK TABLE";
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SLOTS);
        ATB_LOG(INFO) << "END SLOT";
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_INPUT_LENGTHS);
        ATB_LOG(INFO) << "END INPUT LENGTHS";
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_MAX_SEQLEN);
        ATB_LOG(INFO) << "END MAX SEQLEN";
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_LOGITS_INDICES);
        ATB_LOG(INFO) << "END SEQ LEN PRE";
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_MAX + layerId);
        ATB_LOG(INFO) << "END TENSOR MAX ADD" << "inTensorId:" << inTensorId;
        ATB_LOG(INFO) << "layerNum:" << param_.layerNum;
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_MAX + layerId + param_.layerNum);
        ATB_LOG(INFO) << "FINISH inTensors assign";
        layerNode.outTensors = {&graph_.internalTensors.at(INTERMEDIATETENSOR_COUNT_BEFORE_LAYER + layerId)};
        ATB_LOG(INFO) << "FINISH outTensor assign";
        firstInTensor = layerNode.outTensors.at(0);
    }
    ATB_LOG(INFO) << "FINISH for loop";

    auto &finalNormNode = graph_.nodes.at(nodeId++);
    atb::infer::RmsNormParam finalNormParam;
    finalNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    finalNormParam.normParam.epsilon = param_.rmsNormEps;
    atb::CreateOperation(finalNormParam, &op);
    finalNormNode.operation.reset(op);
    const int finalLayerNormWeightTensorId = graph_.weightTensors.size() -
                                            FINALNORMNODE_WEIGHT_COUNT - OUT_LM_HEAD_WEIGHT_COUNT;
    const int finalLayerNormOutTensorId = internalTensorSize - 1;
    finalNormNode.inTensors = {firstInTensor, &graph_.weightTensors.at(finalLayerNormWeightTensorId)};
    finalNormNode.outTensors = {&graph_.internalTensors.at(finalLayerNormOutTensorId)};

    auto &outLinearNode = graph_.nodes.at(nodeId++);
    atb::infer::LinearParam outLinearParm = {false, false, false};
    atb::CreateOperation(outLinearParm, &op);
    outLinearNode.operation.reset(op);
    const int finalLinearWeightTensorId = graph_.weightTensors.size() - OUT_LM_HEAD_WEIGHT_COUNT;
    outLinearNode.inTensors = {&graph_.internalTensors.at(finalLayerNormOutTensorId),
                               &graph_.weightTensors.at(finalLinearWeightTensorId)};
    outLinearNode.outTensors = {&graph_.outTensors.at(0)};
}

atb::Status PAModel::ParseParam(const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);
 
    seqLen_.clear();
    for (auto item : paramJson["seqLen"]) {
        seqLen_.push_back(item.get<int>());
    }
 
    ATB_LOG(INFO) << "PAModel ParseParam seqLen: " << seqLen_;
 
    return atb::NO_ERROR;
}

atb::Status PAModel::BindParamHostTensor(uint32_t nodeId)
{
    if (nodeId < OPERATION_COUNT_BEFORE_LAYER || nodeId >= OPERATION_COUNT_BEFORE_LAYER + param_.layerNum) {
        return atb::NO_ERROR;
    }

    auto &node = graph_.nodes.at(nodeId);
    const uint32_t seqLenTensorId = 12;  // IN_INPUT_LENGTHS of decoder layer
    node.variantPack.inTensors.at(seqLenTensorId).hostData = seqLen_.data();
 
    return atb::NO_ERROR;
}
} // namespace chatglm2_6b
} // namespace atb_speed
