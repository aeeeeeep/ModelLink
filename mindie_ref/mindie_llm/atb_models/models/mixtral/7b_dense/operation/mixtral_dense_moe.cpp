/*
* Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
*
*  * Licensed under the Apache License, Version 2.0 (the "License");
*  * you may not use this file except in compliance with the License.
*  * You may obtain a copy of the License at
*  *
*  * http://www.apache.org/licenses/LICENSE-2.0
*  *
*  * Unless required by applicable law or agreed to in writing, software
*  * distributed under the License is distributed on an "AS IS" BASIS,
*  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  * See the License for the specific language governing permissions and
*  * limitations under the License.
*  */
#include "mixtral_dense_moe.h"
#include <atb/atb_infer.h>
#include <memory>
#include "mixtral/7b_dense/operation/mixtral_dense_mlp.h"
#include "mixtral/7b_dense/operation/mixtral_dense_mask_split.h"

namespace atb_speed {
namespace mixtralDense {
enum MixtralDenseMoeTensorId {
    IN_HIDDEN_STATE = 0,
    IN_GATE_WEIGHT,
    IN_MLPGATEUPWEIGHT_EXPERT_ZERO,
    IN_MLPDOWNWEIGHT_EXPERT_ZERO,
    IN_MLPGATEUPWEIGHT_EXPERT_ONE,
    IN_MLPDOWNWEIGHT_EXPERT_ONE,
    IN_MLPGATEUPWEIGHT_EXPERT_TWO,
    IN_MLPDOWNWEIGHT_EXPERT_TWO,
    IN_MLPGATEUPWEIGHT_EXPERT_THREE,
    IN_MLPDOWNWEIGHT_EXPERT_THREE,
    IN_MLPGATEUPWEIGHT_EXPERT_FOUR,
    IN_MLPDOWNWEIGHT_EXPERT_FOUR,
    IN_MLPGATEUPWEIGHT_EXPERT_FIVE,
    IN_MLPDOWNWEIGHT_EXPERT_FIVE,
    IN_MLPGATEUPWEIGHT_EXPERT_SIX,
    IN_MLPDOWNWEIGHT_EXPERT_SIX,
    IN_MLPGATEUPWEIGHT_EXPERT_SEVEN,
    IN_MLPDOWNWEIGHT_EXPERT_SEVEN,
    IN_SCALER_ONE,
    IN_SCALER_ZERO,
    IN_FINAL_HIDDEN_STATE,
    OUT_MIXTRAL_DENSE_MOE_ROUT,
    INTERMIDATE_ROUTER_LOGITS,
    INTERMIDATE_ROUTER_WEIGHTS,
    INTERMIDATE_ROUTER_WEIGHTS_TOPK,
    INTERMIDATE_SELECTED_EXPERTS,
    INTERMIDATE_EXPERT_MASK,
    INTERMIDATE_EXPERT_MASK_FLOAT16,
    INTERMIDATE_EXPERT_MASK_ZERO,
    INTERMIDATE_EXPERT_MASK_ONE,
    INTERMIDATE_EXPERT_MASK_TWO,
    INTERMIDATE_EXPERT_MASK_THREE,
    INTERMIDATE_EXPERT_MASK_FOUR,
    INTERMIDATE_EXPERT_MASK_FIVE,
    INTERMIDATE_EXPERT_MASK_SIX,
    INTERMIDATE_EXPERT_MASK_SEVEN,
    INTERMIDATE_ROUTER_WEIGHTS_TOPK_SUMED,
    INTERMIDATE_ROUTER_WEIGHTS_TOPK_REDUCED,
    INTERMIDATE_EXPERT_MASK_WITH_WEIGHT,
    INTERMIDATE_FINAL_HIDDEN_STATE_ZERO,
    INTERMIDATE_FINAL_HIDDEN_STATE_ONE,
    INTERMIDATE_FINAL_HIDDEN_STATE_TWO,
    INTERMIDATE_FINAL_HIDDEN_STATE_THREE,
    INTERMIDATE_FINAL_HIDDEN_STATE_FOUR,
    INTERMIDATE_FINAL_HIDDEN_STATE_FIVE,
    INTERMIDATE_FINAL_HIDDEN_STATE_SIX,
    INTERMIDATE_FINAL_HIDDEN_STATE_SEVEN,
};

static const uint64_t IN_TENSOR_COUNT = 21;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT_BEFORE_MLP = 17;
static const uint64_t OPERATION_COUNT_BEFORE_MLP = 9;

atb::Status CreateMixtralDenseMoeOperation(const MixtralDenseMoeParam &param, atb::Operation **operation)
{
    std::shared_ptr<int64_t> batchDimPtr = std::make_shared<int64_t>(0);

    atb::GraphParam opGraph;
    opGraph.name = "MixtralDenseMoe";
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT_BEFORE_MLP + param.numOfExperts / param.expertParallelDegree - 1;
    const int nodeSize = param.numOfExperts / param.expertParallelDegree + OPERATION_COUNT_BEFORE_MLP;
    opGraph.nodes.resize(nodeSize);

    size_t nodeId = 0;
    atb::Node &linearNode = opGraph.nodes.at(nodeId++);
    atb::Node &softMaxNode = opGraph.nodes.at(nodeId++);
    atb::Node &topKNode = opGraph.nodes.at(nodeId++);
    atb::Node &reduceNode = opGraph.nodes.at(nodeId++);
    atb::Node &divideNode = opGraph.nodes.at(nodeId++);
    atb::Node &onehotNode = opGraph.nodes.at(nodeId++);
    atb::Node &castNode = opGraph.nodes.at(nodeId++);
    atb::Node &weightMulNode = opGraph.nodes.at(nodeId++);
    atb::Node &expertMaskSplitNode = opGraph.nodes.at(nodeId++);

    // In_tensor[0]: hiddesn_state: Batch; seq_len; hidden_dim, reshaped to Batch * Seq; hidden_dim
    // In_tensor[1]: Gate_weight: num_experts(8); hidden_dim
    // Out_tensor[0]: router_logits: Batch * Seq; num_experts
    atb::infer::LinearParam moeGateParam;
    moeGateParam.transposeA = false;
    moeGateParam.transposeB = param.transpose;
    moeGateParam.hasBias = false;
    CreateOperation(moeGateParam, &linearNode.operation);
    linearNode.inTensorIds = {IN_HIDDEN_STATE, IN_GATE_WEIGHT};
    linearNode.outTensorIds = {INTERMIDATE_ROUTER_LOGITS};
    ATB_LOG(INFO) << "Router logits calculation success";

    // In_tensor[0]: router_logits: Batch * Seq; num_experts
    // Outt_tensor[0]: router_weights: Batch * Seq; num_experts
    atb::infer::SoftmaxParam softMaxParam;
    softMaxParam.axes = param.axes;
    CreateOperation(softMaxParam, &softMaxNode.operation);
    softMaxNode.inTensorIds = {INTERMIDATE_ROUTER_LOGITS};
    softMaxNode.outTensorIds = {INTERMIDATE_ROUTER_WEIGHTS};
    ATB_LOG(INFO) << "Router weights calculation success";

    // In_tensor[0]: router_weights: Batch * Seq; num_experts
    // Outt_tensor[0]: router_weights: Batch * Seq; 6
    // Outt_tensor[1]: selected_experts: Batch * Seq; 6
    atb::infer::SortParam topKParam;
    topKParam.num = param.num;
    CreateOperation(topKParam, &topKNode.operation);
    topKNode.inTensorIds = {INTERMIDATE_ROUTER_WEIGHTS};
    topKNode.outTensorIds = {INTERMIDATE_ROUTER_WEIGHTS_TOPK, INTERMIDATE_SELECTED_EXPERTS};
    ATB_LOG(INFO) << "Expert selection success";

    // In_tensor[0]: router_weights: Batch * Seq; 2
    atb::infer::ReduceParam reduceParam;
    reduceParam.reduceType = atb::infer::ReduceParam::ReduceType::REDUCE_SUM;
    reduceParam.axis = {1};
    CreateOperation(reduceParam, &reduceNode.operation);
    reduceNode.inTensorIds = {INTERMIDATE_ROUTER_WEIGHTS_TOPK};
    reduceNode.outTensorIds = {INTERMIDATE_ROUTER_WEIGHTS_TOPK_SUMED};
    ATB_LOG(INFO) << "Reduce sum calculated success";

    // In_tensor[0]: router_weights: Batch * Seq; 2
    atb::infer::ElewiseParam divideParam;
    divideParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_REALDIV;
    CreateOperation(divideParam, &divideNode.operation);
    divideNode.inTensorIds = {INTERMIDATE_ROUTER_WEIGHTS_TOPK, INTERMIDATE_ROUTER_WEIGHTS_TOPK_SUMED};
    divideNode.outTensorIds = {INTERMIDATE_ROUTER_WEIGHTS_TOPK_REDUCED};
    divideNode.inTensorReshapeFuncs.resize(divideNode.inTensorIds.size());
    divideNode.inTensorReshapeFuncs[1] = [batchDimPtr](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2; // dimNum: 2
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = 1;
    };
    ATB_LOG(INFO) << "Router weights calculated success";

    atb::infer::OnehotParam onehotParam;
    onehotParam.depth = param.numOfExperts;
    CreateOperation(onehotParam, &onehotNode.operation);
    onehotNode.inTensorIds = {INTERMIDATE_SELECTED_EXPERTS, IN_SCALER_ONE, IN_SCALER_ZERO};
    onehotNode.outTensorIds = {INTERMIDATE_EXPERT_MASK};
    ATB_LOG(INFO) << "Expert Mask created success";

    atb::infer::ElewiseParam castParam;
    castParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_CAST;
    castParam.outTensorType = ACL_FLOAT16;
    CreateOperation(castParam, &castNode.operation);
    castNode.inTensorIds = {INTERMIDATE_EXPERT_MASK};
    castNode.outTensorIds = {INTERMIDATE_EXPERT_MASK_FLOAT16};

    atb::infer::ElewiseParam weightMulParam;
    weightMulParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    CreateOperation(weightMulParam, &weightMulNode.operation);
    weightMulNode.inTensorIds = {INTERMIDATE_ROUTER_WEIGHTS_TOPK_REDUCED, INTERMIDATE_EXPERT_MASK_FLOAT16};
    weightMulNode.outTensorIds = {INTERMIDATE_EXPERT_MASK_WITH_WEIGHT};

    mixtralDense::MixtralDenseMaskSplitParam splitParam;
    CreateMixtralDenseMaskSplitOperation(splitParam, &expertMaskSplitNode.operation);
    expertMaskSplitNode.inTensorIds = {INTERMIDATE_EXPERT_MASK_WITH_WEIGHT};
    expertMaskSplitNode.outTensorIds = {
        INTERMIDATE_EXPERT_MASK_ZERO,
        INTERMIDATE_EXPERT_MASK_ONE,
        INTERMIDATE_EXPERT_MASK_TWO,
        INTERMIDATE_EXPERT_MASK_THREE,
        INTERMIDATE_EXPERT_MASK_FOUR,
        INTERMIDATE_EXPERT_MASK_FIVE,
        INTERMIDATE_EXPERT_MASK_SIX,
        INTERMIDATE_EXPERT_MASK_SEVEN,
    };
    ATB_LOG(INFO) << "Expert Mask splited 0-7 success";
    
    for (int expertId = 0; expertId < (param.numOfExperts / param.expertParallelDegree); ++expertId) {
        auto &expertNode = opGraph.nodes.at(nodeId++);
        ATB_LOG(INFO) << "Expert created " << expertId;
        atb_speed::mixtralDense::MixtralDenseMlpParam mlpExpertParam;
        mlpExpertParam.transpose = param.transpose;
        mixtralDense::CreateMixtralDenseMlpOperation(mlpExpertParam, &expertNode.operation);
        uint mlpGateUpWeightIdx = IN_MLPGATEUPWEIGHT_EXPERT_ZERO + expertId * 2;
        uint mlpDownWeightIdx = IN_MLPDOWNWEIGHT_EXPERT_ZERO + expertId * 2;
        uint finalHiddenStateIdx = INTERMIDATE_FINAL_HIDDEN_STATE_ZERO + expertId - 1;
        if (expertId == 0) {
            finalHiddenStateIdx = IN_FINAL_HIDDEN_STATE;
        }
        uint expertMaskIdx = INTERMIDATE_EXPERT_MASK_ZERO + expertId + param.maskStartIdx * param.numOfExperts / param.expertParallelDegree;
        uint outTensorIdx = OUT_MIXTRAL_DENSE_MOE_ROUT;
        if (expertId != (param.numOfExperts / param.expertParallelDegree - 1)) {
            outTensorIdx = INTERMIDATE_FINAL_HIDDEN_STATE_ZERO + expertId;
        }
        expertNode.inTensorIds = {
            IN_HIDDEN_STATE,
            mlpGateUpWeightIdx,
            mlpDownWeightIdx,
            expertMaskIdx,
            finalHiddenStateIdx};
        expertNode.outTensorIds = {outTensorIdx};
        ATB_LOG(INFO) << "Expert " << expertId << " calculation success";
    }

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                    atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
    };

    return atb::CreateOperation(opGraph, operation);
}
}
} // namespace atb_speed