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
#include "mixtral8x7B_dense/operation/mixtral_dense_mlp.h"

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
    INTERMIDATE_ROUTER_WEIGHTS_TOPK_SUMED,
    INTERMIDATE_ROUTER_WEIGHTS_TOPK_REDUCED,
    INTERMIDATE_EXPERT_MASK,
    INTERMIDATE_EXPERT_MASK_FLOAT16,
    INTERMIDATE_EXPERT_MASK_ZERO_THREE,
    INTERMIDATE_EXPERT_MASK_FOUR_SEVEN,
    INTERMIDATE_EXPERT_MASK_ZERO_ONE,
    INTERMIDATE_EXPERT_MASK_TWO_THREE,
    INTERMIDATE_EXPERT_MASK_FOUR_FIVE,
    INTERMIDATE_EXPERT_MASK_SIX_SEVEN,
    INTERMIDATE_EXPERT_MASK_ZERO,
    INTERMIDATE_EXPERT_MASK_ONE,
    INTERMIDATE_EXPERT_MASK_TWO,
    INTERMIDATE_EXPERT_MASK_THREE,
    INTERMIDATE_EXPERT_MASK_FOUR,
    INTERMIDATE_EXPERT_MASK_FIVE,
    INTERMIDATE_EXPERT_MASK_SIX,
    INTERMIDATE_EXPERT_MASK_SEVEN,
    INTERMIDATE_EXPERT_MASK_WITH_WEIGHT,
    INTERMIDATE_EXPERT_MASK_WITH_WEIGHT_TRANSPOSED,
    INTERMIDATE_FINAL_HIDDEN_STATE_ZERO,
    INTERMIDATE_FINAL_HIDDEN_STATE_ONE,
    INTERMIDATE_FINAL_HIDDEN_STATE_TWO,
    INTERMIDATE_FINAL_HIDDEN_STATE_THREE,
    INTERMIDATE_FINAL_HIDDEN_STATE_FOUR,
    INTERMIDATE_FINAL_HIDDEN_STATE_FIVE,
    INTERMIDATE_FINAL_HIDDEN_STATE_SIX,
};

static const uint64_t IN_TENSOR_COUNT = 21;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 31;
static const uint64_t NODE_COUNT = 24;

atb::Status CreateMixtralDenseMoeOperation(const MixtralDenseMoeParam &param, atb::Operation **operation)
{
    std::shared_ptr<int64_t> batchDimPtr = std::make_shared<int64_t>(0);

    atb::GraphParam opGraph;
    opGraph.name = "MixtralDenseMoe";
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    atb::Node &linearNode = opGraph.nodes.at(nodeId++);
    atb::Node &softMaxNode = opGraph.nodes.at(nodeId++);
    atb::Node &topKNode = opGraph.nodes.at(nodeId++);
    atb::Node &reduceNode = opGraph.nodes.at(nodeId++);
    atb::Node &divideNode = opGraph.nodes.at(nodeId++);
    atb::Node &onehotNode = opGraph.nodes.at(nodeId++);
    atb::Node &castNode = opGraph.nodes.at(nodeId++);
    atb::Node &weightMulNode = opGraph.nodes.at(nodeId++);
    atb::Node &transposeNode = opGraph.nodes.at(nodeId++);
    atb::Node &expertMaskSplit0Node = opGraph.nodes.at(nodeId++);
    atb::Node &expertMaskSplit1Node = opGraph.nodes.at(nodeId++);
    atb::Node &expertMaskSplit2Node = opGraph.nodes.at(nodeId++);
    atb::Node &expertMaskSplit3Node = opGraph.nodes.at(nodeId++);
    atb::Node &expertMaskSplit4Node = opGraph.nodes.at(nodeId++);
    atb::Node &expertMaskSplit5Node = opGraph.nodes.at(nodeId++);
    atb::Node &expertMaskSplit6Node = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpertZeroNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpertOneNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpertTwoNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpertThreeNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpertFourNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpertFiveNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpertSixNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpertSevenNode = opGraph.nodes.at(nodeId++);

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
    linearNode.inTensorReshapeFuncs.resize(linearNode.inTensorIds.size());
    linearNode.inTensorReshapeFuncs[0] = [batchDimPtr](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2; // dimNum: 2
        *batchDimPtr = oldShape.dims[0];
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
        newShape.dims[1] = oldShape.dims[2];
    };
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
    // Outt_tensor[0]: router_weights: Batch * Seq; 2
    // Outt_tensor[1]: selected_experts: Batch * Seq; 2
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

    // In_tensor[0]: router_weights: Batch * Seq; 2
    atb::infer::TransposeParam transposeParam;
    transposeParam.perm = {0, 2, 1};
    CreateOperation(transposeParam, &transposeNode.operation);
    transposeNode.inTensorIds = {INTERMIDATE_EXPERT_MASK_WITH_WEIGHT};
    transposeNode.outTensorIds = {INTERMIDATE_EXPERT_MASK_WITH_WEIGHT_TRANSPOSED};
    ATB_LOG(INFO) << "Router weights TRANSPOSED success";

    atb::infer::SplitParam splitParam = {0, 2};
    CreateOperation(splitParam, &expertMaskSplit0Node.operation);
    expertMaskSplit0Node.inTensorIds = {INTERMIDATE_EXPERT_MASK_WITH_WEIGHT_TRANSPOSED};
    expertMaskSplit0Node.outTensorIds = {
        INTERMIDATE_EXPERT_MASK_ZERO_THREE,
        INTERMIDATE_EXPERT_MASK_FOUR_SEVEN,
    };
    ATB_LOG(INFO) << "Expert Mask splited 0-3, 4-7 success";

    CreateOperation(splitParam, &expertMaskSplit1Node.operation);
    expertMaskSplit1Node.inTensorIds = {INTERMIDATE_EXPERT_MASK_ZERO_THREE};
    expertMaskSplit1Node.outTensorIds = {
        INTERMIDATE_EXPERT_MASK_ZERO_ONE,
        INTERMIDATE_EXPERT_MASK_TWO_THREE,
    };
    ATB_LOG(INFO) << "Expert Mask splited 0-1, 2-3 success";

    CreateOperation(splitParam, &expertMaskSplit2Node.operation);
    expertMaskSplit2Node.inTensorIds = {INTERMIDATE_EXPERT_MASK_ZERO_ONE};
    expertMaskSplit2Node.outTensorIds = {
        INTERMIDATE_EXPERT_MASK_ZERO,
        INTERMIDATE_EXPERT_MASK_ONE,
    };
    ATB_LOG(INFO) << "Expert Mask splited 0, 1 success";

    CreateOperation(splitParam, &expertMaskSplit3Node.operation);
    expertMaskSplit3Node.inTensorIds = {INTERMIDATE_EXPERT_MASK_TWO_THREE};
    expertMaskSplit3Node.outTensorIds = {
        INTERMIDATE_EXPERT_MASK_TWO,
        INTERMIDATE_EXPERT_MASK_THREE,
    };
    ATB_LOG(INFO) << "Expert Mask splited 2, 3 success";

    CreateOperation(splitParam, &expertMaskSplit4Node.operation);
    expertMaskSplit4Node.inTensorIds = {INTERMIDATE_EXPERT_MASK_FOUR_SEVEN};
    expertMaskSplit4Node.outTensorIds = {
        INTERMIDATE_EXPERT_MASK_FOUR_FIVE,
        INTERMIDATE_EXPERT_MASK_SIX_SEVEN,
    };
    ATB_LOG(INFO) << "Expert Mask splited 4-5, 6-7 success";

    CreateOperation(splitParam, &expertMaskSplit5Node.operation);
    expertMaskSplit5Node.inTensorIds = {INTERMIDATE_EXPERT_MASK_FOUR_FIVE};
    expertMaskSplit5Node.outTensorIds = {
        INTERMIDATE_EXPERT_MASK_FOUR,
        INTERMIDATE_EXPERT_MASK_FIVE,
    };
    ATB_LOG(INFO) << "Expert Mask splited 4, 5 success";

    CreateOperation(splitParam, &expertMaskSplit6Node.operation);
    expertMaskSplit6Node.inTensorIds = {INTERMIDATE_EXPERT_MASK_SIX_SEVEN};
    expertMaskSplit6Node.outTensorIds = {
        INTERMIDATE_EXPERT_MASK_SIX,
        INTERMIDATE_EXPERT_MASK_SEVEN,
    };
    ATB_LOG(INFO) << "Expert Mask splited 6, 7 success";

    atb_speed::mixtralDense::MixtralDenseMlpParam MlpExpertZeroParam;
    MlpExpertZeroParam.transpose = param.transpose;
    mixtralDense::CreateMixtralDenseMlpOperation(MlpExpertZeroParam, &mlpExpertZeroNode.operation);
    mlpExpertZeroNode.inTensorIds = {IN_HIDDEN_STATE,
                                        IN_MLPGATEUPWEIGHT_EXPERT_ZERO,
                                        IN_MLPDOWNWEIGHT_EXPERT_ZERO,
                                        INTERMIDATE_EXPERT_MASK_ZERO,
                                        IN_FINAL_HIDDEN_STATE};
    mlpExpertZeroNode.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_ZERO};
    ATB_LOG(INFO) << "Expert zero calculation success";

    atb_speed::mixtralDense::MixtralDenseMlpParam MlpExpertOneParam;
    MlpExpertOneParam.transpose = param.transpose;
    mixtralDense::CreateMixtralDenseMlpOperation(MlpExpertOneParam, &mlpExpertOneNode.operation);
    mlpExpertOneNode.inTensorIds = {IN_HIDDEN_STATE,
                                    IN_MLPGATEUPWEIGHT_EXPERT_ONE,
                                    IN_MLPDOWNWEIGHT_EXPERT_ONE,
                                    INTERMIDATE_EXPERT_MASK_ONE,
                                    INTERMIDATE_FINAL_HIDDEN_STATE_ZERO};
    mlpExpertOneNode.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_ONE};
    ATB_LOG(INFO) << "Expert one calculation success";

    atb_speed::mixtralDense::MixtralDenseMlpParam MlpExpertTwoParam;
    MlpExpertTwoParam.transpose = param.transpose;
    mixtralDense::CreateMixtralDenseMlpOperation(MlpExpertTwoParam, &mlpExpertTwoNode.operation);
    mlpExpertTwoNode.inTensorIds = {IN_HIDDEN_STATE,
                                    IN_MLPGATEUPWEIGHT_EXPERT_TWO,
                                    IN_MLPDOWNWEIGHT_EXPERT_TWO,
                                    INTERMIDATE_EXPERT_MASK_TWO,
                                    INTERMIDATE_FINAL_HIDDEN_STATE_ONE};
    mlpExpertTwoNode.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_TWO};
    ATB_LOG(INFO) << "Expert two calculation success";

    atb_speed::mixtralDense::MixtralDenseMlpParam MlpExpertThreeParam;
    MlpExpertThreeParam.transpose = param.transpose;
    mixtralDense::CreateMixtralDenseMlpOperation(MlpExpertThreeParam, &mlpExpertThreeNode.operation);
    mlpExpertThreeNode.inTensorIds = {IN_HIDDEN_STATE,
                                        IN_MLPGATEUPWEIGHT_EXPERT_THREE,
                                        IN_MLPDOWNWEIGHT_EXPERT_THREE,
                                        INTERMIDATE_EXPERT_MASK_THREE,
                                        INTERMIDATE_FINAL_HIDDEN_STATE_TWO};
    mlpExpertThreeNode.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_THREE};
    ATB_LOG(INFO) << "Expert three calculation success";

    atb_speed::mixtralDense::MixtralDenseMlpParam MlpExpertFourParam;
    MlpExpertFourParam.transpose = param.transpose;
    mixtralDense::CreateMixtralDenseMlpOperation(MlpExpertFourParam, &mlpExpertFourNode.operation);
    mlpExpertFourNode.inTensorIds = {IN_HIDDEN_STATE,
                                        IN_MLPGATEUPWEIGHT_EXPERT_FOUR,
                                        IN_MLPDOWNWEIGHT_EXPERT_FOUR,
                                        INTERMIDATE_EXPERT_MASK_FOUR,
                                        INTERMIDATE_FINAL_HIDDEN_STATE_THREE};
    mlpExpertFourNode.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_FOUR};
    ATB_LOG(INFO) << "Expert four calculation success";

    atb_speed::mixtralDense::MixtralDenseMlpParam MlpExpertFiveParam;
    MlpExpertFiveParam.transpose = param.transpose;
    mixtralDense::CreateMixtralDenseMlpOperation(MlpExpertFiveParam, &mlpExpertFiveNode.operation);
    mlpExpertFiveNode.inTensorIds = {IN_HIDDEN_STATE,
                                        IN_MLPGATEUPWEIGHT_EXPERT_FIVE,
                                        IN_MLPDOWNWEIGHT_EXPERT_FIVE,
                                        INTERMIDATE_EXPERT_MASK_FIVE,
                                        INTERMIDATE_FINAL_HIDDEN_STATE_FOUR};
    mlpExpertFiveNode.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_FIVE};
    ATB_LOG(INFO) << "Expert five calculation success";

    atb_speed::mixtralDense::MixtralDenseMlpParam MlpExpertSixParam;
    MlpExpertSixParam.transpose = param.transpose;
    mixtralDense::CreateMixtralDenseMlpOperation(MlpExpertSixParam, &mlpExpertSixNode.operation);
    mlpExpertSixNode.inTensorIds = {IN_HIDDEN_STATE,
                                    IN_MLPGATEUPWEIGHT_EXPERT_SIX,
                                    IN_MLPDOWNWEIGHT_EXPERT_SIX,
                                    INTERMIDATE_EXPERT_MASK_SIX,
                                    INTERMIDATE_FINAL_HIDDEN_STATE_FIVE};
    mlpExpertSixNode.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_SIX};
    ATB_LOG(INFO) << "Expert six calculation success";

    atb_speed::mixtralDense::MixtralDenseMlpParam MlpExpertSevenParam;
    MlpExpertSevenParam.transpose = param.transpose;
    mixtralDense::CreateMixtralDenseMlpOperation(MlpExpertSevenParam, &mlpExpertSevenNode.operation);
    mlpExpertSevenNode.inTensorIds = {IN_HIDDEN_STATE,
                                        IN_MLPGATEUPWEIGHT_EXPERT_SEVEN,
                                        IN_MLPDOWNWEIGHT_EXPERT_SEVEN,
                                        INTERMIDATE_EXPERT_MASK_SEVEN,
                                        INTERMIDATE_FINAL_HIDDEN_STATE_SIX};
    mlpExpertSevenNode.outTensorIds = {OUT_MIXTRAL_DENSE_MOE_ROUT};
    ATB_LOG(INFO) << "Expert seven calculation success";

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                    atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
    };

    return atb::CreateOperation(opGraph, operation);
}
}
} // namespace atb_speed