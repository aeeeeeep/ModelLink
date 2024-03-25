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
#include "deepseek_dense_moe.h"
#include <atb/atb_infer.h>
#include <memory>
#include "deepseek/16b_dense/operation/deepseek_dense_mlp.h"
#include "deepseek/16b_dense/operation/deepseek_dense_mask_split.h"

namespace atb_speed {
namespace deepseekDense {
enum DeepseekDenseMoeTensorId {
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
    IN_MLPGATEUPWEIGHT_EXPERT_EIGHT,
    IN_MLPDOWNWEIGHT_EXPERT_EIGHT,
    IN_MLPGATEUPWEIGHT_EXPERT_NINE,
    IN_MLPDOWNWEIGHT_EXPERT_NINE,
    IN_MLPGATEUPWEIGHT_EXPERT_TEN,
    IN_MLPDOWNWEIGHT_EXPERT_TEN,
    IN_MLPGATEUPWEIGHT_EXPERT_ELEVEN,
    IN_MLPDOWNWEIGHT_EXPERT_ELEVEN,
    IN_MLPGATEUPWEIGHT_EXPERT_TWELVE,
    IN_MLPDOWNWEIGHT_EXPERT_TWELVE,
    IN_MLPGATEUPWEIGHT_EXPERT_THIRTEEN,
    IN_MLPDOWNWEIGHT_EXPERT_THIRTEEN,
    IN_MLPGATEUPWEIGHT_EXPERT_FOURTEEN,
    IN_MLPDOWNWEIGHT_EXPERT_FOURTEEN,
    IN_MLPGATEUPWEIGHT_EXPERT_FIFTEEN,
    IN_MLPDOWNWEIGHT_EXPERT_FIFTEEN,
    IN_MLPGATEUPWEIGHT_EXPERT_SIXTEEN,
    IN_MLPDOWNWEIGHT_EXPERT_SIXTEEN,
    IN_MLPGATEUPWEIGHT_EXPERT_SEVENTEEN,
    IN_MLPDOWNWEIGHT_EXPERT_SEVENTEEN,
    IN_MLPGATEUPWEIGHT_EXPERT_EIGHTEEN,
    IN_MLPDOWNWEIGHT_EXPERT_EIGHTEEN,
    IN_MLPGATEUPWEIGHT_EXPERT_NINETEEN,
    IN_MLPDOWNWEIGHT_EXPERT_NINETEEN,
    IN_MLPGATEUPWEIGHT_EXPERT_TWENTY,
    IN_MLPDOWNWEIGHT_EXPERT_TWENTY,
    IN_MLPGATEUPWEIGHT_EXPERT_TWENTY_ONE,
    IN_MLPDOWNWEIGHT_EXPERT_TWENTY_ONE,
    IN_MLPGATEUPWEIGHT_EXPERT_TWENTY_TWO,
    IN_MLPDOWNWEIGHT_EXPERT_TWENTY_TWO,
    IN_MLPGATEUPWEIGHT_EXPERT_TWENTY_THREE,
    IN_MLPDOWNWEIGHT_EXPERT_TWENTY_THREE,
    IN_MLPGATEUPWEIGHT_EXPERT_TWENTY_FOUR,
    IN_MLPDOWNWEIGHT_EXPERT_TWENTY_FOUR,
    IN_MLPGATEUPWEIGHT_EXPERT_TWENTY_FIVE,
    IN_MLPDOWNWEIGHT_EXPERT_TWENTY_FIVE,
    IN_MLPGATEUPWEIGHT_EXPERT_TWENTY_SIX,
    IN_MLPDOWNWEIGHT_EXPERT_TWENTY_SIX,
    IN_MLPGATEUPWEIGHT_EXPERT_TWENTY_SEVEN,
    IN_MLPDOWNWEIGHT_EXPERT_TWENTY_SEVEN,
    IN_MLPGATEUPWEIGHT_EXPERT_TWENTY_EIGHT,
    IN_MLPDOWNWEIGHT_EXPERT_TWENTY_EIGHT,
    IN_MLPGATEUPWEIGHT_EXPERT_TWENTY_NINE,
    IN_MLPDOWNWEIGHT_EXPERT_TWENTY_NINE,
    IN_MLPGATEUPWEIGHT_EXPERT_THIRTY,
    IN_MLPDOWNWEIGHT_EXPERT_THIRTY,
    IN_MLPGATEUPWEIGHT_EXPERT_THIRTY_ONE,
    IN_MLPDOWNWEIGHT_EXPERT_THIRTY_ONE,
    IN_MLPGATEUPWEIGHT_EXPERT_THIRTY_TWO,
    IN_MLPDOWNWEIGHT_EXPERT_THIRTY_TWO,
    IN_MLPGATEUPWEIGHT_EXPERT_THIRTY_THREE,
    IN_MLPDOWNWEIGHT_EXPERT_THIRTY_THREE,
    IN_MLPGATEUPWEIGHT_EXPERT_THIRTY_FOUR,
    IN_MLPDOWNWEIGHT_EXPERT_THIRTY_FOUR,
    IN_MLPGATEUPWEIGHT_EXPERT_THIRTY_FIVE,
    IN_MLPDOWNWEIGHT_EXPERT_THIRTY_FIVE,
    IN_MLPGATEUPWEIGHT_EXPERT_THIRTY_SIX,
    IN_MLPDOWNWEIGHT_EXPERT_THIRTY_SIX,
    IN_MLPGATEUPWEIGHT_EXPERT_THIRTY_SEVEN,
    IN_MLPDOWNWEIGHT_EXPERT_THIRTY_SEVEN,
    IN_MLPGATEUPWEIGHT_EXPERT_THIRTY_EIGHT,
    IN_MLPDOWNWEIGHT_EXPERT_THIRTY_EIGHT,
    IN_MLPGATEUPWEIGHT_EXPERT_THIRTY_NINE,
    IN_MLPDOWNWEIGHT_EXPERT_THIRTY_NINE,
    IN_MLPGATEUPWEIGHT_EXPERT_FOURTY,
    IN_MLPDOWNWEIGHT_EXPERT_FOURTY,
    IN_MLPGATEUPWEIGHT_EXPERT_FOURTY_ONE,
    IN_MLPDOWNWEIGHT_EXPERT_FOURTY_ONE,
    IN_MLPGATEUPWEIGHT_EXPERT_FOURTY_TWO,
    IN_MLPDOWNWEIGHT_EXPERT_FOURTY_TWO,
    IN_MLPGATEUPWEIGHT_EXPERT_FOURTY_THREE,
    IN_MLPDOWNWEIGHT_EXPERT_FOURTY_THREE,
    IN_MLPGATEUPWEIGHT_EXPERT_FOURTY_FOUR,
    IN_MLPDOWNWEIGHT_EXPERT_FOURTY_FOUR,
    IN_MLPGATEUPWEIGHT_EXPERT_FOURTY_FIVE,
    IN_MLPDOWNWEIGHT_EXPERT_FOURTY_FIVE,
    IN_MLPGATEUPWEIGHT_EXPERT_FOURTY_SIX,
    IN_MLPDOWNWEIGHT_EXPERT_FOURTY_SIX,
    IN_MLPGATEUPWEIGHT_EXPERT_FOURTY_SEVEN,
    IN_MLPDOWNWEIGHT_EXPERT_FOURTY_SEVEN,
    IN_MLPGATEUPWEIGHT_EXPERT_FOURTY_EIGHT,
    IN_MLPDOWNWEIGHT_EXPERT_FOURTY_EIGHT,
    IN_MLPGATEUPWEIGHT_EXPERT_FOURTY_NINE,
    IN_MLPDOWNWEIGHT_EXPERT_FOURTY_NINE,
    IN_MLPGATEUPWEIGHT_EXPERT_FIFTY,
    IN_MLPDOWNWEIGHT_EXPERT_FIFTY,
    IN_MLPGATEUPWEIGHT_EXPERT_FIFTY_ONE,
    IN_MLPDOWNWEIGHT_EXPERT_FIFTY_ONE,
    IN_MLPGATEUPWEIGHT_EXPERT_FIFTY_TWO,
    IN_MLPDOWNWEIGHT_EXPERT_FIFTY_TWO,
    IN_MLPGATEUPWEIGHT_EXPERT_FIFTY_THREE,
    IN_MLPDOWNWEIGHT_EXPERT_FIFTY_THREE,
    IN_MLPGATEUPWEIGHT_EXPERT_FIFTY_FOUR,
    IN_MLPDOWNWEIGHT_EXPERT_FIFTY_FOUR,
    IN_MLPGATEUPWEIGHT_EXPERT_FIFTY_FIVE,
    IN_MLPDOWNWEIGHT_EXPERT_FIFTY_FIVE,
    IN_MLPGATEUPWEIGHT_EXPERT_FIFTY_SIX,
    IN_MLPDOWNWEIGHT_EXPERT_FIFTY_SIX,
    IN_MLPGATEUPWEIGHT_EXPERT_FIFTY_SEVEN,
    IN_MLPDOWNWEIGHT_EXPERT_FIFTY_SEVEN,
    IN_MLPGATEUPWEIGHT_EXPERT_FIFTY_EIGHT,
    IN_MLPDOWNWEIGHT_EXPERT_FIFTY_EIGHT,
    IN_MLPGATEUPWEIGHT_EXPERT_FIFTY_NINE,
    IN_MLPDOWNWEIGHT_EXPERT_FIFTY_NINE,
    IN_MLPGATEUPWEIGHT_EXPERT_SIXTY,
    IN_MLPDOWNWEIGHT_EXPERT_SIXTY,
    IN_MLPGATEUPWEIGHT_EXPERT_SIXTY_ONE,
    IN_MLPDOWNWEIGHT_EXPERT_SIXTY_ONE,
    IN_MLPGATEUPWEIGHT_EXPERT_SIXTY_TWO,
    IN_MLPDOWNWEIGHT_EXPERT_SIXTY_TWO,
    IN_MLPGATEUPWEIGHT_EXPERT_SIXTY_THREE,
    IN_MLPDOWNWEIGHT_EXPERT_SIXTY_THREE,
    IN_SCALER_ONE,
    IN_SCALER_ZERO,
    IN_FINAL_HIDDEN_STATE,
    OUT_DEEPSEEK_DENSE_MOE_ROUT,
    INTERMIDATE_ROUTER_LOGITS,
    INTERMIDATE_ROUTER_WEIGHTS,
    INTERMIDATE_ROUTER_WEIGHTS_TOPK,
    INTERMIDATE_SELECTED_EXPERTS,
    INTERMIDATE_EXPERT_MASK,
    INTERMIDATE_EXPERT_MASK_FLOAT16,
    INTERMIDATE_EXPERT_MASK_ZERO_SEVEN,
    INTERMIDATE_EXPERT_MASK_EIGHT_FIFTEEN,
    INTERMIDATE_EXPERT_MASK_SIXTEEN_TWENTY_THREE,
    INTERMIDATE_EXPERT_MASK_TWENTY_FOUR_THIRTY_ONE,
    INTERMIDATE_EXPERT_MASK_THIRTY_TWO_THIRTY_NINE,
    INTERMIDATE_EXPERT_MASK_FOURTY_FOURTY_SEVEN,
    INTERMIDATE_EXPERT_MASK_FOURTY_EIGHT_FIFTY_FIVE,
    INTERMIDATE_EXPERT_MASK_FIFTY_SIX_DIXTY_THREE,
    INTERMIDATE_EXPERT_MASK_ZERO,
    INTERMIDATE_EXPERT_MASK_ONE,
    INTERMIDATE_EXPERT_MASK_TWO,
    INTERMIDATE_EXPERT_MASK_THREE,
    INTERMIDATE_EXPERT_MASK_FOUR,
    INTERMIDATE_EXPERT_MASK_FIVE,
    INTERMIDATE_EXPERT_MASK_SIX,
    INTERMIDATE_EXPERT_MASK_SEVEN,
    INTERMIDATE_EXPERT_MASK_EIGHT,
    INTERMIDATE_EXPERT_MASK_NINE,
    INTERMIDATE_EXPERT_MASK_TEN,
    INTERMIDATE_EXPERT_MASK_ELEVEN,
    INTERMIDATE_EXPERT_MASK_TWELVE,
    INTERMIDATE_EXPERT_MASK_THIRTEEN,
    INTERMIDATE_EXPERT_MASK_FOURTEEN,
    INTERMIDATE_EXPERT_MASK_FIFTEEN,
    INTERMIDATE_EXPERT_MASK_SIXTEEN,
    INTERMIDATE_EXPERT_MASK_SEVENTEEN,
    INTERMIDATE_EXPERT_MASK_EIGHTEEN,
    INTERMIDATE_EXPERT_MASK_NINETEEN,
    INTERMIDATE_EXPERT_MASK_TWENTY,
    INTERMIDATE_EXPERT_MASK_TWENTY_ONE,
    INTERMIDATE_EXPERT_MASK_TWENTY_TWO,
    INTERMIDATE_EXPERT_MASK_TWENTY_THREE,
    INTERMIDATE_EXPERT_MASK_TWENTY_FOUR,
    INTERMIDATE_EXPERT_MASK_TWENTY_FIVE,
    INTERMIDATE_EXPERT_MASK_TWENTY_SIX,
    INTERMIDATE_EXPERT_MASK_TWENTY_SEVEN,
    INTERMIDATE_EXPERT_MASK_TWENTY_EIGHT,
    INTERMIDATE_EXPERT_MASK_TWENTY_NINE,
    INTERMIDATE_EXPERT_MASK_THIRTY,
    INTERMIDATE_EXPERT_MASK_THIRTY_ONE,
    INTERMIDATE_EXPERT_MASK_THIRTY_TWO,
    INTERMIDATE_EXPERT_MASK_THIRTY_THREE,
    INTERMIDATE_EXPERT_MASK_THIRTY_FOUR,
    INTERMIDATE_EXPERT_MASK_THIRTY_FIVE,
    INTERMIDATE_EXPERT_MASK_THIRTY_SIX,
    INTERMIDATE_EXPERT_MASK_THIRTY_SEVEN,
    INTERMIDATE_EXPERT_MASK_THIRTY_EIGHT,
    INTERMIDATE_EXPERT_MASK_THIRTY_NINE,
    INTERMIDATE_EXPERT_MASK_FOURTY,
    INTERMIDATE_EXPERT_MASK_FOURTY_ONE,
    INTERMIDATE_EXPERT_MASK_FOURTY_TWO,
    INTERMIDATE_EXPERT_MASK_FOURTY_THREE,
    INTERMIDATE_EXPERT_MASK_FOURTY_FOUR,
    INTERMIDATE_EXPERT_MASK_FOURTY_FIVE,
    INTERMIDATE_EXPERT_MASK_FOURTY_SIX,
    INTERMIDATE_EXPERT_MASK_FOURTY_SEVEN,
    INTERMIDATE_EXPERT_MASK_FOURTY_EIGHT,
    INTERMIDATE_EXPERT_MASK_FOURTY_NINE,
    INTERMIDATE_EXPERT_MASK_FIFTY,
    INTERMIDATE_EXPERT_MASK_FIFTY_ONE,
    INTERMIDATE_EXPERT_MASK_FIFTY_TWO,
    INTERMIDATE_EXPERT_MASK_FIFTY_THREE,
    INTERMIDATE_EXPERT_MASK_FIFTY_FOUR,
    INTERMIDATE_EXPERT_MASK_FIFTY_FIVE,
    INTERMIDATE_EXPERT_MASK_FIFTY_SIX,
    INTERMIDATE_EXPERT_MASK_FIFTY_SEVEN,
    INTERMIDATE_EXPERT_MASK_FIFTY_EIGHT,
    INTERMIDATE_EXPERT_MASK_FIFTY_NINE,
    INTERMIDATE_EXPERT_MASK_SIXTY,
    INTERMIDATE_EXPERT_MASK_SIXTY_ONE,
    INTERMIDATE_EXPERT_MASK_SIXTY_TWO,
    INTERMIDATE_EXPERT_MASK_SIXTY_THREE,
    INTERMIDATE_EXPERT_MASK_WITH_WEIGHT,
    INTERMIDATE_FINAL_HIDDEN_STATE_ZERO,
    INTERMIDATE_FINAL_HIDDEN_STATE_ONE,
    INTERMIDATE_FINAL_HIDDEN_STATE_TWO,
    INTERMIDATE_FINAL_HIDDEN_STATE_THREE,
    INTERMIDATE_FINAL_HIDDEN_STATE_FOUR,
    INTERMIDATE_FINAL_HIDDEN_STATE_FIVE,
    INTERMIDATE_FINAL_HIDDEN_STATE_SIX,
    INTERMIDATE_FINAL_HIDDEN_STATE_SEVEN,
    INTERMIDATE_FINAL_HIDDEN_STATE_EIGHT,
    INTERMIDATE_FINAL_HIDDEN_STATE_NINE,
    INTERMIDATE_FINAL_HIDDEN_STATE_TEN,
    INTERMIDATE_FINAL_HIDDEN_STATE_ELEVEN,
    INTERMIDATE_FINAL_HIDDEN_STATE_TWELVE,
    INTERMIDATE_FINAL_HIDDEN_STATE_THIRTEEN,
    INTERMIDATE_FINAL_HIDDEN_STATE_FOURTEEN,
    INTERMIDATE_FINAL_HIDDEN_STATE_FIFTEEN,
    INTERMIDATE_FINAL_HIDDEN_STATE_SIXTEEN,
    INTERMIDATE_FINAL_HIDDEN_STATE_SEVENTEEN,
    INTERMIDATE_FINAL_HIDDEN_STATE_EIGHTEEN,
    INTERMIDATE_FINAL_HIDDEN_STATE_NINETEEN,
    INTERMIDATE_FINAL_HIDDEN_STATE_TWENTY,
    INTERMIDATE_FINAL_HIDDEN_STATE_TWENTY_ONE,
    INTERMIDATE_FINAL_HIDDEN_STATE_TWENTY_TWO,
    INTERMIDATE_FINAL_HIDDEN_STATE_TWENTY_THREE,
    INTERMIDATE_FINAL_HIDDEN_STATE_TWENTY_FOUR,
    INTERMIDATE_FINAL_HIDDEN_STATE_TWENTY_FIVE,
    INTERMIDATE_FINAL_HIDDEN_STATE_TWENTY_SIX,
    INTERMIDATE_FINAL_HIDDEN_STATE_TWENTY_SEVEN,
    INTERMIDATE_FINAL_HIDDEN_STATE_TWENTY_EIGHT,
    INTERMIDATE_FINAL_HIDDEN_STATE_TWENTY_NINE,
    INTERMIDATE_FINAL_HIDDEN_STATE_THIRTY,
    INTERMIDATE_FINAL_HIDDEN_STATE_THIRTY_ONE,
    INTERMIDATE_FINAL_HIDDEN_STATE_THIRTY_TWO,
    INTERMIDATE_FINAL_HIDDEN_STATE_THIRTY_THREE,
    INTERMIDATE_FINAL_HIDDEN_STATE_THIRTY_FOUR,
    INTERMIDATE_FINAL_HIDDEN_STATE_THIRTY_FIVE,
    INTERMIDATE_FINAL_HIDDEN_STATE_THIRTY_SIX,
    INTERMIDATE_FINAL_HIDDEN_STATE_THIRTY_SEVEN,
    INTERMIDATE_FINAL_HIDDEN_STATE_THIRTY_EIGHT,
    INTERMIDATE_FINAL_HIDDEN_STATE_THIRTY_NINE,
    INTERMIDATE_FINAL_HIDDEN_STATE_FOURTY,
    INTERMIDATE_FINAL_HIDDEN_STATE_FOURTY_ONE,
    INTERMIDATE_FINAL_HIDDEN_STATE_FOURTY_TWO,
    INTERMIDATE_FINAL_HIDDEN_STATE_FOURTY_THREE,
    INTERMIDATE_FINAL_HIDDEN_STATE_FOURTY_FOUR,
    INTERMIDATE_FINAL_HIDDEN_STATE_FOURTY_FIVE,
    INTERMIDATE_FINAL_HIDDEN_STATE_FOURTY_SIX,
    INTERMIDATE_FINAL_HIDDEN_STATE_FOURTY_SEVEN,
    INTERMIDATE_FINAL_HIDDEN_STATE_FOURTY_EIGHT,
    INTERMIDATE_FINAL_HIDDEN_STATE_FOURTY_NINE,
    INTERMIDATE_FINAL_HIDDEN_STATE_FIFTY,
    INTERMIDATE_FINAL_HIDDEN_STATE_FIFTY_ONE,
    INTERMIDATE_FINAL_HIDDEN_STATE_FIFTY_TWO,
    INTERMIDATE_FINAL_HIDDEN_STATE_FIFTY_THREE,
    INTERMIDATE_FINAL_HIDDEN_STATE_FIFTY_FOUR,
    INTERMIDATE_FINAL_HIDDEN_STATE_FIFTY_FIVE,
    INTERMIDATE_FINAL_HIDDEN_STATE_FIFTY_SIX,
    INTERMIDATE_FINAL_HIDDEN_STATE_FIFTY_SEVEN,
    INTERMIDATE_FINAL_HIDDEN_STATE_FIFTY_EIGHT,
    INTERMIDATE_FINAL_HIDDEN_STATE_FIFTY_NINE,
    INTERMIDATE_FINAL_HIDDEN_STATE_SIXTY,
    INTERMIDATE_FINAL_HIDDEN_STATE_SIXTY_ONE,
    INTERMIDATE_FINAL_HIDDEN_STATE_SIXTY_TWO,
};

static const uint64_t IN_TENSOR_COUNT = 133;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT_BEFORE_MLP = 79;
static const uint64_t OPERATION_COUNT_BEFORE_MLP = 15;

atb::Status CreateDeepseekDenseMoeOperation(const DeepseekDenseMoeParam &param, atb::Operation **operation)
{
    std::shared_ptr<int64_t> batchDimPtr = std::make_shared<int64_t>(0);

    atb::GraphParam opGraph;
    opGraph.name = "DeepseekDenseMoe";
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT_BEFORE_MLP + param.numOfExperts / param.expertParallelDegree - 1;
    const int nodeSize = param.numOfExperts / param.expertParallelDegree + OPERATION_COUNT_BEFORE_MLP;
    opGraph.nodes.resize(nodeSize);

    size_t nodeId = 0;
    atb::Node &linearNode = opGraph.nodes.at(nodeId++);
    atb::Node &softMaxNode = opGraph.nodes.at(nodeId++);
    atb::Node &topKNode = opGraph.nodes.at(nodeId++);
    atb::Node &onehotNode = opGraph.nodes.at(nodeId++);
    atb::Node &castNode = opGraph.nodes.at(nodeId++);
    atb::Node &weightMulNode = opGraph.nodes.at(nodeId++);
    atb::Node &expertMaskSplit0Node = opGraph.nodes.at(nodeId++);
    atb::Node &expertMaskSplit1Node = opGraph.nodes.at(nodeId++);
    atb::Node &expertMaskSplit2Node = opGraph.nodes.at(nodeId++);
    atb::Node &expertMaskSplit3Node = opGraph.nodes.at(nodeId++);
    atb::Node &expertMaskSplit4Node = opGraph.nodes.at(nodeId++);
    atb::Node &expertMaskSplit5Node = opGraph.nodes.at(nodeId++);
    atb::Node &expertMaskSplit6Node = opGraph.nodes.at(nodeId++);
    atb::Node &expertMaskSplit7Node = opGraph.nodes.at(nodeId++);
    atb::Node &expertMaskSplit8Node = opGraph.nodes.at(nodeId++);

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
    weightMulNode.inTensorIds = {INTERMIDATE_ROUTER_WEIGHTS_TOPK, INTERMIDATE_EXPERT_MASK_FLOAT16};
    weightMulNode.outTensorIds = {INTERMIDATE_EXPERT_MASK_WITH_WEIGHT};

    deepseekDense::DeepseekDenseMaskSplitParam splitParam;
    CreateDeepseekDenseMaskSplitOperation(splitParam, &expertMaskSplit0Node.operation);
    expertMaskSplit0Node.inTensorIds = {INTERMIDATE_EXPERT_MASK_WITH_WEIGHT};
    expertMaskSplit0Node.outTensorIds = {
        INTERMIDATE_EXPERT_MASK_ZERO_SEVEN,
        INTERMIDATE_EXPERT_MASK_EIGHT_FIFTEEN,
        INTERMIDATE_EXPERT_MASK_SIXTEEN_TWENTY_THREE,
        INTERMIDATE_EXPERT_MASK_TWENTY_FOUR_THIRTY_ONE,
        INTERMIDATE_EXPERT_MASK_THIRTY_TWO_THIRTY_NINE,
        INTERMIDATE_EXPERT_MASK_FOURTY_FOURTY_SEVEN,
        INTERMIDATE_EXPERT_MASK_FOURTY_EIGHT_FIFTY_FIVE,
        INTERMIDATE_EXPERT_MASK_FIFTY_SIX_DIXTY_THREE,
    };
    ATB_LOG(INFO) << "Expert Mask splited 0-7, 8-15, 16-23, 24-31, 32-39, 40-47, 48-55, 56-63 success";

    CreateDeepseekDenseMaskSplitOperation(splitParam, &expertMaskSplit1Node.operation);
    expertMaskSplit1Node.inTensorIds = {INTERMIDATE_EXPERT_MASK_ZERO_SEVEN};
    expertMaskSplit1Node.outTensorIds = {
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

    CreateDeepseekDenseMaskSplitOperation(splitParam, &expertMaskSplit2Node.operation);
    expertMaskSplit2Node.inTensorIds = {INTERMIDATE_EXPERT_MASK_EIGHT_FIFTEEN};
    expertMaskSplit2Node.outTensorIds = {
        INTERMIDATE_EXPERT_MASK_EIGHT,
        INTERMIDATE_EXPERT_MASK_NINE,
        INTERMIDATE_EXPERT_MASK_TEN,
        INTERMIDATE_EXPERT_MASK_ELEVEN,
        INTERMIDATE_EXPERT_MASK_TWELVE,
        INTERMIDATE_EXPERT_MASK_THIRTEEN,
        INTERMIDATE_EXPERT_MASK_FOURTEEN,
        INTERMIDATE_EXPERT_MASK_FIFTEEN,
    };
    ATB_LOG(INFO) << "Expert Mask splited 8-15 success";

    CreateDeepseekDenseMaskSplitOperation(splitParam, &expertMaskSplit3Node.operation);
    expertMaskSplit3Node.inTensorIds = {INTERMIDATE_EXPERT_MASK_SIXTEEN_TWENTY_THREE};
    expertMaskSplit3Node.outTensorIds = {
        INTERMIDATE_EXPERT_MASK_SIXTEEN,
        INTERMIDATE_EXPERT_MASK_SEVENTEEN,
        INTERMIDATE_EXPERT_MASK_EIGHTEEN,
        INTERMIDATE_EXPERT_MASK_NINETEEN,
        INTERMIDATE_EXPERT_MASK_TWENTY,
        INTERMIDATE_EXPERT_MASK_TWENTY_ONE,
        INTERMIDATE_EXPERT_MASK_TWENTY_TWO,
        INTERMIDATE_EXPERT_MASK_TWENTY_THREE,
    };
    ATB_LOG(INFO) << "Expert Mask splited 16-23 success";

    CreateDeepseekDenseMaskSplitOperation(splitParam, &expertMaskSplit4Node.operation);
    expertMaskSplit4Node.inTensorIds = {INTERMIDATE_EXPERT_MASK_TWENTY_FOUR_THIRTY_ONE};
    expertMaskSplit4Node.outTensorIds = {
        INTERMIDATE_EXPERT_MASK_TWENTY_FOUR,
        INTERMIDATE_EXPERT_MASK_TWENTY_FIVE,
        INTERMIDATE_EXPERT_MASK_TWENTY_SIX,
        INTERMIDATE_EXPERT_MASK_TWENTY_SEVEN,
        INTERMIDATE_EXPERT_MASK_TWENTY_EIGHT,
        INTERMIDATE_EXPERT_MASK_TWENTY_NINE,
        INTERMIDATE_EXPERT_MASK_THIRTY,
        INTERMIDATE_EXPERT_MASK_THIRTY_ONE,
    };
    ATB_LOG(INFO) << "Expert Mask splited 24-31 success";

    CreateDeepseekDenseMaskSplitOperation(splitParam, &expertMaskSplit5Node.operation);
    expertMaskSplit5Node.inTensorIds = {INTERMIDATE_EXPERT_MASK_THIRTY_TWO_THIRTY_NINE};
    expertMaskSplit5Node.outTensorIds = {
        INTERMIDATE_EXPERT_MASK_THIRTY_TWO,
        INTERMIDATE_EXPERT_MASK_THIRTY_THREE,
        INTERMIDATE_EXPERT_MASK_THIRTY_FOUR,
        INTERMIDATE_EXPERT_MASK_THIRTY_FIVE,
        INTERMIDATE_EXPERT_MASK_THIRTY_SIX,
        INTERMIDATE_EXPERT_MASK_THIRTY_SEVEN,
        INTERMIDATE_EXPERT_MASK_THIRTY_EIGHT,
        INTERMIDATE_EXPERT_MASK_THIRTY_NINE,
    };
    ATB_LOG(INFO) << "Expert Mask splited 32-39 success";

    CreateDeepseekDenseMaskSplitOperation(splitParam, &expertMaskSplit6Node.operation);
    expertMaskSplit6Node.inTensorIds = {INTERMIDATE_EXPERT_MASK_FOURTY_FOURTY_SEVEN};
    expertMaskSplit6Node.outTensorIds = {
        INTERMIDATE_EXPERT_MASK_FOURTY,
        INTERMIDATE_EXPERT_MASK_FOURTY_ONE,
        INTERMIDATE_EXPERT_MASK_FOURTY_TWO,
        INTERMIDATE_EXPERT_MASK_FOURTY_THREE,
        INTERMIDATE_EXPERT_MASK_FOURTY_FOUR,
        INTERMIDATE_EXPERT_MASK_FOURTY_FIVE,
        INTERMIDATE_EXPERT_MASK_FOURTY_SIX,
        INTERMIDATE_EXPERT_MASK_FOURTY_SEVEN,
    };
    ATB_LOG(INFO) << "Expert Mask splited 40-47 success";

    CreateDeepseekDenseMaskSplitOperation(splitParam, &expertMaskSplit7Node.operation);
    expertMaskSplit7Node.inTensorIds = {INTERMIDATE_EXPERT_MASK_FOURTY_EIGHT_FIFTY_FIVE};
    expertMaskSplit7Node.outTensorIds = {
        INTERMIDATE_EXPERT_MASK_FOURTY_EIGHT,
        INTERMIDATE_EXPERT_MASK_FOURTY_NINE,
        INTERMIDATE_EXPERT_MASK_FIFTY,
        INTERMIDATE_EXPERT_MASK_FIFTY_ONE,
        INTERMIDATE_EXPERT_MASK_FIFTY_TWO,
        INTERMIDATE_EXPERT_MASK_FIFTY_THREE,
        INTERMIDATE_EXPERT_MASK_FIFTY_FOUR,
        INTERMIDATE_EXPERT_MASK_FIFTY_FIVE,
    };
    ATB_LOG(INFO) << "Expert Mask splited 48-55 success";

    CreateDeepseekDenseMaskSplitOperation(splitParam, &expertMaskSplit8Node.operation);
    expertMaskSplit8Node.inTensorIds = {INTERMIDATE_EXPERT_MASK_FIFTY_SIX_DIXTY_THREE};
    expertMaskSplit8Node.outTensorIds = {
        INTERMIDATE_EXPERT_MASK_FIFTY_SIX,
        INTERMIDATE_EXPERT_MASK_FIFTY_SEVEN,
        INTERMIDATE_EXPERT_MASK_FIFTY_EIGHT,
        INTERMIDATE_EXPERT_MASK_FIFTY_NINE,
        INTERMIDATE_EXPERT_MASK_SIXTY,
        INTERMIDATE_EXPERT_MASK_SIXTY_ONE,
        INTERMIDATE_EXPERT_MASK_SIXTY_TWO,
        INTERMIDATE_EXPERT_MASK_SIXTY_THREE,
    };
    ATB_LOG(INFO) << "Expert Mask splited 56-63 success";
    
    for (int expertId = 0; expertId < (param.numOfExperts / param.expertParallelDegree); ++expertId) {
        auto &expertNode = opGraph.nodes.at(nodeId++);
        ATB_LOG(INFO) << "Expert created " << expertId;
        atb_speed::deepseekDense::DeepseekDenseMlpParam mlpExpertParam;
        mlpExpertParam.transpose = param.transpose;
        deepseekDense::CreateDeepseekDenseMlpOperation(mlpExpertParam, &expertNode.operation);
        uint mlpGateUpWeightIdx = IN_MLPGATEUPWEIGHT_EXPERT_ZERO + expertId * 2;
        uint mlpDownWeightIdx = IN_MLPDOWNWEIGHT_EXPERT_ZERO + expertId * 2;
        uint finalHiddenStateIdx = INTERMIDATE_FINAL_HIDDEN_STATE_ZERO + expertId - 1;
        if (expertId == 0) {
            finalHiddenStateIdx = IN_FINAL_HIDDEN_STATE;
        }
        uint expertMaskIdx = INTERMIDATE_EXPERT_MASK_ZERO + expertId + param.maskStartIdx * param.numOfExperts / param.expertParallelDegree;
        uint outTensorIdx = OUT_DEEPSEEK_DENSE_MOE_ROUT;
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