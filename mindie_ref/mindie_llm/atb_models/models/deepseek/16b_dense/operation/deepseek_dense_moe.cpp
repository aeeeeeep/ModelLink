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
    INTERMIDATE_EXPERT_MASK_WITH_WEIGHT_TRANSPOSED,
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
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 143;
static const uint64_t NODE_COUNT = 80;

atb::Status CreateDeepseekDenseMoeOperation(const DeepseekDenseMoeParam &param, atb::Operation **operation)
{
    std::shared_ptr<int64_t> batchDimPtr = std::make_shared<int64_t>(0);

    atb::GraphParam opGraph;
    opGraph.name = "DeepseekDenseMoe";
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    atb::Node &linearNode = opGraph.nodes.at(nodeId++);
    atb::Node &softMaxNode = opGraph.nodes.at(nodeId++);
    atb::Node &topKNode = opGraph.nodes.at(nodeId++);
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
    atb::Node &expertMaskSplit7Node = opGraph.nodes.at(nodeId++);
    atb::Node &expertMaskSplit8Node = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpert0Node = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpert1Node = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpert2Node = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpert3Node = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpert4Node = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpert5Node = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpert6Node = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpert7Node = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpert8Node = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpert9Node = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpert10Node = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpert11Node = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpert12Node = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpert13Node = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpert14Node = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpert15Node = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpert16Node = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpert17Node = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpert18Node = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpert19Node = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpert20Node = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpert21Node = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpert22Node = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpert23Node = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpert24Node = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpert25Node = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpert26Node = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpert27Node = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpert28Node = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpert29Node = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpert30Node = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpert31Node = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpert32Node = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpert33Node = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpert34Node = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpert35Node = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpert36Node = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpert37Node = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpert38Node = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpert39Node = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpert40Node = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpert41Node = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpert42Node = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpert43Node = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpert44Node = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpert45Node = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpert46Node = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpert47Node = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpert48Node = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpert49Node = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpert50Node = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpert51Node = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpert52Node = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpert53Node = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpert54Node = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpert55Node = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpert56Node = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpert57Node = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpert58Node = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpert59Node = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpert60Node = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpert61Node = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpert62Node = opGraph.nodes.at(nodeId++);
    atb::Node &mlpExpert63Node = opGraph.nodes.at(nodeId++);

    // In_tensor[0]: hiddesn_state: Batch; seq_len; hidden_dim, reshaped to Batch * Seq; hidden_dim
    // In_tensor[1]: Gate_weight: num_experts(8); hidden_dim
    // Out_tensor[0]: router_logits: Batch * Seq; num_experts
    atb::infer::MatmulParam moeGateParam = {false, param.transpose};
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

    // In_tensor[0]: router_weights: Batch * Seq; 2
    atb::infer::TransposeParam transposeParam;
    transposeParam.perm = {0, 2, 1};
    CreateOperation(transposeParam, &transposeNode.operation);
    transposeNode.inTensorIds = {INTERMIDATE_EXPERT_MASK_WITH_WEIGHT};
    transposeNode.outTensorIds = {INTERMIDATE_EXPERT_MASK_WITH_WEIGHT_TRANSPOSED};
    ATB_LOG(INFO) << "Router weights TRANSPOSED success";

    deepseekDense::DeepseekDenseMaskSplitParam splitParam;
    CreateDeepseekDenseMaskSplitOperation(splitParam, &expertMaskSplit0Node.operation);
    expertMaskSplit0Node.inTensorIds = {INTERMIDATE_EXPERT_MASK_WITH_WEIGHT_TRANSPOSED};
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

    atb_speed::deepseekDense::DeepseekDenseMlpParam mlpExpertParam;
    mlpExpertParam.transpose = param.transpose;
    deepseekDense::CreateDeepseekDenseMlpOperation(mlpExpertParam, &mlpExpert0Node.operation);
    mlpExpert0Node.inTensorIds = {
        IN_HIDDEN_STATE,
        IN_MLPGATEUPWEIGHT_EXPERT_ZERO,
        IN_MLPDOWNWEIGHT_EXPERT_ZERO,
        INTERMIDATE_EXPERT_MASK_ZERO,
        IN_FINAL_HIDDEN_STATE};
    mlpExpert0Node.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_ZERO};
    ATB_LOG(INFO) << "Expert 0 calculation success";

    deepseekDense::CreateDeepseekDenseMlpOperation(mlpExpertParam, &mlpExpert1Node.operation);
    mlpExpert1Node.inTensorIds = {
        IN_HIDDEN_STATE,
        IN_MLPGATEUPWEIGHT_EXPERT_ONE,
        IN_MLPDOWNWEIGHT_EXPERT_ONE,
        INTERMIDATE_EXPERT_MASK_ONE,
        INTERMIDATE_FINAL_HIDDEN_STATE_ZERO};
    mlpExpert1Node.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_ONE};
    ATB_LOG(INFO) << "Expert 1 calculation success";

    deepseekDense::CreateDeepseekDenseMlpOperation(mlpExpertParam, &mlpExpert2Node.operation);
    mlpExpert2Node.inTensorIds = {
        IN_HIDDEN_STATE,
        IN_MLPGATEUPWEIGHT_EXPERT_TWO,
        IN_MLPDOWNWEIGHT_EXPERT_TWO,
        INTERMIDATE_EXPERT_MASK_TWO,
        INTERMIDATE_FINAL_HIDDEN_STATE_ONE};
    mlpExpert2Node.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_TWO};
    ATB_LOG(INFO) << "Expert 2 calculation success";

    deepseekDense::CreateDeepseekDenseMlpOperation(mlpExpertParam, &mlpExpert3Node.operation);
    mlpExpert3Node.inTensorIds = {
        IN_HIDDEN_STATE,
        IN_MLPGATEUPWEIGHT_EXPERT_THREE,
        IN_MLPDOWNWEIGHT_EXPERT_THREE,
        INTERMIDATE_EXPERT_MASK_THREE,
        INTERMIDATE_FINAL_HIDDEN_STATE_TWO};
    mlpExpert3Node.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_THREE};
    ATB_LOG(INFO) << "Expert 3 calculation success";

    deepseekDense::CreateDeepseekDenseMlpOperation(mlpExpertParam, &mlpExpert4Node.operation);
    mlpExpert4Node.inTensorIds = {
        IN_HIDDEN_STATE,
        IN_MLPGATEUPWEIGHT_EXPERT_FOUR,
        IN_MLPDOWNWEIGHT_EXPERT_FOUR,
        INTERMIDATE_EXPERT_MASK_FOUR,
        INTERMIDATE_FINAL_HIDDEN_STATE_THREE};
    mlpExpert4Node.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_FOUR};
    ATB_LOG(INFO) << "Expert 4 calculation success";

    deepseekDense::CreateDeepseekDenseMlpOperation(mlpExpertParam, &mlpExpert5Node.operation);
    mlpExpert5Node.inTensorIds = {
        IN_HIDDEN_STATE,
        IN_MLPGATEUPWEIGHT_EXPERT_FIVE,
        IN_MLPDOWNWEIGHT_EXPERT_FIVE,
        INTERMIDATE_EXPERT_MASK_FIVE,
        INTERMIDATE_FINAL_HIDDEN_STATE_FOUR};
    mlpExpert5Node.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_FIVE};
    ATB_LOG(INFO) << "Expert 5 calculation success";

    deepseekDense::CreateDeepseekDenseMlpOperation(mlpExpertParam, &mlpExpert6Node.operation);
    mlpExpert6Node.inTensorIds = {
        IN_HIDDEN_STATE,
        IN_MLPGATEUPWEIGHT_EXPERT_SIX,
        IN_MLPDOWNWEIGHT_EXPERT_SIX,
        INTERMIDATE_EXPERT_MASK_SIX,
        INTERMIDATE_FINAL_HIDDEN_STATE_FIVE};
    mlpExpert6Node.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_SIX};
    ATB_LOG(INFO) << "Expert 6 calculation success";

    deepseekDense::CreateDeepseekDenseMlpOperation(mlpExpertParam, &mlpExpert7Node.operation);
    mlpExpert7Node.inTensorIds = {
        IN_HIDDEN_STATE,
        IN_MLPGATEUPWEIGHT_EXPERT_SEVEN,
        IN_MLPDOWNWEIGHT_EXPERT_SEVEN,
        INTERMIDATE_EXPERT_MASK_SEVEN,
        INTERMIDATE_FINAL_HIDDEN_STATE_SIX};
    mlpExpert7Node.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_SEVEN};
    ATB_LOG(INFO) << "Expert 7 calculation success";

    deepseekDense::CreateDeepseekDenseMlpOperation(mlpExpertParam, &mlpExpert8Node.operation);
    mlpExpert8Node.inTensorIds = {
        IN_HIDDEN_STATE,
        IN_MLPGATEUPWEIGHT_EXPERT_EIGHT,
        IN_MLPDOWNWEIGHT_EXPERT_EIGHT,
        INTERMIDATE_EXPERT_MASK_EIGHT,
        INTERMIDATE_FINAL_HIDDEN_STATE_SEVEN};
    mlpExpert8Node.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_EIGHT};
    ATB_LOG(INFO) << "Expert 8 calculation success";

    deepseekDense::CreateDeepseekDenseMlpOperation(mlpExpertParam, &mlpExpert9Node.operation);
    mlpExpert9Node.inTensorIds = {
        IN_HIDDEN_STATE,
        IN_MLPGATEUPWEIGHT_EXPERT_NINE,
        IN_MLPDOWNWEIGHT_EXPERT_NINE,
        INTERMIDATE_EXPERT_MASK_NINE,
        INTERMIDATE_FINAL_HIDDEN_STATE_EIGHT};
    mlpExpert9Node.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_NINE};
    ATB_LOG(INFO) << "Expert 9 calculation success";

    deepseekDense::CreateDeepseekDenseMlpOperation(mlpExpertParam, &mlpExpert10Node.operation);
    mlpExpert10Node.inTensorIds = {
        IN_HIDDEN_STATE,
        IN_MLPGATEUPWEIGHT_EXPERT_TEN,
        IN_MLPDOWNWEIGHT_EXPERT_TEN,
        INTERMIDATE_EXPERT_MASK_TEN,
        INTERMIDATE_FINAL_HIDDEN_STATE_NINE};
    mlpExpert10Node.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_TEN};
    ATB_LOG(INFO) << "Expert 10 calculation success";

    deepseekDense::CreateDeepseekDenseMlpOperation(mlpExpertParam, &mlpExpert11Node.operation);
    mlpExpert11Node.inTensorIds = {
        IN_HIDDEN_STATE,
        IN_MLPGATEUPWEIGHT_EXPERT_ELEVEN,
        IN_MLPDOWNWEIGHT_EXPERT_ELEVEN,
        INTERMIDATE_EXPERT_MASK_ELEVEN,
        INTERMIDATE_FINAL_HIDDEN_STATE_TEN};
    mlpExpert11Node.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_ELEVEN};
    ATB_LOG(INFO) << "Expert 11 calculation success";

    deepseekDense::CreateDeepseekDenseMlpOperation(mlpExpertParam, &mlpExpert12Node.operation);
    mlpExpert12Node.inTensorIds = {
        IN_HIDDEN_STATE,
        IN_MLPGATEUPWEIGHT_EXPERT_TWELVE,
        IN_MLPDOWNWEIGHT_EXPERT_TWELVE,
        INTERMIDATE_EXPERT_MASK_TWELVE,
        INTERMIDATE_FINAL_HIDDEN_STATE_ELEVEN};
    mlpExpert12Node.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_TWELVE};
    ATB_LOG(INFO) << "Expert 12 calculation success";

    deepseekDense::CreateDeepseekDenseMlpOperation(mlpExpertParam, &mlpExpert13Node.operation);
    mlpExpert13Node.inTensorIds = {
        IN_HIDDEN_STATE,
        IN_MLPGATEUPWEIGHT_EXPERT_THIRTEEN,
        IN_MLPDOWNWEIGHT_EXPERT_THIRTEEN,
        INTERMIDATE_EXPERT_MASK_THIRTEEN,
        INTERMIDATE_FINAL_HIDDEN_STATE_TWELVE};
    mlpExpert13Node.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_THIRTEEN};
    ATB_LOG(INFO) << "Expert 13 calculation success";

    deepseekDense::CreateDeepseekDenseMlpOperation(mlpExpertParam, &mlpExpert14Node.operation);
    mlpExpert14Node.inTensorIds = {
        IN_HIDDEN_STATE,
        IN_MLPGATEUPWEIGHT_EXPERT_FOURTEEN,
        IN_MLPDOWNWEIGHT_EXPERT_FOURTEEN,
        INTERMIDATE_EXPERT_MASK_FOURTEEN,
        INTERMIDATE_FINAL_HIDDEN_STATE_THIRTEEN};
    mlpExpert14Node.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_FOURTEEN};
    ATB_LOG(INFO) << "Expert 14 calculation success";

    deepseekDense::CreateDeepseekDenseMlpOperation(mlpExpertParam, &mlpExpert15Node.operation);
    mlpExpert15Node.inTensorIds = {
        IN_HIDDEN_STATE,
        IN_MLPGATEUPWEIGHT_EXPERT_FIFTEEN,
        IN_MLPDOWNWEIGHT_EXPERT_FIFTEEN,
        INTERMIDATE_EXPERT_MASK_FIFTEEN,
        INTERMIDATE_FINAL_HIDDEN_STATE_FOURTEEN};
    mlpExpert15Node.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_FIFTEEN};
    ATB_LOG(INFO) << "Expert seven calculation success";

    deepseekDense::CreateDeepseekDenseMlpOperation(mlpExpertParam, &mlpExpert16Node.operation);
    mlpExpert16Node.inTensorIds = {
        IN_HIDDEN_STATE,
        IN_MLPGATEUPWEIGHT_EXPERT_SIXTEEN,
        IN_MLPDOWNWEIGHT_EXPERT_SIXTEEN,
        INTERMIDATE_EXPERT_MASK_SIXTEEN,
        INTERMIDATE_FINAL_HIDDEN_STATE_FIFTEEN};
    mlpExpert16Node.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_SIXTEEN};
    ATB_LOG(INFO) << "Expert 16 calculation success";

    deepseekDense::CreateDeepseekDenseMlpOperation(mlpExpertParam, &mlpExpert17Node.operation);
    mlpExpert17Node.inTensorIds = {
        IN_HIDDEN_STATE,
        IN_MLPGATEUPWEIGHT_EXPERT_SEVENTEEN,
        IN_MLPDOWNWEIGHT_EXPERT_SEVENTEEN,
        INTERMIDATE_EXPERT_MASK_SEVENTEEN,
        INTERMIDATE_FINAL_HIDDEN_STATE_SIXTEEN};
    mlpExpert17Node.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_SEVENTEEN};
    ATB_LOG(INFO) << "Expert 17 calculation success";

    deepseekDense::CreateDeepseekDenseMlpOperation(mlpExpertParam, &mlpExpert18Node.operation);
    mlpExpert18Node.inTensorIds = {
        IN_HIDDEN_STATE,
        IN_MLPGATEUPWEIGHT_EXPERT_EIGHTEEN,
        IN_MLPDOWNWEIGHT_EXPERT_EIGHTEEN,
        INTERMIDATE_EXPERT_MASK_EIGHTEEN,
        INTERMIDATE_FINAL_HIDDEN_STATE_SEVENTEEN};
    mlpExpert18Node.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_EIGHTEEN};
    ATB_LOG(INFO) << "Expert 18 calculation success";

    deepseekDense::CreateDeepseekDenseMlpOperation(mlpExpertParam, &mlpExpert19Node.operation);
    mlpExpert19Node.inTensorIds = {
        IN_HIDDEN_STATE,
        IN_MLPGATEUPWEIGHT_EXPERT_NINETEEN,
        IN_MLPDOWNWEIGHT_EXPERT_NINETEEN,
        INTERMIDATE_EXPERT_MASK_NINETEEN,
        INTERMIDATE_FINAL_HIDDEN_STATE_EIGHTEEN};
    mlpExpert19Node.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_NINETEEN};
    ATB_LOG(INFO) << "Expert 19 calculation success";

    deepseekDense::CreateDeepseekDenseMlpOperation(mlpExpertParam, &mlpExpert20Node.operation);
    mlpExpert20Node.inTensorIds = {
        IN_HIDDEN_STATE,
        IN_MLPGATEUPWEIGHT_EXPERT_TWENTY,
        IN_MLPDOWNWEIGHT_EXPERT_TWENTY,
        INTERMIDATE_EXPERT_MASK_TWENTY,
        INTERMIDATE_FINAL_HIDDEN_STATE_NINETEEN};
    mlpExpert20Node.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_TWENTY};
    ATB_LOG(INFO) << "Expert 20 calculation success";

    deepseekDense::CreateDeepseekDenseMlpOperation(mlpExpertParam, &mlpExpert21Node.operation);
    mlpExpert21Node.inTensorIds = {
        IN_HIDDEN_STATE,
        IN_MLPGATEUPWEIGHT_EXPERT_TWENTY_ONE,
        IN_MLPDOWNWEIGHT_EXPERT_TWENTY_ONE,
        INTERMIDATE_EXPERT_MASK_TWENTY_ONE,
        INTERMIDATE_FINAL_HIDDEN_STATE_TWENTY};
    mlpExpert21Node.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_TWENTY_ONE};
    ATB_LOG(INFO) << "Expert 21 calculation success";

    deepseekDense::CreateDeepseekDenseMlpOperation(mlpExpertParam, &mlpExpert22Node.operation);
    mlpExpert22Node.inTensorIds = {
        IN_HIDDEN_STATE,
        IN_MLPGATEUPWEIGHT_EXPERT_TWENTY_TWO,
        IN_MLPDOWNWEIGHT_EXPERT_TWENTY_TWO,
        INTERMIDATE_EXPERT_MASK_TWENTY_TWO,
        INTERMIDATE_FINAL_HIDDEN_STATE_TWENTY_ONE};
    mlpExpert22Node.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_TWENTY_TWO};
    ATB_LOG(INFO) << "Expert 22 calculation success";

    deepseekDense::CreateDeepseekDenseMlpOperation(mlpExpertParam, &mlpExpert23Node.operation);
    mlpExpert23Node.inTensorIds = {
        IN_HIDDEN_STATE,
        IN_MLPGATEUPWEIGHT_EXPERT_TWENTY_THREE,
        IN_MLPDOWNWEIGHT_EXPERT_TWENTY_THREE,
        INTERMIDATE_EXPERT_MASK_TWENTY_THREE,
        INTERMIDATE_FINAL_HIDDEN_STATE_TWENTY_TWO};
    mlpExpert23Node.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_TWENTY_THREE};
    ATB_LOG(INFO) << "Expert seven calculation success";

    deepseekDense::CreateDeepseekDenseMlpOperation(mlpExpertParam, &mlpExpert24Node.operation);
    mlpExpert24Node.inTensorIds = {
        IN_HIDDEN_STATE,
        IN_MLPGATEUPWEIGHT_EXPERT_TWENTY_FOUR,
        IN_MLPDOWNWEIGHT_EXPERT_TWENTY_FOUR,
        INTERMIDATE_EXPERT_MASK_TWENTY_FOUR,
        INTERMIDATE_FINAL_HIDDEN_STATE_TWENTY_THREE};
    mlpExpert24Node.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_TWENTY_FOUR};
    ATB_LOG(INFO) << "Expert 24 calculation success";

    deepseekDense::CreateDeepseekDenseMlpOperation(mlpExpertParam, &mlpExpert25Node.operation);
    mlpExpert25Node.inTensorIds = {
        IN_HIDDEN_STATE,
        IN_MLPGATEUPWEIGHT_EXPERT_TWENTY_FIVE,
        IN_MLPDOWNWEIGHT_EXPERT_TWENTY_FIVE,
        INTERMIDATE_EXPERT_MASK_TWENTY_FIVE,
        INTERMIDATE_FINAL_HIDDEN_STATE_TWENTY_FOUR};
    mlpExpert25Node.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_TWENTY_FIVE};
    ATB_LOG(INFO) << "Expert 25 calculation success";

    deepseekDense::CreateDeepseekDenseMlpOperation(mlpExpertParam, &mlpExpert26Node.operation);
    mlpExpert26Node.inTensorIds = {
        IN_HIDDEN_STATE,
        IN_MLPGATEUPWEIGHT_EXPERT_TWENTY_SIX,
        IN_MLPDOWNWEIGHT_EXPERT_TWENTY_SIX,
        INTERMIDATE_EXPERT_MASK_TWENTY_SIX,
        INTERMIDATE_FINAL_HIDDEN_STATE_TWENTY_FIVE};
    mlpExpert26Node.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_TWENTY_SIX};
    ATB_LOG(INFO) << "Expert 26 calculation success";

    deepseekDense::CreateDeepseekDenseMlpOperation(mlpExpertParam, &mlpExpert27Node.operation);
    mlpExpert27Node.inTensorIds = {
        IN_HIDDEN_STATE,
        IN_MLPGATEUPWEIGHT_EXPERT_TWENTY_SEVEN,
        IN_MLPDOWNWEIGHT_EXPERT_TWENTY_SEVEN,
        INTERMIDATE_EXPERT_MASK_TWENTY_SEVEN,
        INTERMIDATE_FINAL_HIDDEN_STATE_TWENTY_SIX};
    mlpExpert27Node.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_TWENTY_SEVEN};
    ATB_LOG(INFO) << "Expert 27 calculation success";

    deepseekDense::CreateDeepseekDenseMlpOperation(mlpExpertParam, &mlpExpert28Node.operation);
    mlpExpert28Node.inTensorIds = {
        IN_HIDDEN_STATE,
        IN_MLPGATEUPWEIGHT_EXPERT_TWENTY_EIGHT,
        IN_MLPDOWNWEIGHT_EXPERT_TWENTY_EIGHT,
        INTERMIDATE_EXPERT_MASK_TWENTY_EIGHT,
        INTERMIDATE_FINAL_HIDDEN_STATE_TWENTY_SEVEN};
    mlpExpert28Node.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_TWENTY_EIGHT};
    ATB_LOG(INFO) << "Expert 28 calculation success";

    deepseekDense::CreateDeepseekDenseMlpOperation(mlpExpertParam, &mlpExpert29Node.operation);
    mlpExpert29Node.inTensorIds = {
        IN_HIDDEN_STATE,
        IN_MLPGATEUPWEIGHT_EXPERT_TWENTY_NINE,
        IN_MLPDOWNWEIGHT_EXPERT_TWENTY_NINE,
        INTERMIDATE_EXPERT_MASK_TWENTY_NINE,
        INTERMIDATE_FINAL_HIDDEN_STATE_TWENTY_EIGHT};
    mlpExpert29Node.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_TWENTY_NINE};
    ATB_LOG(INFO) << "Expert 29 calculation success";

    deepseekDense::CreateDeepseekDenseMlpOperation(mlpExpertParam, &mlpExpert30Node.operation);
    mlpExpert30Node.inTensorIds = {
        IN_HIDDEN_STATE,
        IN_MLPGATEUPWEIGHT_EXPERT_THIRTY,
        IN_MLPDOWNWEIGHT_EXPERT_THIRTY,
        INTERMIDATE_EXPERT_MASK_THIRTY,
        INTERMIDATE_FINAL_HIDDEN_STATE_TWENTY_NINE};
    mlpExpert30Node.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_THIRTY};
    ATB_LOG(INFO) << "Expert 30 calculation success";

    deepseekDense::CreateDeepseekDenseMlpOperation(mlpExpertParam, &mlpExpert31Node.operation);
    mlpExpert31Node.inTensorIds = {
        IN_HIDDEN_STATE,
        IN_MLPGATEUPWEIGHT_EXPERT_THIRTY_ONE,
        IN_MLPDOWNWEIGHT_EXPERT_THIRTY_ONE,
        INTERMIDATE_EXPERT_MASK_THIRTY_ONE,
        INTERMIDATE_FINAL_HIDDEN_STATE_THIRTY};
    mlpExpert31Node.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_THIRTY_ONE};
    ATB_LOG(INFO) << "Expert 31 calculation success";

    deepseekDense::CreateDeepseekDenseMlpOperation(mlpExpertParam, &mlpExpert32Node.operation);
    mlpExpert32Node.inTensorIds = {
        IN_HIDDEN_STATE,
        IN_MLPGATEUPWEIGHT_EXPERT_THIRTY_TWO,
        IN_MLPDOWNWEIGHT_EXPERT_THIRTY_TWO,
        INTERMIDATE_EXPERT_MASK_THIRTY_TWO,
        INTERMIDATE_FINAL_HIDDEN_STATE_THIRTY_ONE};
    mlpExpert32Node.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_THIRTY_TWO};
    ATB_LOG(INFO) << "Expert 32 calculation success";

    deepseekDense::CreateDeepseekDenseMlpOperation(mlpExpertParam, &mlpExpert33Node.operation);
    mlpExpert33Node.inTensorIds = {
        IN_HIDDEN_STATE,
        IN_MLPGATEUPWEIGHT_EXPERT_THIRTY_THREE,
        IN_MLPDOWNWEIGHT_EXPERT_THIRTY_THREE,
        INTERMIDATE_EXPERT_MASK_THIRTY_THREE,
        INTERMIDATE_FINAL_HIDDEN_STATE_THIRTY_TWO};
    mlpExpert33Node.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_THIRTY_THREE};
    ATB_LOG(INFO) << "Expert 33 calculation success";

    deepseekDense::CreateDeepseekDenseMlpOperation(mlpExpertParam, &mlpExpert34Node.operation);
    mlpExpert34Node.inTensorIds = {
        IN_HIDDEN_STATE,
        IN_MLPGATEUPWEIGHT_EXPERT_THIRTY_FOUR,
        IN_MLPDOWNWEIGHT_EXPERT_THIRTY_FOUR,
        INTERMIDATE_EXPERT_MASK_THIRTY_FOUR,
        INTERMIDATE_FINAL_HIDDEN_STATE_THIRTY_THREE};
    mlpExpert34Node.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_THIRTY_FOUR};
    ATB_LOG(INFO) << "Expert 34 calculation success";

    deepseekDense::CreateDeepseekDenseMlpOperation(mlpExpertParam, &mlpExpert35Node.operation);
    mlpExpert35Node.inTensorIds = {
        IN_HIDDEN_STATE,
        IN_MLPGATEUPWEIGHT_EXPERT_THIRTY_FIVE,
        IN_MLPDOWNWEIGHT_EXPERT_THIRTY_FIVE,
        INTERMIDATE_EXPERT_MASK_THIRTY_FIVE,
        INTERMIDATE_FINAL_HIDDEN_STATE_THIRTY_FOUR};
    mlpExpert35Node.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_THIRTY_FIVE};
    ATB_LOG(INFO) << "Expert 35 calculation success";

    deepseekDense::CreateDeepseekDenseMlpOperation(mlpExpertParam, &mlpExpert36Node.operation);
    mlpExpert36Node.inTensorIds = {
        IN_HIDDEN_STATE,
        IN_MLPGATEUPWEIGHT_EXPERT_THIRTY_SIX,
        IN_MLPDOWNWEIGHT_EXPERT_THIRTY_SIX,
        INTERMIDATE_EXPERT_MASK_THIRTY_SIX,
        INTERMIDATE_FINAL_HIDDEN_STATE_THIRTY_FIVE};
    mlpExpert36Node.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_THIRTY_SIX};
    ATB_LOG(INFO) << "Expert 36 calculation success";

    deepseekDense::CreateDeepseekDenseMlpOperation(mlpExpertParam, &mlpExpert37Node.operation);
    mlpExpert37Node.inTensorIds = {
        IN_HIDDEN_STATE,
        IN_MLPGATEUPWEIGHT_EXPERT_THIRTY_SEVEN,
        IN_MLPDOWNWEIGHT_EXPERT_THIRTY_SEVEN,
        INTERMIDATE_EXPERT_MASK_THIRTY_SEVEN,
        INTERMIDATE_FINAL_HIDDEN_STATE_THIRTY_SIX};
    mlpExpert37Node.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_THIRTY_SEVEN};
    ATB_LOG(INFO) << "Expert 37 calculation success";

    deepseekDense::CreateDeepseekDenseMlpOperation(mlpExpertParam, &mlpExpert38Node.operation);
    mlpExpert38Node.inTensorIds = {
        IN_HIDDEN_STATE,
        IN_MLPGATEUPWEIGHT_EXPERT_THIRTY_EIGHT,
        IN_MLPDOWNWEIGHT_EXPERT_THIRTY_EIGHT,
        INTERMIDATE_EXPERT_MASK_THIRTY_EIGHT,
        INTERMIDATE_FINAL_HIDDEN_STATE_THIRTY_SEVEN};
    mlpExpert38Node.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_THIRTY_EIGHT};
    ATB_LOG(INFO) << "Expert 38 calculation success";

    deepseekDense::CreateDeepseekDenseMlpOperation(mlpExpertParam, &mlpExpert39Node.operation);
    mlpExpert39Node.inTensorIds = {
        IN_HIDDEN_STATE,
        IN_MLPGATEUPWEIGHT_EXPERT_THIRTY_NINE,
        IN_MLPDOWNWEIGHT_EXPERT_THIRTY_NINE,
        INTERMIDATE_EXPERT_MASK_THIRTY_NINE,
        INTERMIDATE_FINAL_HIDDEN_STATE_THIRTY_EIGHT};
    mlpExpert39Node.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_THIRTY_NINE};
    ATB_LOG(INFO) << "Expert 39 calculation success";

    deepseekDense::CreateDeepseekDenseMlpOperation(mlpExpertParam, &mlpExpert40Node.operation);
    mlpExpert40Node.inTensorIds = {
        IN_HIDDEN_STATE,
        IN_MLPGATEUPWEIGHT_EXPERT_FOURTY,
        IN_MLPDOWNWEIGHT_EXPERT_FOURTY,
        INTERMIDATE_EXPERT_MASK_FOURTY,
        INTERMIDATE_FINAL_HIDDEN_STATE_THIRTY_NINE};
    mlpExpert40Node.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_FOURTY};
    ATB_LOG(INFO) << "Expert 40 calculation success";

    deepseekDense::CreateDeepseekDenseMlpOperation(mlpExpertParam, &mlpExpert41Node.operation);
    mlpExpert41Node.inTensorIds = {
        IN_HIDDEN_STATE,
        IN_MLPGATEUPWEIGHT_EXPERT_FOURTY_ONE,
        IN_MLPDOWNWEIGHT_EXPERT_FOURTY_ONE,
        INTERMIDATE_EXPERT_MASK_FOURTY_ONE,
        INTERMIDATE_FINAL_HIDDEN_STATE_FOURTY};
    mlpExpert41Node.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_FOURTY_ONE};
    ATB_LOG(INFO) << "Expert 41 calculation success";

    deepseekDense::CreateDeepseekDenseMlpOperation(mlpExpertParam, &mlpExpert42Node.operation);
    mlpExpert42Node.inTensorIds = {
        IN_HIDDEN_STATE,
        IN_MLPGATEUPWEIGHT_EXPERT_FOURTY_TWO,
        IN_MLPDOWNWEIGHT_EXPERT_FOURTY_TWO,
        INTERMIDATE_EXPERT_MASK_FOURTY_TWO,
        INTERMIDATE_FINAL_HIDDEN_STATE_FOURTY_ONE};
    mlpExpert42Node.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_FOURTY_TWO};
    ATB_LOG(INFO) << "Expert 42 calculation success";

    deepseekDense::CreateDeepseekDenseMlpOperation(mlpExpertParam, &mlpExpert43Node.operation);
    mlpExpert43Node.inTensorIds = {
        IN_HIDDEN_STATE,
        IN_MLPGATEUPWEIGHT_EXPERT_FOURTY_THREE,
        IN_MLPDOWNWEIGHT_EXPERT_FOURTY_THREE,
        INTERMIDATE_EXPERT_MASK_FOURTY_THREE,
        INTERMIDATE_FINAL_HIDDEN_STATE_FOURTY_TWO};
    mlpExpert43Node.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_FOURTY_THREE};
    ATB_LOG(INFO) << "Expert 43 calculation success";

    deepseekDense::CreateDeepseekDenseMlpOperation(mlpExpertParam, &mlpExpert44Node.operation);
    mlpExpert44Node.inTensorIds = {
        IN_HIDDEN_STATE,
        IN_MLPGATEUPWEIGHT_EXPERT_FOURTY_FOUR,
        IN_MLPDOWNWEIGHT_EXPERT_FOURTY_FOUR,
        INTERMIDATE_EXPERT_MASK_FOURTY_FOUR,
        INTERMIDATE_FINAL_HIDDEN_STATE_FOURTY_THREE};
    mlpExpert44Node.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_FOURTY_FOUR};
    ATB_LOG(INFO) << "Expert 44 calculation success";

    deepseekDense::CreateDeepseekDenseMlpOperation(mlpExpertParam, &mlpExpert45Node.operation);
    mlpExpert45Node.inTensorIds = {
        IN_HIDDEN_STATE,
        IN_MLPGATEUPWEIGHT_EXPERT_FOURTY_FIVE,
        IN_MLPDOWNWEIGHT_EXPERT_FOURTY_FIVE,
        INTERMIDATE_EXPERT_MASK_FOURTY_FIVE,
        INTERMIDATE_FINAL_HIDDEN_STATE_FOURTY_FOUR};
    mlpExpert45Node.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_FOURTY_FIVE};
    ATB_LOG(INFO) << "Expert 45 calculation success";

    deepseekDense::CreateDeepseekDenseMlpOperation(mlpExpertParam, &mlpExpert46Node.operation);
    mlpExpert46Node.inTensorIds = {
        IN_HIDDEN_STATE,
        IN_MLPGATEUPWEIGHT_EXPERT_FOURTY_SIX,
        IN_MLPDOWNWEIGHT_EXPERT_FOURTY_SIX,
        INTERMIDATE_EXPERT_MASK_FOURTY_SIX,
        INTERMIDATE_FINAL_HIDDEN_STATE_FOURTY_FIVE};
    mlpExpert46Node.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_FOURTY_SIX};
    ATB_LOG(INFO) << "Expert 46 calculation success";

    deepseekDense::CreateDeepseekDenseMlpOperation(mlpExpertParam, &mlpExpert47Node.operation);
    mlpExpert47Node.inTensorIds = {
        IN_HIDDEN_STATE,
        IN_MLPGATEUPWEIGHT_EXPERT_FOURTY_SEVEN,
        IN_MLPDOWNWEIGHT_EXPERT_FOURTY_SEVEN,
        INTERMIDATE_EXPERT_MASK_FOURTY_SEVEN,
        INTERMIDATE_FINAL_HIDDEN_STATE_FOURTY_SIX};
    mlpExpert47Node.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_FOURTY_SEVEN};
    ATB_LOG(INFO) << "Expert 47 calculation success";

    deepseekDense::CreateDeepseekDenseMlpOperation(mlpExpertParam, &mlpExpert48Node.operation);
    mlpExpert48Node.inTensorIds = {
        IN_HIDDEN_STATE,
        IN_MLPGATEUPWEIGHT_EXPERT_FOURTY_EIGHT,
        IN_MLPDOWNWEIGHT_EXPERT_FOURTY_EIGHT,
        INTERMIDATE_EXPERT_MASK_FOURTY_EIGHT,
        INTERMIDATE_FINAL_HIDDEN_STATE_FOURTY_SEVEN};
    mlpExpert48Node.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_FOURTY_EIGHT};
    ATB_LOG(INFO) << "Expert 48 calculation success";

    deepseekDense::CreateDeepseekDenseMlpOperation(mlpExpertParam, &mlpExpert49Node.operation);
    mlpExpert49Node.inTensorIds = {
        IN_HIDDEN_STATE,
        IN_MLPGATEUPWEIGHT_EXPERT_FOURTY_NINE,
        IN_MLPDOWNWEIGHT_EXPERT_FOURTY_NINE,
        INTERMIDATE_EXPERT_MASK_FOURTY_NINE,
        INTERMIDATE_FINAL_HIDDEN_STATE_FOURTY_EIGHT};
    mlpExpert49Node.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_FOURTY_NINE};
    ATB_LOG(INFO) << "Expert 49 calculation success";

    deepseekDense::CreateDeepseekDenseMlpOperation(mlpExpertParam, &mlpExpert50Node.operation);
    mlpExpert50Node.inTensorIds = {
        IN_HIDDEN_STATE,
        IN_MLPGATEUPWEIGHT_EXPERT_FIFTY,
        IN_MLPDOWNWEIGHT_EXPERT_FIFTY,
        INTERMIDATE_EXPERT_MASK_FIFTY,
        INTERMIDATE_FINAL_HIDDEN_STATE_FOURTY_NINE};
    mlpExpert50Node.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_FIFTY};
    ATB_LOG(INFO) << "Expert 50 calculation success";

    deepseekDense::CreateDeepseekDenseMlpOperation(mlpExpertParam, &mlpExpert51Node.operation);
    mlpExpert51Node.inTensorIds = {
        IN_HIDDEN_STATE,
        IN_MLPGATEUPWEIGHT_EXPERT_FIFTY_ONE,
        IN_MLPDOWNWEIGHT_EXPERT_FIFTY_ONE,
        INTERMIDATE_EXPERT_MASK_FIFTY_ONE,
        INTERMIDATE_FINAL_HIDDEN_STATE_FIFTY};
    mlpExpert51Node.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_FIFTY_ONE};
    ATB_LOG(INFO) << "Expert 51 calculation success";

    deepseekDense::CreateDeepseekDenseMlpOperation(mlpExpertParam, &mlpExpert52Node.operation);
    mlpExpert52Node.inTensorIds = {
        IN_HIDDEN_STATE,
        IN_MLPGATEUPWEIGHT_EXPERT_FIFTY_TWO,
        IN_MLPDOWNWEIGHT_EXPERT_FIFTY_TWO,
        INTERMIDATE_EXPERT_MASK_FIFTY_TWO,
        INTERMIDATE_FINAL_HIDDEN_STATE_FIFTY_ONE};
    mlpExpert52Node.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_FIFTY_TWO};
    ATB_LOG(INFO) << "Expert 52 calculation success";

    deepseekDense::CreateDeepseekDenseMlpOperation(mlpExpertParam, &mlpExpert53Node.operation);
    mlpExpert53Node.inTensorIds = {
        IN_HIDDEN_STATE,
        IN_MLPGATEUPWEIGHT_EXPERT_FIFTY_THREE,
        IN_MLPDOWNWEIGHT_EXPERT_FIFTY_THREE,
        INTERMIDATE_EXPERT_MASK_FIFTY_THREE,
        INTERMIDATE_FINAL_HIDDEN_STATE_FIFTY_TWO};
    mlpExpert53Node.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_FIFTY_THREE};
    ATB_LOG(INFO) << "Expert 53 calculation success";

    deepseekDense::CreateDeepseekDenseMlpOperation(mlpExpertParam, &mlpExpert54Node.operation);
    mlpExpert54Node.inTensorIds = {
        IN_HIDDEN_STATE,
        IN_MLPGATEUPWEIGHT_EXPERT_FIFTY_FOUR,
        IN_MLPDOWNWEIGHT_EXPERT_FIFTY_FOUR,
        INTERMIDATE_EXPERT_MASK_FIFTY_FOUR,
        INTERMIDATE_FINAL_HIDDEN_STATE_FIFTY_THREE};
    mlpExpert54Node.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_FIFTY_FOUR};
    ATB_LOG(INFO) << "Expert 54 calculation success";

    deepseekDense::CreateDeepseekDenseMlpOperation(mlpExpertParam, &mlpExpert55Node.operation);
    mlpExpert55Node.inTensorIds = {
        IN_HIDDEN_STATE,
        IN_MLPGATEUPWEIGHT_EXPERT_FIFTY_FIVE,
        IN_MLPDOWNWEIGHT_EXPERT_FIFTY_FIVE,
        INTERMIDATE_EXPERT_MASK_FIFTY_FIVE,
        INTERMIDATE_FINAL_HIDDEN_STATE_FIFTY_FOUR};
    mlpExpert55Node.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_FIFTY_FIVE};
    ATB_LOG(INFO) << "Expert 55 calculation success";

    deepseekDense::CreateDeepseekDenseMlpOperation(mlpExpertParam, &mlpExpert56Node.operation);
    mlpExpert56Node.inTensorIds = {
        IN_HIDDEN_STATE,
        IN_MLPGATEUPWEIGHT_EXPERT_FIFTY_SIX,
        IN_MLPDOWNWEIGHT_EXPERT_FIFTY_SIX,
        INTERMIDATE_EXPERT_MASK_FIFTY_SIX,
        INTERMIDATE_FINAL_HIDDEN_STATE_FIFTY_FIVE};
    mlpExpert56Node.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_FIFTY_SIX};
    ATB_LOG(INFO) << "Expert 56 calculation success";

    deepseekDense::CreateDeepseekDenseMlpOperation(mlpExpertParam, &mlpExpert57Node.operation);
    mlpExpert57Node.inTensorIds = {
        IN_HIDDEN_STATE,
        IN_MLPGATEUPWEIGHT_EXPERT_FIFTY_SEVEN,
        IN_MLPDOWNWEIGHT_EXPERT_FIFTY_SEVEN,
        INTERMIDATE_EXPERT_MASK_FIFTY_SEVEN,
        INTERMIDATE_FINAL_HIDDEN_STATE_FIFTY_SIX};
    mlpExpert57Node.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_FIFTY_SEVEN};
    ATB_LOG(INFO) << "Expert 57 calculation success";

    deepseekDense::CreateDeepseekDenseMlpOperation(mlpExpertParam, &mlpExpert58Node.operation);
    mlpExpert58Node.inTensorIds = {
        IN_HIDDEN_STATE,
        IN_MLPGATEUPWEIGHT_EXPERT_FIFTY_EIGHT,
        IN_MLPDOWNWEIGHT_EXPERT_FIFTY_EIGHT,
        INTERMIDATE_EXPERT_MASK_FIFTY_EIGHT,
        INTERMIDATE_FINAL_HIDDEN_STATE_FIFTY_SEVEN};
    mlpExpert58Node.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_FIFTY_EIGHT};
    ATB_LOG(INFO) << "Expert 58 calculation success";

    deepseekDense::CreateDeepseekDenseMlpOperation(mlpExpertParam, &mlpExpert59Node.operation);
    mlpExpert59Node.inTensorIds = {
        IN_HIDDEN_STATE,
        IN_MLPGATEUPWEIGHT_EXPERT_FIFTY_NINE,
        IN_MLPDOWNWEIGHT_EXPERT_FIFTY_NINE,
        INTERMIDATE_EXPERT_MASK_FIFTY_NINE,
        INTERMIDATE_FINAL_HIDDEN_STATE_FIFTY_EIGHT};
    mlpExpert59Node.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_FIFTY_NINE};
    ATB_LOG(INFO) << "Expert 59 calculation success";

    deepseekDense::CreateDeepseekDenseMlpOperation(mlpExpertParam, &mlpExpert60Node.operation);
    mlpExpert60Node.inTensorIds = {
        IN_HIDDEN_STATE,
        IN_MLPGATEUPWEIGHT_EXPERT_SIXTY,
        IN_MLPDOWNWEIGHT_EXPERT_SIXTY,
        INTERMIDATE_EXPERT_MASK_SIXTY,
        INTERMIDATE_FINAL_HIDDEN_STATE_FIFTY_NINE};
    mlpExpert60Node.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_SIXTY};
    ATB_LOG(INFO) << "Expert 60 calculation success";

    deepseekDense::CreateDeepseekDenseMlpOperation(mlpExpertParam, &mlpExpert61Node.operation);
    mlpExpert61Node.inTensorIds = {
        IN_HIDDEN_STATE,
        IN_MLPGATEUPWEIGHT_EXPERT_SIXTY_ONE,
        IN_MLPDOWNWEIGHT_EXPERT_SIXTY_ONE,
        INTERMIDATE_EXPERT_MASK_SIXTY_ONE,
        INTERMIDATE_FINAL_HIDDEN_STATE_SIXTY};
    mlpExpert61Node.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_SIXTY_ONE};
    ATB_LOG(INFO) << "Expert 61 calculation success";

    deepseekDense::CreateDeepseekDenseMlpOperation(mlpExpertParam, &mlpExpert62Node.operation);
    mlpExpert62Node.inTensorIds = {
        IN_HIDDEN_STATE,
        IN_MLPGATEUPWEIGHT_EXPERT_SIXTY_TWO,
        IN_MLPDOWNWEIGHT_EXPERT_SIXTY_TWO,
        INTERMIDATE_EXPERT_MASK_SIXTY_TWO,
        INTERMIDATE_FINAL_HIDDEN_STATE_SIXTY_ONE};
    mlpExpert62Node.outTensorIds = {INTERMIDATE_FINAL_HIDDEN_STATE_SIXTY_TWO};
    ATB_LOG(INFO) << "Expert 62 calculation success";

    deepseekDense::CreateDeepseekDenseMlpOperation(mlpExpertParam, &mlpExpert63Node.operation);
    mlpExpert63Node.inTensorIds = {
        IN_HIDDEN_STATE,
        IN_MLPGATEUPWEIGHT_EXPERT_SIXTY_THREE,
        IN_MLPDOWNWEIGHT_EXPERT_SIXTY_THREE,
        INTERMIDATE_EXPERT_MASK_SIXTY_THREE,
        INTERMIDATE_FINAL_HIDDEN_STATE_SIXTY_TWO};
    mlpExpert63Node.outTensorIds = {OUT_DEEPSEEK_DENSE_MOE_ROUT};
    ATB_LOG(INFO) << "Expert 63 calculation success";

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                    atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
    };

    return atb::CreateOperation(opGraph, operation);
}
}
} // namespace atb_speed