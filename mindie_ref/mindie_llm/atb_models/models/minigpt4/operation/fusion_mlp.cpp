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
#include "fusion_mlp.h"

namespace atb_speed {
namespace llama_7b {
enum FusionMlpTensorId {
    IN_HIDDENSTATES_ID = 0,             // [batch, seqLen, hiddenSize], half
    IN_WEIGHT_GATE_UP_ID,                  // [11008*2, hiddenSize], half
    IN_WEIGHT_DOWN_ID,                  // [hiddenSize, 11008], half
    OUT_TRANSPOSED_RESULT_ID,           // [batch, seqLen, hiddenSize], half
    INTERMEDIATE_MATMUL_GATEUP_OUT_ID, // [batch, seqLen, 11008], half
    INTERMEDIATE_MATMUL_GATE_OUT_ID, // [batch, seqLen, 11008], half
    INTERMEDIATE_MATMUL_UP_OUT_ID, // [batch, seqLen, 11008], half
    INTERMEDIATE_SWISH_OUT_ID,          // [batch, seqLen, 11008], half
    INTERMEDIATE_MUL_OUT_ID,            // [batch, seqLen, 11008], half
};

static const uint64_t IN_TENSOR_COUNT = 3;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 5;
static const uint64_t NODE_COUNT = 5;

atb::Status FusionMlp(const FusionMlpParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);
    opGraph.name = "fusion_mlp";

    size_t nodeId = 0;
    auto &matmulGateUpNode = opGraph.nodes.at(nodeId++);
    auto &splitNode = opGraph.nodes.at(nodeId++);
    auto &swishNode = opGraph.nodes.at(nodeId++);
    auto &mulNode = opGraph.nodes.at(nodeId++);
    auto &matmulDownNode = opGraph.nodes.at(nodeId++);

    atb::infer::LinearParam matmulGateParam = {false, param.transpose, false};
    CREATE_OPERATION(matmulGateParam, &matmulGateUpNode.operation);
    matmulGateUpNode.inTensorIds = {IN_HIDDENSTATES_ID, IN_WEIGHT_GATE_UP_ID};
    matmulGateUpNode.outTensorIds = {INTERMEDIATE_MATMUL_GATEUP_OUT_ID};

    atb::infer::SplitParam splitParam = {2, 2};
    CREATE_OPERATION(splitParam, &splitNode.operation);
    splitNode.inTensorIds = {INTERMEDIATE_MATMUL_GATEUP_OUT_ID};
    splitNode.outTensorIds = {INTERMEDIATE_MATMUL_GATE_OUT_ID, INTERMEDIATE_MATMUL_UP_OUT_ID};

    atb::infer::ActivationParam swishParam;
    swishParam.activationType = atb::infer::ActivationType::ACTIVATION_SWISH;
    CREATE_OPERATION(swishParam, &swishNode.operation);
    swishNode.inTensorIds = {INTERMEDIATE_MATMUL_GATE_OUT_ID};
    swishNode.outTensorIds = {INTERMEDIATE_SWISH_OUT_ID};

    atb::infer::ElewiseParam mulParam;
    mulParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    CREATE_OPERATION(mulParam, &mulNode.operation);
    mulNode.inTensorIds = {INTERMEDIATE_SWISH_OUT_ID, INTERMEDIATE_MATMUL_UP_OUT_ID};
    mulNode.outTensorIds = {INTERMEDIATE_MUL_OUT_ID};

    atb::infer::LinearParam matmulDownParam = {false, param.transpose, false};
    CREATE_OPERATION(matmulDownParam, &matmulDownNode.operation);
    matmulDownNode.inTensorIds = {INTERMEDIATE_MUL_OUT_ID, IN_WEIGHT_DOWN_ID};
    matmulDownNode.outTensorIds = {OUT_TRANSPOSED_RESULT_ID};

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
    };

    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}
} // namespace llama_7b
} // namespace atb_speed