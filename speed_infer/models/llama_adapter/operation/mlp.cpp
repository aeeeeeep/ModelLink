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
#include "mlp.h"

#include <cmath>

#include "atb/atb_infer.h"

#include "atb_speed/log.h"

namespace atb_speed {
namespace llama_adapter {
enum MlpAdapterTensorId : int {
    IN_HIDDENSTATES = 0, // [bs, sq, hidden_states]
    IN_WEIGHT_GATE,
    IN_BIAS_GATE,
    IN_WEIGHT_DOWN,
    IN_BIAS_DOWN,
    IN_WEIGHT_UP,
    IN_BIAS_UP,
    OUT_MLP_OUT,        // [bs, sq, hidden_states]
    INTERMEDIATE_MATMUL_GATE_OUT, // [batch, seqLen, 11008], half
    INTERMEDIATE_SWISH_OUT,       // [batch, seqLen, 11008], half
    INTERMEDIATE_MATMUL_UP_OUT,   // [batch, seqLen, 11008], half
    INTERMEDIATE_MUL_OUT,         // [batch, seqLen, 11008], half
};

static const uint64_t IN_TENSOR_COUNT = 7;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 4;
static const uint64_t NODE_COUNT = 5;

atb::Status MlpAdapter(const MlpParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.name = "MlpAdapter";
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    auto &matmulGateNode = opGraph.nodes.at(nodeId++);
    auto &swishNode = opGraph.nodes.at(nodeId++);
    auto &matmulUpNode = opGraph.nodes.at(nodeId++);
    auto &mulNode = opGraph.nodes.at(nodeId++);
    auto &matmulDownNode = opGraph.nodes.at(nodeId++);

    atb::infer::LinearParam matmulGateParam = { false, false, true };
    CREATE_OPERATION(matmulGateParam, &matmulGateNode.operation);
    matmulGateNode.inTensorIds = { IN_HIDDENSTATES, IN_WEIGHT_GATE, IN_BIAS_GATE };
    matmulGateNode.outTensorIds = { INTERMEDIATE_MATMUL_GATE_OUT };

    atb::infer::ActivationParam swishParam;
    swishParam.activationType = atb::infer::ActivationType::ACTIVATION_SWISH;
    CREATE_OPERATION(swishParam, &swishNode.operation);
    swishNode.inTensorIds = { INTERMEDIATE_MATMUL_GATE_OUT };
    swishNode.outTensorIds = { INTERMEDIATE_SWISH_OUT };

    atb::infer::LinearParam matmulUpParam = { false, false, true };
    CREATE_OPERATION(matmulUpParam, &matmulUpNode.operation);
    matmulUpNode.inTensorIds = { IN_HIDDENSTATES, IN_WEIGHT_UP, IN_BIAS_UP };
    matmulUpNode.outTensorIds = { INTERMEDIATE_MATMUL_UP_OUT };

    atb::infer::ElewiseParam mulParam;
    mulParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    CREATE_OPERATION(mulParam, &mulNode.operation);
    mulNode.inTensorIds = { INTERMEDIATE_SWISH_OUT, INTERMEDIATE_MATMUL_UP_OUT };
    mulNode.outTensorIds = { INTERMEDIATE_MUL_OUT };

    atb::infer::LinearParam matmulDownParam = { false, false, true };
    CREATE_OPERATION(matmulDownParam, &matmulDownNode.operation);
    matmulDownNode.inTensorIds = { INTERMEDIATE_MUL_OUT, IN_WEIGHT_DOWN, IN_BIAS_DOWN };
    matmulDownNode.outTensorIds = { OUT_MLP_OUT };

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
        atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
        outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1];
        outTensorDescs.at(0).shape.dims[2] = inTensorDescs.at(0).shape.dims[2];
        return atb::NO_ERROR;
    };

    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}
} // namespace llama_adapter
} // namespace atb_speed
