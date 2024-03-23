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
#include "layers/plugin_op/w8a16_operation.h"
#include "layers/plugin_op/w8a16_bias_operation.h"
#include "layers/plugin_op/w8a16_operation.h"
#include "layers/quant_parallel_layer.h"

namespace atb_speed {
namespace glm130b {

enum MlpTensorId : int {
    IN_HIDDENSTATES_ID = 0,             // [batch, seqLen, hiddenSize], half
    IN_WEIGHT_UP_ID,                    // [hiddenSize, ffnHiddenSize], half
    IN_WEIGHT_DOWN_ID,                  // [ffnHiddenSize, hiddenSize], half
    IN_BIAS_UP_ID,                      //
    IN_BIAS_DOWN_ID,                    //
    IN_ANTIQUQNT_SCALE_5,
    IN_ANTIQUQNT_OFFSET_5,
    IN_ANTIQUQNT_SCALE_6,
    IN_ANTIQUQNT_OFFSET_6,

    OUT_TRANSPOSED_RESULT_ID,           // [batch, seqLen, hiddenSize], half

    INTERMEDIATE_MATMUL_GATEUP_OUT_ID,
    INTERMEDIATE_SPLIT_OUTPUTA_ID,
    INTERMEDIATE_SPLIT_OUTPUTB_ID,
    INTERMEDIATE_GELU_OUTPUT_ID,
    INTERMEDIATE_GEGLU_OUTPUT_ID,
};

atb::Status FusionMlpGlmBase(const FusionMlpGlmParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = 9;
    opGraph.outTensorNum = 1;
    opGraph.internalTensorNum = 5;
    opGraph.nodes.resize(5);
    opGraph.name = "fusion_mlp";

    size_t nodeId = 0;
    auto &matmulGateUpNode = opGraph.nodes.at(nodeId++);
    atb::Node &splitNode = opGraph.nodes.at(nodeId++);
    atb::Node &geluNode = opGraph.nodes.at(nodeId++);
    atb::Node &mulNode = opGraph.nodes.at(nodeId++);
    auto &matmulDownNode = opGraph.nodes.at(nodeId++);

    std::shared_ptr<int64_t> bsPtr2 = std::make_shared<int64_t>(0);

    matmulGateUpNode.operation = new atb_speed::common::W8A16BiasOperation("matmulGateUpNode_Bias");
    matmulGateUpNode.inTensorIds = {IN_HIDDENSTATES_ID, IN_WEIGHT_UP_ID, IN_ANTIQUQNT_SCALE_5,
                                    IN_ANTIQUQNT_OFFSET_5, IN_BIAS_UP_ID};
    matmulGateUpNode.outTensorIds = {INTERMEDIATE_MATMUL_GATEUP_OUT_ID};
    matmulGateUpNode.inTensorReshapeFuncs.resize(matmulGateUpNode.inTensorIds.size());
    matmulGateUpNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        *bsPtr2 = oldShape.dims[0];

        newShape.dimNum = 2; // dimNum is 2
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
        newShape.dims[1] = oldShape.dims[2];
    };
    
    atb::infer::SplitParam splitParam;
    splitParam.splitDim = 2; // 2: 在第三维上进行切分
    splitParam.splitNum = 2; // 2: 进行二等分
    CreateOperation(splitParam, &splitNode.operation);
    splitNode.inTensorIds = {INTERMEDIATE_MATMUL_GATEUP_OUT_ID};
    splitNode.outTensorIds = {INTERMEDIATE_SPLIT_OUTPUTA_ID,
                              INTERMEDIATE_SPLIT_OUTPUTB_ID}; // [bs, seq_len, hidden_size * 8 / 3 / world_size]
    splitNode.inTensorReshapeFuncs.resize(splitNode.inTensorIds.size());
    splitNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 3; // dimNum is 3
        newShape.dims[0] = *bsPtr2;
        newShape.dims[1] = oldShape.dims[0] / *bsPtr2;
        newShape.dims[2] = oldShape.dims[1];
    };

    atb::infer::ActivationParam activationParam;
    activationParam.activationType = atb::infer::ActivationType::ACTIVATION_GELU;
    CreateOperation(activationParam, &geluNode.operation);
    geluNode.inTensorIds = {INTERMEDIATE_SPLIT_OUTPUTB_ID};
    geluNode.outTensorIds = {INTERMEDIATE_GELU_OUTPUT_ID};

    atb::infer::ElewiseParam elewiseParam;
    elewiseParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    CreateOperation(elewiseParam, &mulNode.operation);
    mulNode.inTensorIds = {INTERMEDIATE_SPLIT_OUTPUTA_ID, INTERMEDIATE_GELU_OUTPUT_ID};
    mulNode.outTensorIds = {INTERMEDIATE_GEGLU_OUTPUT_ID};

    atb_speed::common::QuantParallelParam matmulDownParam;
    matmulDownParam.rank = param.rank;
    matmulDownParam.rankSize = param.rankSize;
    matmulDownParam.isBias = param.isBias;
    matmulDownParam.backend = param.backend;
    atb_speed::common::QuantRowParallelLinear(matmulDownParam, &matmulDownNode.operation);
    if (matmulDownParam.isBias) {
        matmulDownNode.inTensorIds = {INTERMEDIATE_GEGLU_OUTPUT_ID, IN_WEIGHT_DOWN_ID,
                                      IN_ANTIQUQNT_SCALE_6, IN_ANTIQUQNT_OFFSET_6, IN_BIAS_DOWN_ID};
    } else {
        matmulDownNode.inTensorIds = {INTERMEDIATE_GEGLU_OUTPUT_ID, IN_WEIGHT_DOWN_ID,
                                      IN_ANTIQUQNT_SCALE_6, IN_ANTIQUQNT_OFFSET_6};
    }
    matmulDownNode.outTensorIds = {OUT_TRANSPOSED_RESULT_ID};

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
    };

    atb::CreateOperation(opGraph, operation);
    return atb::NO_ERROR;
}

atb::Status FusionMlpGlm(const FusionMlpGlmParam &param, atb::Operation **operation)
{
    return FusionMlpGlmBase(param, operation);
}
} // namespace llama_7b
} // namespace atb_speed