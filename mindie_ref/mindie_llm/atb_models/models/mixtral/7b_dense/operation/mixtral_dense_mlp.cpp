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
#include "mixtral_dense_mlp.h"
#include <atb/atb_infer.h>
#include <memory>

namespace atb_speed {
namespace mixtralDense {
enum MixtralDenseMlpTensorId {
    IN_HIDDENSTATUS = 0,
    IN_MLP_GATE_UP_WEIGHTTENSOR,
    IN_MLP_DOWN_WEIGHTTENSOR,
    IN_EXPERT_MASK_WITH_WEIGHT,
    IN_FINAL_HIDDENS_STATE,
    OUT_MLPRESULTSTENSOR,
    INTERMIDATE_MATMUL_GATE_UP_OUT,
    INTERMIDATE_MATMUL_GATE_OUT,
    INTERMIDATE_MATMUL_UP_OUT,
    INTERMIDATE_SWISH_OUT,
    INTERMIDATE_HIDDENSTATUS,
    INTERMIDATE_MLP_OUT,
    INTERMIDATE_EXPERT_MASK,
    INTERMIDATE_MASKED_MLP_OUT,
};

static const uint64_t IN_TENSOR_COUNT = 5;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 8;
static const uint64_t NODE_COUNT = 8;

atb::Status CreateMixtralDenseMlpOperation(const MixtralDenseMlpParam &param, atb::Operation **operation)
{
    std::shared_ptr<int64_t> batchDimPtr = std::make_shared<int64_t>(0);

    atb::GraphParam opGraph;
    opGraph.name = "MixtralDenseMlp";
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    atb::Node &linearNode = opGraph.nodes.at(nodeId++);
    atb::Node &splitNode = opGraph.nodes.at(nodeId++);
    atb::Node &swishNode = opGraph.nodes.at(nodeId++);
    atb::Node &mulNode = opGraph.nodes.at(nodeId++);
    atb::Node &linearDownNode = opGraph.nodes.at(nodeId++);
    atb::Node &maskSumNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpMulNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpAddNode = opGraph.nodes.at(nodeId++);

    atb::infer::LinearParam linearParam;
    linearParam.transposeA = false;
    linearParam.transposeB = param.transpose;
    linearParam.hasBias = false;
    CreateOperation(linearParam, &linearNode.operation);
    linearNode.inTensorIds = {IN_HIDDENSTATUS, IN_MLP_GATE_UP_WEIGHTTENSOR};
    linearNode.outTensorIds = {INTERMIDATE_MATMUL_GATE_UP_OUT};

    atb::infer::SplitParam splitParam = {1, 2};
    CreateOperation(splitParam, &splitNode.operation);
    splitNode.inTensorIds = {INTERMIDATE_MATMUL_GATE_UP_OUT};
    splitNode.outTensorIds = {INTERMIDATE_MATMUL_GATE_OUT, INTERMIDATE_MATMUL_UP_OUT};

    atb::infer::ActivationParam activationParam;
    activationParam.activationType = atb::infer::ActivationType::ACTIVATION_SWISH;
    CreateOperation(activationParam, &swishNode.operation);
    swishNode.inTensorIds = {INTERMIDATE_MATMUL_GATE_OUT};
    swishNode.outTensorIds = {INTERMIDATE_SWISH_OUT};

    atb::infer::ElewiseParam elewiseParam;
    elewiseParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    CreateOperation(elewiseParam, &mulNode.operation);
    mulNode.inTensorIds = {INTERMIDATE_SWISH_OUT, INTERMIDATE_MATMUL_UP_OUT};
    mulNode.outTensorIds = {INTERMIDATE_HIDDENSTATUS};

    atb::infer::LinearParam linearDownParam;
    linearDownParam.transposeA = false;
    linearDownParam.transposeB = param.transpose;
    linearDownParam.hasBias = false;
    CreateOperation(linearDownParam, &linearDownNode.operation);
    linearDownNode.inTensorIds = {INTERMIDATE_HIDDENSTATUS, IN_MLP_DOWN_WEIGHTTENSOR};
    linearDownNode.outTensorIds = {INTERMIDATE_MLP_OUT};

    atb::infer::ReduceParam maskSumParam;
    maskSumParam.reduceType = atb::infer::ReduceParam::ReduceType::REDUCE_SUM;
    maskSumParam.axis = {1};
    CreateOperation(maskSumParam, &maskSumNode.operation);
    maskSumNode.inTensorIds = {IN_EXPERT_MASK_WITH_WEIGHT};
    maskSumNode.outTensorIds = {INTERMIDATE_EXPERT_MASK};
    maskSumNode.inTensorReshapeFuncs.resize(maskSumNode.inTensorIds.size());
    maskSumNode.inTensorReshapeFuncs[0] = [batchDimPtr](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2; // dimNum: 2
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
        newShape.dims[1] = oldShape.dims[2];
    };

    atb::infer::ElewiseParam mlpMulParam;
    mlpMulParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    CreateOperation(mlpMulParam, &mlpMulNode.operation);
    mlpMulNode.inTensorIds = {INTERMIDATE_MLP_OUT, INTERMIDATE_EXPERT_MASK};
    mlpMulNode.outTensorIds = {INTERMIDATE_MASKED_MLP_OUT};
    mlpMulNode.inTensorReshapeFuncs.resize(mlpMulNode.inTensorIds.size());
    mlpMulNode.inTensorReshapeFuncs[1] = [batchDimPtr](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2; // dimNum: 1
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = 1;
    };

    atb::infer::ElewiseParam mlpAddParam;
    mlpAddParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CreateOperation(mlpAddParam, &mlpAddNode.operation);
    mlpAddNode.inTensorIds = {INTERMIDATE_MASKED_MLP_OUT, IN_FINAL_HIDDENS_STATE};
    mlpAddNode.outTensorIds = {OUT_MLPRESULTSTENSOR};

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                    atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(IN_FINAL_HIDDENS_STATE);
        return atb::NO_ERROR;
    };

    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}
}
} // namespace atb_speed
