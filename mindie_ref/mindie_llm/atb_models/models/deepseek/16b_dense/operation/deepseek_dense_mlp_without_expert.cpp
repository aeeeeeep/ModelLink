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
#include "deepseek_dense_mlp_without_expert.h"
#include <atb/atb_infer.h>
#include <memory>

namespace atb_speed {
namespace deepseekDense {
enum DeepseekDenseMlpWithoutExpertTensorId {
    IN_HIDDENSTATUS = 0,
    IN_MLP_GATE_UP_WEIGHTTENSOR,
    IN_MLP_DOWN_WEIGHTTENSOR,
    OUT_MLP_OUT_TENSOR,
    INTERMIDATE_MATMUL_GATE_UP_OUT,
    INTERMIDATE_MATMUL_GATE_OUT,
    INTERMIDATE_MATMUL_UP_OUT,
    INTERMIDATE_SWISH_OUT,
    INTERMIDATE_HIDDENSTATUS,
};

static const uint64_t IN_TENSOR_COUNT = 3;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 5;
static const uint64_t NODE_COUNT = 5;

atb::Status CreateDeepseekDenseMlpWithoutExpertOperation(
    const DeepseekDenseMlpWithoutExpertParam &param, atb::Operation **operation)
{
    std::shared_ptr<int64_t> batchDimPtr = std::make_shared<int64_t>(0);

    atb::GraphParam opGraph;
    opGraph.name = "DeepseekDenseMlpWithoutExpert";
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

    atb::infer::MatmulParam linearParam = {false, param.transpose};
    CreateOperation(linearParam, &linearNode.operation);
    linearNode.inTensorIds = {IN_HIDDENSTATUS, IN_MLP_GATE_UP_WEIGHTTENSOR};
    linearNode.outTensorIds = {INTERMIDATE_MATMUL_GATE_UP_OUT};
    linearNode.inTensorReshapeFuncs.resize(linearNode.inTensorIds.size());
    linearNode.inTensorReshapeFuncs[0] = [batchDimPtr](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2; // dimNum: 2
        *batchDimPtr = oldShape.dims[0];
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
        newShape.dims[1] = oldShape.dims[2];
    };

    atb::infer::SplitParam splitParam = {2, 2};
    CreateOperation(splitParam, &splitNode.operation);
    splitNode.inTensorIds = {INTERMIDATE_MATMUL_GATE_UP_OUT};
    splitNode.outTensorIds = {INTERMIDATE_MATMUL_GATE_OUT, INTERMIDATE_MATMUL_UP_OUT};
    splitNode.inTensorReshapeFuncs.resize(splitNode.inTensorIds.size());
    splitNode.inTensorReshapeFuncs[0] = [batchDimPtr](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 3; // dimNum: 3
        newShape.dims[0] = (*batchDimPtr);
        newShape.dims[1] = oldShape.dims[0] / (*batchDimPtr);
        newShape.dims[2] = oldShape.dims[1];
    };

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

    atb::infer::MatmulParam linearDownParam = {false, param.transpose};
    CreateOperation(linearDownParam, &linearDownNode.operation);
    linearDownNode.inTensorIds = {INTERMIDATE_HIDDENSTATUS, IN_MLP_DOWN_WEIGHTTENSOR};
    linearDownNode.outTensorIds = {OUT_MLP_OUT_TENSOR};

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                    atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(IN_HIDDENSTATUS);
        return atb::NO_ERROR;
    };

    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}
}
} // namespace atb_speed
