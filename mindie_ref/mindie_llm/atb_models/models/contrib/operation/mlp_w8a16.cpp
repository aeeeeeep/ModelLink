/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
#include <atb/atb_infer.h>
#include <memory>
#include "atb_speed/log.h"
#include "linear_parallel_w8a16.h"
#include "layers/plugin_op/w8a16_operation.h"
#include "mlp_w8a16.h"

namespace atb_speed {
namespace contrib {

enum MlpW8A16TensorId {
    IN_INPUT = 0,
    IN_MLPUPWEIGHT,
    IN_MLPUPSCALE,
    IN_MLPUPOFFSET,
    IN_MLPGATEWEIGHT,
    IN_MLPGATESCALE,
    IN_MLPGATEOFFSET,
    IN_MLPDOWNWEIGHT,
    IN_MLPDOWNSCALE,
    IN_MLPDOWNOFFSET,
    OUT_MLPRESULT,
    INTERMIDATE_GATE_OUT,
    INTERMIDATE_UP_OUT,
    INTERMIDATE_SWISH_OUT,
    INTERMIDATE_MUL_OUT,
};

static const uint64_t IN_TENSOR_COUNT = 10;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 4;
static const uint64_t NODE_COUNT = 5;

atb::Status CreateMlpW8A16Operation(const MlpW8A16Param &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);
    opGraph.name = "MlpW8A16";

    size_t nodeId = 0;
    atb::Node &linearUpNode = opGraph.nodes.at(nodeId++);
    atb::Node &linearGateNode = opGraph.nodes.at(nodeId++);
    atb::Node &swishNode = opGraph.nodes.at(nodeId++);
    atb::Node &mulNode = opGraph.nodes.at(nodeId++);
    atb::Node &linearDownNode = opGraph.nodes.at(nodeId++);

    linearUpNode.operation = new atb_speed::common::W8A16Operation("MlpUpNode");
    linearUpNode.inTensorIds = {IN_INPUT, IN_MLPUPWEIGHT, IN_MLPUPSCALE, IN_MLPUPOFFSET};
    linearUpNode.outTensorIds = {INTERMIDATE_UP_OUT};

    linearGateNode.operation = new atb_speed::common::W8A16Operation("MlpGateNode");
    linearGateNode.inTensorIds = {IN_INPUT, IN_MLPGATEWEIGHT, IN_MLPGATESCALE, IN_MLPGATEOFFSET};
    linearGateNode.outTensorIds = {INTERMIDATE_GATE_OUT};

    atb::infer::ActivationParam activationParam;
    activationParam.activationType = atb::infer::ActivationType::ACTIVATION_SWISH;
    CREATE_OPERATION(activationParam, &swishNode.operation);
    swishNode.inTensorIds = {INTERMIDATE_GATE_OUT};
    swishNode.outTensorIds = {INTERMIDATE_SWISH_OUT};

    atb::infer::ElewiseParam elewiseParam;
    elewiseParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    CREATE_OPERATION(elewiseParam, &mulNode.operation);
    mulNode.inTensorIds = {INTERMIDATE_SWISH_OUT, INTERMIDATE_UP_OUT};
    mulNode.outTensorIds = {INTERMIDATE_MUL_OUT};

    atb_speed::contrib::LinearParallelW8A16Param linearParallelParam;
    linearParallelParam.transWeight = param.transposeB;
    linearParallelParam.rank = param.rank;
    linearParallelParam.rankSize = param.rankSize;
    linearParallelParam.rankRoot = param.rankRoot;
    linearParallelParam.bias = param.isBias;
    linearParallelParam.parallelType = "RowParallel";
    linearParallelParam.backend = param.backend;
    CreateLinearParallelW8A16(linearParallelParam, &linearDownNode.operation);
    linearDownNode.inTensorIds = {INTERMIDATE_MUL_OUT, IN_MLPDOWNWEIGHT, IN_MLPDOWNSCALE, IN_MLPDOWNOFFSET};
    linearDownNode.outTensorIds = {OUT_MLPRESULT};

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
    };

    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}
} // namespace contrib
} // namespace atb_speed