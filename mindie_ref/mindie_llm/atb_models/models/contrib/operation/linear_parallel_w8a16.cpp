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
#include <cmath>
#include <numeric>
#include "atb_speed/log.h"
#include "layers/plugin_op/w8a16_operation.h"
#include "linear_parallel_w8a16.h"

namespace atb_speed {
namespace contrib {

enum LinearParallelW8A16TensorId {
    IN_INPUT = 0,
    IN_WEIGHT,
    IN_SCALE,
    IN_OFFSET,
    OUT_LINEAROUT,
    INTERMIDATE_LINEAROUT,
};

static const uint64_t IN_TENSOR_COUNT = 4;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 1;
static const uint64_t NODE_COUNT = 2;

atb::Status CreateLinearParallelW8A16(const LinearParallelW8A16Param &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);
    opGraph.name = "LinearParallelW8A16";

    size_t nodeId = 0;
    atb::Node &linearNode = opGraph.nodes.at(nodeId++);
    atb::Node &allReduceNode = opGraph.nodes.at(nodeId++);

    linearNode.operation = new atb_speed::common::W8A16Operation("LinearNode");
    linearNode.inTensorIds = {IN_INPUT, IN_WEIGHT, IN_SCALE, IN_OFFSET};
    linearNode.outTensorIds = {INTERMIDATE_LINEAROUT};

    atb::infer::AllReduceParam allReduceParam = {param.rank, param.rankSize, param.rankRoot,
                                                 "sum",      param.backend,  param.hcclComm};
    CREATE_OPERATION(allReduceParam, &allReduceNode.operation);
    allReduceNode.inTensorIds = {INTERMIDATE_LINEAROUT};
    allReduceNode.outTensorIds = {OUT_LINEAROUT};

    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}
} // namespace contrib
} // namespace atb_speed