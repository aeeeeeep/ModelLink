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
#include <cmath>
#include <numeric>
#include "atb_speed/log.h"
#include "linear_w8a8.h"
#include "linear_parallel_w8a8.h"

namespace atb_speed {
namespace llama2_70b {

enum LinearParallelW8A8TensorId {
    IN_INPUT = 0,
    IN_WEIGHT,
    IN_SCALE,
    IN_OFFSET,
    IN_DESCALE,
    OUT_LINEAROUT,
    INTERMIDATE_LINEAROUT,
};

static const uint64_t IN_TENSOR_COUNT = 5;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 1;
static const uint64_t NODE_COUNT = 2;

atb::Status CreateLinearParallelW8A8(const LinearParallelW8A8Param &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);
    opGraph.name = "LinearParallelW8A8";

    size_t nodeId = 0;
    atb::Node &linearNode = opGraph.nodes.at(nodeId++);
    atb::Node &allReduceNode = opGraph.nodes.at(nodeId++);

    atb_speed::llama2_70b::LinearW8A8Param linearParam = {true};
    CreateLinearW8A8(linearParam, &linearNode.operation, atb_speed::llama2_70b::ROW_PARALLEL);
    linearNode.inTensorIds = {IN_INPUT, IN_WEIGHT, IN_SCALE, IN_OFFSET, IN_DESCALE};
    linearNode.outTensorIds = {INTERMIDATE_LINEAROUT};
    
    atb::infer::AllReduceParam allReduceParam;
    allReduceParam.rank = param.rank;
    allReduceParam.rankSize = param.rankSize;
    allReduceParam.rankRoot = param.rankRoot;
    allReduceParam.allReduceType = "sum";
    allReduceParam.backend = param.backend;
    allReduceParam.hcclComm = param.hcclComm;

    CreateOperation(allReduceParam, &allReduceNode.operation);
    allReduceNode.inTensorIds = {INTERMIDATE_LINEAROUT};
    allReduceNode.outTensorIds = {OUT_LINEAROUT};

    atb::CreateOperation(opGraph, operation);
    return atb::NO_ERROR;
}
} // namespace llama2_70b
} // namespace atb_speed