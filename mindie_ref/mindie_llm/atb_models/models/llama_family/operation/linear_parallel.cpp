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
#include "models/llama_family/operation/linear.h"
#include "models/llama_family/operation/linear_parallel.h"

namespace atb_speed {
namespace llama_family {

static const uint64_t IN_TENSOR_COUNT = 5;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 1;
static const uint64_t NODE_COUNT = 2;

enum LinearParallelTensorIdx : uint32_t {
    IN_INPUT = 0,
    IN_WEIGHT,
    IN_SCALE,
    IN_OFFSET,
    IN_DESCALE,
    OUT_LINEAR_PARALLEL,
    INTERMIDATE_LINEAR_OUT,
};

atb::Status CreateLinearParallel(const LinearParallelParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);
    opGraph.name = param.parallelType == ROW_PARALLEL ? "LinearRowParallel" : "LinearColumnParallel";

    size_t nodeId = 0;

    atb::Node &linearNode = opGraph.nodes.at(nodeId++);
    atb_speed::llama_family::FusionLinearParam linearParam = param.fusionLinearParam;
    FusionLinear(linearParam, &linearNode.operation);
    linearNode.inTensorIds = {LinearParallelTensorIdx::IN_INPUT, LinearParallelTensorIdx::IN_WEIGHT, LinearParallelTensorIdx::IN_SCALE, LinearParallelTensorIdx::IN_OFFSET, LinearParallelTensorIdx::IN_DESCALE};
    linearNode.outTensorIds = {LinearParallelTensorIdx::INTERMIDATE_LINEAR_OUT};

    if (param.parallelType == ROW_PARALLEL) {
        atb::Node &allReduceNode = opGraph.nodes.at(nodeId++);
        atb::infer::AllReduceParam allReduceParam;
        allReduceParam.rank = param.rank;
        allReduceParam.rankSize = param.worldSize;
        allReduceParam.backend = param.backend;
        CreateOperation(allReduceParam, &allReduceNode.operation);
        allReduceNode.inTensorIds = {LinearParallelTensorIdx::INTERMIDATE_LINEAR_OUT};
        allReduceNode.outTensorIds = {LinearParallelTensorIdx::OUT_LINEAR_PARALLEL};
    } else {
        atb::Node &allGatherNode = opGraph.nodes.at(nodeId++);
        atb::infer::AllGatherParam allGatherParam;
        allGatherParam.rank = param.rank;
        allGatherParam.rankSize = param.worldSize;
        allGatherParam.backend = param.backend;
        atb::CreateOperation(allGatherParam, &allGatherNode.operation);
        allGatherNode.inTensorIds = {LinearParallelTensorIdx::INTERMIDATE_LINEAR_OUT};
        allGatherNode.outTensorIds = {LinearParallelTensorIdx::OUT_LINEAR_PARALLEL};
    }

    atb::CreateOperation(opGraph, operation);
    return atb::NO_ERROR;
}

atb::Status LinearParallel(const LinearParallelParam &param_, atb::Operation **operation)
{
    if (param_.worldSize <= 1) {
        return FusionLinear(param_.fusionLinearParam, operation);
    } else if (param_.parallelType == ROW_PARALLEL || param_.parallelType == COLUMN_PARALLEL) {
        return CreateLinearParallel(param_, operation);
    } else {
        ATB_LOG(ERROR) << "LinearParallel operation doesn't support parallelType: " << param_.parallelType;
        return atb::ERROR_INVALID_PARAM;
    }
}

} // namespace llama_family
} // namespace atb_speed