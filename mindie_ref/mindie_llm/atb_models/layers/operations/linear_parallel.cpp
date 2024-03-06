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
#include "layers/operations/linear_parallel.h"

namespace atb_speed {
namespace common {

static const uint64_t IN_TENSOR_COUNT = 6;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t ROW_PARALLEL_NO_ADD_INTERMEDIATE_TENSOR_COUNT = 1;
static const uint64_t ROW_PARALLEL_NO_ADD_NODE_COUNT = 2;
static const uint64_t DEFAULT_INTERMEDIATE_TENSOR_COUNT = 2;
static const uint64_t DEFAULT_NODE_COUNT = 3;

enum LinearParallelTensorIdx : uint32_t {
    IN_INPUT = 0,
    IN_WEIGHT,
    IN_SCALE,
    IN_OFFSET,
    IN_DESCALE,
    IN_BIAS,
    OUT_LINEAR_PARALLEL,
    INTERMIDATE_LINEAR_OUT,
    INTERMIDATE_SYNC_OUT
};

atb::Status CreateLinearParallel(const LinearParallelParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    if (param.parallelType == ROW_PARALLEL && !param.biasAfterSync) {
        opGraph.internalTensorNum = ROW_PARALLEL_NO_ADD_INTERMEDIATE_TENSOR_COUNT;
        opGraph.nodes.resize(ROW_PARALLEL_NO_ADD_NODE_COUNT);
        opGraph.name = "LinearRowParallelNoAdd";
    } else {
        opGraph.internalTensorNum = DEFAULT_INTERMEDIATE_TENSOR_COUNT;
        opGraph.nodes.resize(DEFAULT_NODE_COUNT);
        opGraph.name = param.parallelType == COLUMN_PARALLEL ?  "LinearColumnParallel" : "LinearRowParallelAndAdd";
    }

    size_t nodeId = 0;

    atb::Node &linearNode = opGraph.nodes.at(nodeId++);
    atb_speed::common::FusionLinearParam linearParam = param.fusionLinearParam;
    FusionLinear(linearParam, &linearNode.operation);
    linearNode.inTensorIds = {
        LinearParallelTensorIdx::IN_INPUT, LinearParallelTensorIdx::IN_WEIGHT, LinearParallelTensorIdx::IN_SCALE,
        LinearParallelTensorIdx::IN_OFFSET, LinearParallelTensorIdx::IN_DESCALE, LinearParallelTensorIdx::IN_BIAS
    };
    linearNode.outTensorIds = {LinearParallelTensorIdx::INTERMIDATE_LINEAR_OUT};

    if (param.parallelType == ROW_PARALLEL) {
        atb::Node &allReduceNode = opGraph.nodes.at(nodeId++);
        atb::infer::AllReduceParam allReduceParam;
        allReduceParam.rank = param.tensorParallelInfo.rank;
        allReduceParam.rankSize = param.tensorParallelInfo.worldSize;
        allReduceParam.backend = param.tensorParallelInfo.backend;
        CREATE_OPERATION(allReduceParam, &allReduceNode.operation);
        allReduceNode.inTensorIds = {LinearParallelTensorIdx::INTERMIDATE_LINEAR_OUT};
        allReduceNode.outTensorIds = {
            param.biasAfterSync ? LinearParallelTensorIdx::INTERMIDATE_SYNC_OUT : LinearParallelTensorIdx::OUT_LINEAR_PARALLEL
        };

        if (param.biasAfterSync) {
            atb::Node &addNode = opGraph.nodes.at(nodeId++);
            atb::infer::ElewiseParam addParam;
            addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
            CREATE_OPERATION(addParam, &addNode.operation);
            addNode.inTensorIds = {LinearParallelTensorIdx::INTERMIDATE_SYNC_OUT, LinearParallelTensorIdx::IN_BIAS};
            addNode.outTensorIds = {LinearParallelTensorIdx::OUT_LINEAR_PARALLEL};
        }
    } else {
        atb::Node &allGatherNode = opGraph.nodes.at(nodeId++);
        atb::infer::AllGatherParam allGatherParam;
        allGatherParam.rank = param.tensorParallelInfo.rank;
        allGatherParam.rankSize = param.tensorParallelInfo.worldSize;
        allGatherParam.backend = param.tensorParallelInfo.backend;
        CREATE_OPERATION(allGatherParam, &allGatherNode.operation);
        allGatherNode.inTensorIds = {LinearParallelTensorIdx::INTERMIDATE_LINEAR_OUT};
        allGatherNode.outTensorIds = {LinearParallelTensorIdx::INTERMIDATE_SYNC_OUT};

        atb::Node &transposeNode = opGraph.nodes.at(nodeId++);
        atb::infer::TransposeParam transposeParam;
        if (param.unpadInputs) {
            transposeParam.perm = {1, 0, 2};
        } else {
            transposeParam.perm = {1, 2, 0, 3};
        }
        CREATE_OPERATION(transposeParam, &transposeNode.operation);
        transposeNode.inTensorIds = {LinearParallelTensorIdx::INTERMIDATE_SYNC_OUT};
        transposeNode.outTensorIds = {LinearParallelTensorIdx::OUT_LINEAR_PARALLEL};
    }

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        auto dimLast = inTensorDescs.at(0).shape.dimNum - 1;
        if (param.parallelType == COLUMN_PARALLEL) {
            outTensorDescs.at(0).shape.dims[dimLast] \
                = inTensorDescs.at(1).shape.dims[0] * param.tensorParallelInfo.worldSize;
        } else {
            outTensorDescs.at(0).shape.dims[dimLast] = param.fusionLinearParam.quantType == W8A16 \
                ? inTensorDescs.at(1).shape.dims[1] : inTensorDescs.at(1).shape.dims[0];
        }
        return atb::NO_ERROR;
    };

    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}

atb::Status LinearParallel(const LinearParallelParam &param_, atb::Operation **operation)
{
    if (param_.tensorParallelInfo.worldSize <= 1) {
        return FusionLinear(param_.fusionLinearParam, operation);
    } else if (param_.parallelType == ROW_PARALLEL) {
        return CreateLinearParallel(param_, operation);
    } else if (param_.parallelType == COLUMN_PARALLEL) {
        return CreateLinearParallel(param_, operation);
    } else {
        ATB_LOG(ERROR) << "LinearParallel operation doesn't support parallelType: " << param_.parallelType;
        return atb::ERROR_INVALID_PARAM;
    }
}

} // namespace common
} // namespace atb_speed