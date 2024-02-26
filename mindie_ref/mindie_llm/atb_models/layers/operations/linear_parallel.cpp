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

static const uint64_t IN_TENSOR_COUNT = 5;
static const uint64_t OUT_TENSOR_COUNT = 1;

enum LinearParallelTensorIdx : uint32_t {
    IN_INPUT = 0,
    IN_WEIGHT,
    IN_SCALE,
    IN_OFFSET,
    IN_DESCALE,
    OUT_LINEAR_PARALLEL,
};

template <class T>
atb::Status CreateLinearParallel(const LinearParallelParam &param, atb::Operation **operation, T config)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = config.INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(config.NODE_COUNT);
    opGraph.name = param.parallelType == ROW_PARALLEL ? "LinearRowParallel" : "LinearColumnParallel";

    size_t nodeId = 0;

    atb::Node &linearNode = opGraph.nodes.at(nodeId++);
    atb_speed::common::FusionLinearParam linearParam = param.fusionLinearParam;
    FusionLinear(linearParam, &linearNode.operation);
    linearNode.inTensorIds = {
        LinearParallelTensorIdx::IN_INPUT, LinearParallelTensorIdx::IN_WEIGHT, LinearParallelTensorIdx::IN_SCALE,
        LinearParallelTensorIdx::IN_OFFSET, LinearParallelTensorIdx::IN_DESCALE
    };
    linearNode.outTensorIds = {config.INTERMIDATE_LINEAR_OUT};

    if (param.parallelType == ROW_PARALLEL) {
        atb::Node &allReduceNode = opGraph.nodes.at(nodeId++);
        atb::infer::AllReduceParam allReduceParam;
        allReduceParam.rank = param.rank;
        allReduceParam.rankSize = param.worldSize;
        allReduceParam.backend = param.backend;
        allReduceParam.rankTableFile = param.rankTableFile;
        CREATE_OPERATION(allReduceParam, &allReduceNode.operation);
        allReduceNode.inTensorIds = {config.INTERMIDATE_LINEAR_OUT};
        allReduceNode.outTensorIds = {LinearParallelTensorIdx::OUT_LINEAR_PARALLEL};
    } else {
        atb::Node &allGatherNode = opGraph.nodes.at(nodeId++);
        atb::infer::AllGatherParam allGatherParam;
        allGatherParam.rank = param.rank;
        allGatherParam.rankSize = param.worldSize;
        allGatherParam.backend = param.backend;
        allGatherParam.rankTableFile = param.rankTableFile;
        CREATE_OPERATION(allGatherParam, &allGatherNode.operation);
        allGatherNode.inTensorIds = {config.INTERMIDATE_LINEAR_OUT};
        allGatherNode.outTensorIds = {config.INTERMIDATE_ALL_GATHER_OUT};
    }

    if (param.parallelType == COLUMN_PARALLEL) {
        atb::Node &transposeNode = opGraph.nodes.at(nodeId++);
        atb::infer::TransposeParam transposeParam;
        if (param.unpadInputs) {
            transposeParam.perm = {1, 0, 2};
        } else {
            transposeParam.perm = {1, 2, 0, 3};
        }
        CREATE_OPERATION(transposeParam, &transposeNode.operation);
        transposeNode.inTensorIds = {config.INTERMIDATE_ALL_GATHER_OUT};
        transposeNode.outTensorIds = {LinearParallelTensorIdx::OUT_LINEAR_PARALLEL};
    }

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        auto dimLast = inTensorDescs.at(0).shape.dimNum - 1;
        if (param.parallelType == COLUMN_PARALLEL) {
            outTensorDescs.at(0).shape.dims[dimLast] \
                = inTensorDescs.at(1).shape.dims[0] * param.worldSize;
        } else {
            outTensorDescs.at(0).shape.dims[dimLast] = inTensorDescs.at(1).shape.dims[0];
        }
        return atb::NO_ERROR;
    };

    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}

class LinearRowParallelConfig {
public:

    uint64_t NODE_COUNT = 2;
    uint64_t INTERMEDIATE_TENSOR_COUNT = 1;

    enum LinearRowParallelTensorIdx : uint32_t {
        INTERMIDATE_LINEAR_OUT = LinearParallelTensorIdx::OUT_LINEAR_PARALLEL + 1,
        INTERMIDATE_ALL_GATHER_OUT,  // no usage
    };
};

class LinearColumnParallelConfig {
public:

    uint64_t NODE_COUNT = 3;
    uint64_t INTERMEDIATE_TENSOR_COUNT = 2;

    enum LinearColumnParallelTensorIdx : uint32_t {
        INTERMIDATE_LINEAR_OUT = LinearParallelTensorIdx::OUT_LINEAR_PARALLEL + 1,
        INTERMIDATE_ALL_GATHER_OUT,
    };
};

atb::Status LinearParallel(const LinearParallelParam &param_, atb::Operation **operation)
{
    if (param_.worldSize <= 1) {
        return FusionLinear(param_.fusionLinearParam, operation);
    } else if (param_.parallelType == ROW_PARALLEL) {
        LinearRowParallelConfig linearRowParallelConfig;
        return CreateLinearParallel(param_, operation, linearRowParallelConfig);
    } else if (param_.parallelType == COLUMN_PARALLEL) {
        LinearColumnParallelConfig linearColumnParallelConfig;
        return CreateLinearParallel(param_, operation, linearColumnParallelConfig);
    } else {
        ATB_LOG(ERROR) << "LinearParallel operation doesn't support parallelType: " << param_.parallelType;
        return atb::ERROR_INVALID_PARAM;
    }
}

} // namespace common
} // namespace atb_speed