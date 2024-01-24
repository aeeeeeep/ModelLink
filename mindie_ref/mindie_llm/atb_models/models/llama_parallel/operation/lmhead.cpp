
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

#include <atb/atb_infer.h>
#include "atb_speed/log.h"

#include "models/llama_parallel/operation/lmhead.h"

namespace atb_speed {
namespace llama_parallel {

enum LmHeadTensorIdx : uint32_t {
    IN_HIDDENSTATES = 0,
    IN_WEIGHT,
    IN_SCALE,
    IN_OFFSET,
    IN_DESCALE,
    IN_INDICES,
    OUT_LOGITS,
};

static const uint64_t IN_TENSOR_COUNT = 6;
static const uint64_t OUT_TENSOR_COUNT = 1;

template <class T>
atb::Status CreateLmHead(const LmHeadParam &param, atb::Operation **operation, T config)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = param.gatherAhead ? config.INTERMEDIATE_TENSOR_COUNT : config.INTERMEDIATE_TENSOR_COUNT - 1;
    opGraph.nodes.resize(param.gatherAhead ? config.NODE_COUNT : config.NODE_COUNT - 1);
    opGraph.name = "LmHead";

    size_t nodeId = 0;

    if (param.gatherAhead) {
        auto &gatherNode = opGraph.nodes.at(nodeId++);
        atb::infer::GatherParam gatherParam;
        gatherParam.axis = param.unpadInputs ? 0 : 1;
        CreateOperation(gatherParam, &gatherNode.operation);
        gatherNode.inTensorIds = {LmHeadTensorIdx::IN_HIDDENSTATES, LmHeadTensorIdx::IN_INDICES};
        gatherNode.outTensorIds = {config.INTERMEDIATE_GATHER_OUT};
    }

    if (param.linearParallelParam.parallelType == ROW_PARALLEL) {
        atb::Node &sliceNode = opGraph.nodes.at(nodeId++);
        atb::infer::SliceParam slicePassParam;
        slicePassParam.offsets = {0, 0, param.hiddenSizePerAttentionHead * param.linearParallelParam.rank};
        slicePassParam.size = {-1, -1, param.hiddenSizePerAttentionHead};
        CreateOperation(slicePassParam, &sliceNode.operation);
        sliceNode.inTensorIds = {param.gatherAhead ? config.INTERMEDIATE_GATHER_OUT : LmHeadTensorIdx::IN_HIDDENSTATES};
        sliceNode.outTensorIds = {config.INTERMEDIATE_SLICE_OUT};
    }

    atb::Node &linearParallelNode = opGraph.nodes.at(nodeId++);
    LinearParallel(param.linearParallelParam, &linearParallelNode.operation);
    linearParallelNode.inTensorIds = {
        param.linearParallelParam.parallelType == ROW_PARALLEL ? config.INTERMEDIATE_SLICE_OUT : param.gatherAhead ? config.INTERMEDIATE_GATHER_OUT : LmHeadTensorIdx::IN_HIDDENSTATES,
        LmHeadTensorIdx::IN_WEIGHT, LmHeadTensorIdx::IN_SCALE, LmHeadTensorIdx::IN_OFFSET, LmHeadTensorIdx::IN_DESCALE};
    linearParallelNode.outTensorIds = {
        param.linearParallelParam.parallelType == COLUMN_PARALLEL ? config.INTERMEDIATE_LINEAR_PARALLEL_OUT : LmHeadTensorIdx::OUT_LOGITS
    };

    if (param.linearParallelParam.parallelType == COLUMN_PARALLEL) {
        atb::Node &transposeNode = opGraph.nodes.at(nodeId++);
        atb::infer::TransposeParam transposeParam;
        if (param.unpadInputs) {
            transposeParam.perm = {1, 0, 2};
        } else {
            transposeParam.perm = {1, 2, 0, 3};
        }
        CreateOperation(transposeParam, &transposeNode.operation);
        transposeNode.inTensorIds = {config.INTERMEDIATE_LINEAR_PARALLEL_OUT};
        transposeNode.outTensorIds = {LmHeadTensorIdx::OUT_LOGITS};
    }

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        auto dimLast = inTensorDescs.at(0).shape.dimNum - 1;
        if (param.unpadInputs) {
            outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(2).shape.dims[0];
        }
        outTensorDescs.at(0).shape.dims[dimLast] = inTensorDescs.at(1).shape.dims[0] * param.linearParallelParam.worldSize;
        return atb::NO_ERROR;
    };

    return atb::CreateOperation(opGraph, operation);
}

class LmHeadNoParallelConfig {
public:
    uint64_t NODE_COUNT = 2;
    uint64_t INTERMEDIATE_TENSOR_COUNT = 1;

    enum LmHeadNoParallelTensorIdx : uint32_t {
        INTERMEDIATE_GATHER_OUT = LmHeadTensorIdx::OUT_LOGITS + 1,
        INTERMEDIATE_LINEAR_PARALLEL_OUT,  // no usage
        INTERMEDIATE_SLICE_OUT  // no usage
    };
};

class LmHeadRowParallelConfig {
public:

    uint64_t NODE_COUNT = 3;
    uint64_t INTERMEDIATE_TENSOR_COUNT = 2;

    enum LmHeadRowParallelTensorIdx : uint32_t {
        INTERMEDIATE_SLICE_OUT = LmHeadTensorIdx::OUT_LOGITS + 1,
        INTERMEDIATE_GATHER_OUT,
        INTERMEDIATE_LINEAR_PARALLEL_OUT  // no usage
    };
};

class LmHeadColumnParallelConfig {
public:

    uint64_t NODE_COUNT = 3;
    uint64_t INTERMEDIATE_TENSOR_COUNT = 2;

    enum LmHeadColumnParallelTensorIdx : uint32_t {
        INTERMEDIATE_LINEAR_PARALLEL_OUT = LmHeadTensorIdx::OUT_LOGITS + 1,
        INTERMEDIATE_GATHER_OUT,
        INTERMEDIATE_SLICE_OUT  // no usage
    };
};

atb::Status LmHead(const LmHeadParam &param_, atb::Operation **operation)
{
    if (param_.linearParallelParam.worldSize <= 1) {
        LmHeadNoParallelConfig lmHeadNoParallelConfig;
        return CreateLmHead(param_, operation, lmHeadNoParallelConfig);
    } else if (param_.linearParallelParam.parallelType == ROW_PARALLEL) {
        LmHeadRowParallelConfig lmHeadRowParallelConfig;
        return CreateLmHead(param_, operation, lmHeadRowParallelConfig);
    } else if (param_.linearParallelParam.parallelType == COLUMN_PARALLEL) {
        LmHeadColumnParallelConfig lmHeadColumnParallelConfig;
        return CreateLmHead(param_, operation, lmHeadColumnParallelConfig);
    } else {
        ATB_LOG(ERROR) << "LmHead operation doesn't support parallelType: " << param_.linearParallelParam.parallelType;
        return atb::ERROR_INVALID_PARAM;
    }
}

} // namespace common
} // namespace atb_speed