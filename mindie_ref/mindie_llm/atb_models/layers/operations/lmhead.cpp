
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
#include "atb_speed/log.h"

#include "layers/operations/lmhead.h"

namespace atb_speed {
namespace common {

enum LmHeadTensorIdx : uint32_t {
    IN_HIDDENSTATES = 0,
    IN_WEIGHT,
    IN_SCALE,
    IN_OFFSET,
    IN_DESCALE,
    IN_DEOFFSET,
    IN_INDICES,
    OUT_LOGITS,
};

static const uint64_t IN_TENSOR_COUNT = 7;
static const uint64_t OUT_TENSOR_COUNT = 1;

template <class T>
atb::Status CreateLmHead(const LmHeadParam &param, atb::Operation **operation, T config, atb_speed::common::LinearParallelType parallelType)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum =
        param.gatherAhead ? config.INTERMEDIATE_TENSOR_COUNT : config.INTERMEDIATE_TENSOR_COUNT - 1;
    opGraph.nodes.resize(param.gatherAhead ? config.NODE_COUNT : config.NODE_COUNT - 1);
    opGraph.name = "LmHead";

    size_t nodeId = 0;

    if (param.gatherAhead) {
        auto &gatherNode = opGraph.nodes.at(nodeId++);
        atb::infer::GatherParam gatherParam;
        gatherParam.axis = param.unpadInputs ? 0 : 1;
        CREATE_OPERATION(gatherParam, &gatherNode.operation);
        gatherNode.inTensorIds = {LmHeadTensorIdx::IN_HIDDENSTATES, LmHeadTensorIdx::IN_INDICES};
        gatherNode.outTensorIds = {config.INTERMEDIATE_GATHER_OUT};
    }

    if (parallelType == ROW_PARALLEL) {
        atb::Node &sliceNode = opGraph.nodes.at(nodeId++);
        atb::infer::SliceParam slicePassParam;
        if (param.unpadInputs) {
            slicePassParam.offsets = {0, param.hiddenSizePerAttentionHead * param.linearParallelParam.tensorParallelInfo.rank};
            slicePassParam.size = {-1, param.hiddenSizePerAttentionHead};
        } else {
            slicePassParam.offsets = {0, 0, param.hiddenSizePerAttentionHead * param.linearParallelParam.tensorParallelInfo.rank};
            slicePassParam.size = {-1, -1, param.hiddenSizePerAttentionHead};
        }
        CREATE_OPERATION(slicePassParam, &sliceNode.operation);
        if (param.gatherAhead) {
            sliceNode.inTensorIds = {config.INTERMEDIATE_GATHER_OUT};
        } else {
            sliceNode.inTensorIds = {LmHeadTensorIdx::IN_HIDDENSTATES};
        }
        sliceNode.outTensorIds = {config.INTERMEDIATE_SLICE_OUT};
    }

    atb::Node &linearParallelNode = opGraph.nodes.at(nodeId++);
    LinearParallel(param.linearParallelParam, &linearParallelNode.operation);
    if (parallelType == ROW_PARALLEL) {
        linearParallelNode.inTensorIds = {
            config.INTERMEDIATE_SLICE_OUT, LmHeadTensorIdx::IN_WEIGHT, LmHeadTensorIdx::IN_SCALE,
            LmHeadTensorIdx::IN_OFFSET, LmHeadTensorIdx::IN_DESCALE, LmHeadTensorIdx::IN_DEOFFSET
        };
    } else if (param.gatherAhead) {
        linearParallelNode.inTensorIds = {
            config.INTERMEDIATE_GATHER_OUT, LmHeadTensorIdx::IN_WEIGHT, LmHeadTensorIdx::IN_SCALE,
            LmHeadTensorIdx::IN_OFFSET, LmHeadTensorIdx::IN_DESCALE, LmHeadTensorIdx::IN_DEOFFSET
        };
    } else {
        linearParallelNode.inTensorIds = {
            LmHeadTensorIdx::IN_HIDDENSTATES, LmHeadTensorIdx::IN_WEIGHT, LmHeadTensorIdx::IN_SCALE,
            LmHeadTensorIdx::IN_OFFSET, LmHeadTensorIdx::IN_DESCALE, LmHeadTensorIdx::IN_DEOFFSET
        };
    }
    linearParallelNode.outTensorIds = {LmHeadTensorIdx::OUT_LOGITS};

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        auto dimLast = inTensorDescs.at(0).shape.dimNum - 1;
        if (param.gatherAhead) {
            outTensorDescs.at(0).shape.dims[param.unpadInputs ? 0 : 1] = inTensorDescs.at(5).shape.dims[0];
        }
        if (parallelType == COLUMN_PARALLEL) {
            outTensorDescs.at(0).shape.dims[dimLast] \
                = inTensorDescs.at(1).shape.dims[0] * param.linearParallelParam.tensorParallelInfo.worldSize;
        } else {
            outTensorDescs.at(0).shape.dims[dimLast] = inTensorDescs.at(1).shape.dims[0];
        }
        return atb::NO_ERROR;
    };

    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}

class LmHeadNoParallelConfig {
public:
    uint64_t NODE_COUNT = 2;
    uint64_t INTERMEDIATE_TENSOR_COUNT = 1;

    enum LmHeadNoParallelTensorIdx : uint32_t {
        INTERMEDIATE_GATHER_OUT = LmHeadTensorIdx::OUT_LOGITS + 1,
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
    };
};

class LmHeadColumnParallelConfig {
public:

    uint64_t NODE_COUNT = 2;
    uint64_t INTERMEDIATE_TENSOR_COUNT = 1;

    enum LmHeadColumnParallelTensorIdx : uint32_t {
        INTERMEDIATE_GATHER_OUT = LmHeadTensorIdx::OUT_LOGITS + 1,
        INTERMEDIATE_SLICE_OUT  // no usage
    };
};

atb::Status LmHead(const LmHeadParam &param_, atb::Operation **operation)
{
    if (param_.linearParallelParam.tensorParallelInfo.worldSize <= 1) {
        LmHeadNoParallelConfig lmHeadNoParallelConfig;
        return CreateLmHead(param_, operation, lmHeadNoParallelConfig, UNDEFINED);
    } else if (param_.linearParallelParam.parallelType == ROW_PARALLEL) {
        LmHeadRowParallelConfig lmHeadRowParallelConfig;
        return CreateLmHead(param_, operation, lmHeadRowParallelConfig, ROW_PARALLEL);
    } else if (param_.linearParallelParam.parallelType == COLUMN_PARALLEL) {
        LmHeadColumnParallelConfig lmHeadColumnParallelConfig;
        return CreateLmHead(param_, operation, lmHeadColumnParallelConfig, COLUMN_PARALLEL);
    } else {
        ATB_LOG(ERROR) << "LmHead operation doesn't support parallelType: " << param_.linearParallelParam.parallelType;
        return atb::ERROR_INVALID_PARAM;
    }
}

} // namespace common
} // namespace atb_speed