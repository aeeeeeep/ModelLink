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
#include "quant_parallel_layer.h"

#include <atb/atb_infer.h>
#include <nlohmann/json.hpp>

#include "atb_speed/log.h"

#include "layers/plugin_op/w8a16_operation.h"

namespace atb_speed {
namespace common {
enum ParallelType : int {
    ROW_PARALLEL = 0,
    COLUMN_PARALLEL,
};

template <class T>
atb::Status QuantParallelLinearBase(const QuantParallelParam &param_, atb::Operation **operation, T config,
                                    const ParallelType parallelType)
{
    atb::GraphParam opGraph;
    opGraph.name = "ParallelLinearBase";
    opGraph.inTensorNum = config.inTensorNum;
    opGraph.outTensorNum = config.outTensorNum;
    opGraph.internalTensorNum = config.interTensorNum;
    opGraph.nodes.resize(config.nodeCount);

    std::shared_ptr<int64_t> bsPtr1 = std::make_shared<int64_t>(0);

    size_t nodeId = 0;
    atb::Node &matmulNode = opGraph.nodes.at(nodeId++);

    matmulNode.operation = new atb_speed::common::W8A16Operation(opGraph.name);
    matmulNode.inTensorIds = {config.IN_INPUT, config.IN_WEIGHT, config.IN_DEQUANT_SCALE, config.IN_DEQUANT_OFFSET};
    matmulNode.outTensorIds = {config.INTERMIDATE_MATMULOUT};
    matmulNode.inTensorReshapeFuncs.resize(matmulNode.inTensorIds.size());
    matmulNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        *bsPtr1 = oldShape.dims[0];
        newShape.dimNum = 2; // dimNum is 2
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
        newShape.dims[1] = oldShape.dims[2];
    };

    if (param_.rankSize > 1) {
        atb::Node &parallelNode = opGraph.nodes.at(nodeId++);

        if (parallelType == ROW_PARALLEL) {
            atb::infer::AllReduceParam allReduceParam;
            allReduceParam.rank = param_.rank;
            allReduceParam.rankSize = param_.rankSize;
            allReduceParam.backend = param_.backend;
            atb::CreateOperation(allReduceParam, &parallelNode.operation);
        } else {
            atb::infer::AllGatherParam allGatherParam;
            allGatherParam.rank = param_.rank;
            allGatherParam.rankSize = param_.rankSize;
            allGatherParam.backend = param_.backend;
            atb::CreateOperation(allGatherParam, &parallelNode.operation);
        }

        parallelNode.inTensorIds = {config.INTERMIDATE_MATMULOUT};
        parallelNode.outTensorIds = {config.INTERMIDATE_ALLREDUCEOUT};
        parallelNode.inTensorReshapeFuncs.resize(parallelNode.inTensorIds.size());
        parallelNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 3; // dimNum is 3
        newShape.dims[0] = *bsPtr1;
        newShape.dims[1] = oldShape.dims[0] / *bsPtr1;
        newShape.dims[2] = oldShape.dims[1];
    };
    }

    if (param_.isBias) {
        atb::Node &addNode = opGraph.nodes.at(nodeId++);
        atb::infer::ElewiseParam addParam;
        addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
        atb::CreateOperation(addParam, &addNode.operation);
        addNode.inTensorIds = {param_.rankSize > 1 ? config.INTERMIDATE_ALLREDUCEOUT : config.INTERMIDATE_MATMULOUT,
                               config.IN_BIAS};
        addNode.outTensorIds = {config.OUT_LINEAROUT};
    }

    if (parallelType == ROW_PARALLEL) {
        opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                     atb::SVector<atb::TensorDesc> &outTensorDescs) {
            outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
            outTensorDescs.at(0).format = inTensorDescs.at(0).format;
            auto dimNum = inTensorDescs.at(0).shape.dimNum;
            outTensorDescs.at(0).shape.dimNum = dimNum;
            outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
            if (dimNum == 3) {
                outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1];
            }
            if (param_.transposeB) {
                outTensorDescs.at(0).shape.dims[dimNum - 1] = inTensorDescs.at(1).shape.dims[0];
            } else {
                outTensorDescs.at(0).shape.dims[dimNum - 1] = inTensorDescs.at(1).shape.dims[1];
            }

            return atb::NO_ERROR;
        };
    } else {
        opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                     atb::SVector<atb::TensorDesc> &outTensorDescs) {
            outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
            outTensorDescs.at(0).format = inTensorDescs.at(0).format;
            auto dimNum = inTensorDescs.at(0).shape.dimNum;
            outTensorDescs.at(0).shape.dimNum = dimNum + 1; // add rank dim
            outTensorDescs.at(0).shape.dims[0] = param_.rankSize;
            outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[0];
            if (dimNum == 3) {
                outTensorDescs.at(0).shape.dims[2] = inTensorDescs.at(0).shape.dims[1]; // dim 2
            }
            if (param_.transposeB) {
                outTensorDescs.at(0).shape.dims[dimNum] = inTensorDescs.at(1).shape.dims[0]; // last dim
            } else {
                outTensorDescs.at(0).shape.dims[dimNum] = inTensorDescs.at(1).shape.dims[1]; // last dim
            }

            return atb::NO_ERROR;
        };
    }

    atb::CreateOperation(opGraph, operation);
    return atb::NO_ERROR;
}

atb::Status QuantParallelLinear(const QuantParallelParam &param_, atb::Operation **operation, const ParallelType parallelType)
{
    if (param_.isBias && (param_.rankSize > 1)) {
        // 5:in 1:out 2:inter 3:node
        return QuantParallelLinearBase(param_, operation, QuantLinearWithBiasAndParallel(5, 1, 2, 3), parallelType);
    } else if (param_.isBias) {
        // 5:in 1:out 1:inter 2:node
        return QuantParallelLinearBase(param_, operation, QuantLinearWithBias(5, 1, 1, 2), parallelType);
    } else if (param_.rankSize > 1) {
        // 2:in 1:out 1:inter 2:node
        return QuantParallelLinearBase(param_, operation, QuantLinearWithParallel(4, 1, 1, 2), parallelType);
    } else {
        // 4:in 1:out 0:inter 1:node
        return QuantParallelLinearBase(param_, operation, QuantLinearOnly(4, 1, 0, 1), parallelType);
    }
}

atb::Status QuantRowParallelLinear(const QuantParallelParam &param_, atb::Operation **operation)
{
    return QuantParallelLinear(param_, operation, ROW_PARALLEL);
}

atb::Status QuantColumnParallelLinear(const QuantParallelParam &param_, atb::Operation **operation)
{
    return QuantParallelLinear(param_, operation, COLUMN_PARALLEL);
}

} // namespace common
} // namespace atb_speed
