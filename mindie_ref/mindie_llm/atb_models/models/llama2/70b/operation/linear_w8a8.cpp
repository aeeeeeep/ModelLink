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
#include <cmath>
#include <numeric>
#include "atb_speed/log.h"
#include "linear_w8a8.h"

namespace atb_speed {
namespace llama2_70b {

enum LinearW8A8TensorId {
    IN_INPUT = 0,
    IN_WEIGHT,
    IN_SCALE,
    IN_OFFSET,
    IN_DEQSCALE,
    OUT_LINEAROUT,
    INTERMIDATE_INPUT,
};

static const uint64_t IN_TENSOR_COUNT = 5;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 1;
static const uint64_t NODE_COUNT = 2;

atb::Status CreateLinearW8A8(const LinearW8A8Param &param, atb::Operation **operation, const ParallelType parallelType)
{
    ATB_LOG(INFO) << "CreateLinearW8A8 called";

    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);
    opGraph.name = "LinearW8A8";

    size_t nodeId = 0;
    atb::Node &inputQuantNode = opGraph.nodes.at(nodeId++);
    atb::Node &linearQuantNode = opGraph.nodes.at(nodeId++);

    atb::infer::ElewiseParam inputQuantParam;
    inputQuantParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_QUANT_PER_CHANNEL;
    CreateOperation(inputQuantParam, &inputQuantNode.operation);
    inputQuantNode.inTensorIds = {IN_INPUT, IN_SCALE, IN_OFFSET};
    inputQuantNode.outTensorIds = {INTERMIDATE_INPUT};

    atb::infer::LinearQuantParam linearQuantParam;
    linearQuantParam.transposeA = false;
    linearQuantParam.transposeB = param.transWeight;
    linearQuantParam.hasBias = false;
    CreateOperation(linearQuantParam, &linearQuantNode.operation);
    linearQuantNode.inTensorIds = {INTERMIDATE_INPUT, IN_WEIGHT, IN_DEQSCALE};
    linearQuantNode.outTensorIds = {OUT_LINEAROUT};

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0).format = inTensorDescs.at(0).format;
        outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
        outTensorDescs.at(0).shape = inTensorDescs.at(0).shape;
        auto outDimSize = outTensorDescs.at(0).shape.dimNum;
        if (parallelType == ROW_PARALLEL) {
            if (param.transWeight) {
                outTensorDescs.at(0).shape.dims[outDimSize - 1] = inTensorDescs.at(1).shape.dims[0];
            } else {
                outTensorDescs.at(0).shape.dims[outDimSize - 1] = inTensorDescs.at(1).shape.dims[1];
            }
        } else {
            if (param.transWeight) {
                outTensorDescs.at(0).shape.dims[outDimSize - 1] = inTensorDescs.at(1).shape.dims[0];
            } else {
                outTensorDescs.at(0).shape.dims[outDimSize - 1] = inTensorDescs.at(1).shape.dims[1];
            }
        }
        return atb::NO_ERROR;
    };

    atb::CreateOperation(opGraph, operation);
    return atb::NO_ERROR;
}
} // namespace llama2_70b
} // namespace atb_speed