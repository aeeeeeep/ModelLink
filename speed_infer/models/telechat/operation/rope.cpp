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
#include "rope.h"

namespace atb_speed {
namespace telechat {
enum RopeTensorId {
    IN_MIXED_Q = 0,
	IN_MIXED_K,
	IN_COS_EMBED,
	IN_SIN_EMBED,
	IN_SEQ_LEN,
	OUT_EMBED_Q,
	OUT_EMBED_K,
};

static const uint64_t IN_TENSOR_COUNT = 5;
static const uint64_t OUT_TENSOR_COUNT = 2;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 0;
static const uint64_t NODE_COUNT = 1;

atb::Status Rope(const RopeParam& param, atb::Operation** operation)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    auto& ropeNode = opGraph.nodes.at(nodeId++);

    atb::infer::RopeParam ropeParam;
    ropeParam.rotaryCoeff = param.rotaryCoeff;
    CreateOperation(ropeParam, &ropeNode.operation);
    ropeNode.inTensorIds = {IN_MIXED_Q, IN_MIXED_K, IN_COS_EMBED, IN_SIN_EMBED, IN_SEQ_LEN};
    ropeNode.outTensorIds = {OUT_EMBED_Q, OUT_EMBED_K};
    ropeNode.inTensorReshapeFuncs.resize(ropeNode.inTensorIds.size());
        ropeNode.inTensorReshapeFuncs.at(0) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            newShape.dimNum = 2;
            newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
            newShape.dims[1] = oldShape.dims[2];
        };
        ropeNode.inTensorReshapeFuncs.at(1) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            newShape.dimNum = 2;
            newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
            newShape.dims[1] = oldShape.dims[2] * oldShape.dims[3];
        };
        ropeNode.inTensorReshapeFuncs.at(2) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            newShape.dimNum = 2;
            newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
            newShape.dims[1] = oldShape.dims[2];
        };
        ropeNode.inTensorReshapeFuncs.at(3) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            newShape.dimNum = 2;
            newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
            newShape.dims[1] = oldShape.dims[2];
        };

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc>& inTensorDescs,
								atb::SVector<atb::TensorDesc>& outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        outTensorDescs.at(0).shape.dimNum = 4;
        outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
        outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1];
        outTensorDescs.at(0).shape.dims[2] = param.headNum;
        outTensorDescs.at(0).shape.dims[3] = inTensorDescs.at(0).shape.dims[2] / param.headNum;
        outTensorDescs.at(1) = outTensorDescs.at(0);
        return atb::NO_ERROR;
    };

    atb::CreateOperation(opGraph, operation);
    return atb::NO_ERROR;
}
} // namespace telechat
} // namespace atb_speed