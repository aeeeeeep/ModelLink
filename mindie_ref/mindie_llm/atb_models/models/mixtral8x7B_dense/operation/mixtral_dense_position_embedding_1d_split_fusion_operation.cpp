/*
* Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
*
*  * Licensed under the Apache License, Version 2.0 (the "License");
*  * you may not use this file except in compliance with the License.
*  * You may obtain a copy of the License at
*  *
*  * http://www.apache.org/licenses/LICENSE-2.0
*  *
*  * Unless required by applicable law or agreed to in writing, software
*  * distributed under the License is distributed on an "AS IS" BASIS,
*  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  * See the License for the specific language governing permissions and
*  * limitations under the License.
*  */
#include "mixtral_dense_position_embedding_1d_split_fusion_operation.h"
#include <atb/atb_infer.h>

namespace atb_speed {
namespace mixtralDense {
enum MixtralDensePositionEmbedding1DSplitFusionTensorId {
    IN_QLAYERTENSOR = 0,
    IN_KLAYERTENSOR,
    IN_COSTABLETENSOR,
    IN_SINTABLETENSOR,
    IN_SEQLENTENSOR,
    OUT_QEMBEDDEDTENSOR,
    OUT_KEMBEDDEDTENSOR,
};

static const uint64_t IN_TENSOR_COUNT = 5;
static const uint64_t OUT_TENSOR_COUNT = 2;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 0;
static const uint64_t NODE_COUNT = 1;

atb::Status CreateMixtralDensePositionEmbedding1DSplitFusionOperation(const MixtralDensePositionEmbedding1DSplitFusionParam &param,
                                                                      atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.name = "MixtralDenseROPEfusion";
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    atb::Node &ropeNode = opGraph.nodes.at(nodeId++);

    atb::infer::RopeParam ropeParam;
    ropeParam.rotaryCoeff = param.rotaryCoeff;
    CreateOperation(ropeParam, &ropeNode.operation);
    ropeNode.inTensorIds = {IN_QLAYERTENSOR, IN_KLAYERTENSOR, IN_COSTABLETENSOR, IN_SINTABLETENSOR, IN_SEQLENTENSOR};
    ropeNode.outTensorIds = {OUT_QEMBEDDEDTENSOR, OUT_KEMBEDDEDTENSOR};
    ropeNode.inTensorReshapeFuncs.resize(ropeNode.inTensorIds.size());
    ropeNode.inTensorReshapeFuncs[2] = [](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2;
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
        newShape.dims[1] = oldShape.dims[2];
    };
    ropeNode.inTensorReshapeFuncs[3] = [](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2;
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
        newShape.dims[1] = oldShape.dims[2];
    };

    opGraph.inferShapeFunc = [&](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                    atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.resize(2);
        outTensorDescs.at(0) = inTensorDescs.at(0);
        outTensorDescs.at(0).shape.dimNum = 2;
        outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
        outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1];
        outTensorDescs.at(1) = inTensorDescs.at(1);
        outTensorDescs.at(1).shape.dimNum = 2;
        outTensorDescs.at(1).shape.dims[0] = inTensorDescs.at(1).shape.dims[0];
        outTensorDescs.at(1).shape.dims[1] = inTensorDescs.at(1).shape.dims[1];
        return atb::NO_ERROR;
    };

    return atb::CreateOperation(opGraph, operation);
}
}
} // namespace atb_speed