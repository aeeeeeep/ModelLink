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
namespace baichuan2_7b {
enum RopeTensorId : int {
    IN_MIXED_Q = 0, // [batch_size,seq_len,hidden_size]
    IN_MIXED_K,
    IN_COS_EMBED, // fp16 当前支持FP16
    IN_SIN_EMBED, // fp16
    IN_SEQ_LEN,
    OUT_EMBED_Q,
    OUT_EMBED_K,
};

static const uint64_t IN_TENSOR_COUNT = 5;
static const uint64_t OUT_TENSOR_COUNT = 2;
static const uint64_t INTERNAL_TENSOR_COUNT = 0;
static const uint64_t NODE_COUNT = 1;

atb::Operation *CreateRope(const nlohmann::json &paramJson)
{
    atb_speed::baichuan2_7b::RopeParam param;
    param.rotaryCoeff = paramJson["rotaryCoeff"].get<int>();
    param.headNum = paramJson["headNum"].get<int>();
    atb::Operation *op;
    atb_speed::baichuan2_7b::Rope(param, &op);
    return op;
}

static void mergeBatchNTokens(const atb::Dims &oldShape, atb::Dims &newShape)
{
    newShape.dimNum = 2;
    newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
    newShape.dims[1] = oldShape.dims[2];
}
atb::Status Rope(const RopeParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERNAL_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);
    opGraph.name = GetFuncNameAndNameSpace(__PRETTY_FUNCTION__);

    size_t nodeId = 0;
    auto &ropeNode = opGraph.nodes.at(nodeId++);

    atb::infer::RopeParam ropeParam;
    ropeParam.rotaryCoeff = param.rotaryCoeff;
    CreateOperation(ropeParam, &ropeNode.operation);
    ropeNode.inTensorIds = {IN_MIXED_Q, IN_MIXED_K, IN_COS_EMBED, IN_SIN_EMBED, IN_SEQ_LEN};
    ropeNode.outTensorIds = {OUT_EMBED_Q, OUT_EMBED_K};
    ropeNode.inTensorReshapeFuncs.resize(ropeNode.inTensorIds.size());
    ropeNode.inTensorReshapeFuncs.at(0) = &mergeBatchNTokens;
    ropeNode.inTensorReshapeFuncs.at(1) = &mergeBatchNTokens;
    ropeNode.inTensorReshapeFuncs.at(2) = &mergeBatchNTokens;
    ropeNode.inTensorReshapeFuncs.at(3) = &mergeBatchNTokens;

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
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
} // namespace baichuan2_7b
} // namespace atb_speed