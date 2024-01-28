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
namespace internlm_7b {
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
static const uint64_t ROPE_IN_MIXED_Q_ID = 0;
static const uint64_t ROPE_IN_MIXED_K_ID = 1;
static const uint64_t ROPE_IN_COS_EMBED_ID = 2;
static const uint64_t ROPE_IN_SIN_EMBED_ID = 3;
static const uint64_t OUT_TENSOR_EMBED_Q_ID = 0;
static const uint64_t OUT_TENSOR_EMBED_K_ID = 1;
static const uint64_t OUT_TENSOR_EMBED_Q_SHAPE = 4;

static void mergeBatchNTokens(const atb::Dims &oldShape, atb::Dims &newShape)
{
    size_t newShapeDimNum = 2;
    size_t shapeFirstDim = 0;
    size_t shapeSecondDim = 1;
    size_t shapeThirdDim = 2;
    newShape.dimNum = newShapeDimNum;
    newShape.dims[shapeFirstDim] = oldShape.dims[shapeFirstDim] * oldShape.dims[shapeSecondDim];
    newShape.dims[shapeSecondDim] = oldShape.dims[shapeThirdDim];
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
    CREATE_OPERATION(ropeParam, &ropeNode.operation);
    ropeNode.inTensorIds = {IN_MIXED_Q, IN_MIXED_K, IN_COS_EMBED, IN_SIN_EMBED, IN_SEQ_LEN};
    ropeNode.outTensorIds = {OUT_EMBED_Q, OUT_EMBED_K};
    ropeNode.inTensorReshapeFuncs.resize(ropeNode.inTensorIds.size());
    ropeNode.inTensorReshapeFuncs.at(ROPE_IN_MIXED_Q_ID) = &mergeBatchNTokens;
    ropeNode.inTensorReshapeFuncs.at(ROPE_IN_MIXED_K_ID) = &mergeBatchNTokens;
    ropeNode.inTensorReshapeFuncs.at(ROPE_IN_COS_EMBED_ID) = &mergeBatchNTokens;
    ropeNode.inTensorReshapeFuncs.at(ROPE_IN_SIN_EMBED_ID) = &mergeBatchNTokens;

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        size_t firstDim = 0;
        size_t secondDim = 1;
        size_t headNumDim = 2;
        size_t outTensorEmbedQThirdDim = 3;
        outTensorDescs.at(OUT_TENSOR_EMBED_Q_ID) = inTensorDescs.at(ROPE_IN_MIXED_Q_ID);
        outTensorDescs.at(OUT_TENSOR_EMBED_Q_ID).shape.dimNum = OUT_TENSOR_EMBED_Q_SHAPE;
        outTensorDescs.at(OUT_TENSOR_EMBED_Q_ID).shape.dims[firstDim] =
            inTensorDescs.at(ROPE_IN_MIXED_Q_ID).shape.dims[firstDim];
        outTensorDescs.at(OUT_TENSOR_EMBED_Q_ID).shape.dims[secondDim] =
            inTensorDescs.at(ROPE_IN_MIXED_Q_ID).shape.dims[secondDim];
        outTensorDescs.at(OUT_TENSOR_EMBED_Q_ID).shape.dims[headNumDim] = param.headNum;
        outTensorDescs.at(OUT_TENSOR_EMBED_Q_ID).shape.dims[outTensorEmbedQThirdDim] =
            inTensorDescs.at(ROPE_IN_MIXED_Q_ID).shape.dims[headNumDim] / param.headNum;
        outTensorDescs.at(OUT_TENSOR_EMBED_K_ID) = outTensorDescs.at(OUT_TENSOR_EMBED_Q_ID);
        return atb::NO_ERROR;
    };

    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}

atb::Operation *CreateRope(const nlohmann::json &paramJson)
{
    atb_speed::internlm_7b::RopeParam param;
    param.rotaryCoeff = paramJson["rotaryCoeff"].get<int>();
    param.headNum = paramJson["headNum"].get<int>();
    atb::Operation *op;
    atb_speed::internlm_7b::Rope(param, &op);
    return op;
}

} // namespace internlm_7b
} // namespace atb_speed