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
#include "position_embedding.h"

namespace atb_speed {
namespace glm130b {
enum PositionEmbeddingTensorId : int {
    IN_MIXEDQKV_ID = 0,
    IN_COS_TABLE_ID,
    IN_SIN_TABLE_ID,
    IN_SEQLEN_ID,
    OUT_QEMBEDDED_ID,
    OUT_KEMBEDDED_ID,
    OUT_VALUE,
    INTERMEDIATE_QLAYER_ID,
    INTERMEDIATE_KLAYER_ID
};

static const uint64_t IN_TENSOR_COUNT = 4;
static const uint64_t OUT_TENSOR_COUNT = 3;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 2;
static const uint64_t NODE_COUNT = 2;
static const uint64_t DIM_NUM_2 = 2;

void CosSinReshapeFunc(const atb::Dims &oldShape, atb::Dims &newShape)
{
    newShape.dimNum = oldShape.dimNum - 1;
    newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1]; // batch_size * seq_len
    newShape.dims[1] = oldShape.dims[2];                    // 2: 设置新张量第二维的长度
}

void QKReshapeFunc(const atb::Dims &oldShape, atb::Dims &newShape)
{
    newShape.dimNum = oldShape.dimNum - DIM_NUM_2;
    newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1]; // seq_len * batch_size
    newShape.dims[1] =
        oldShape.dims[2] * oldShape.dims[3]; // 2, 3: 设置新张量第二维的长度,(head_num / world_size) * head_size
}

atb::Status CreatePositionEmbedding(const PositionEmbeddingParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    atb::Node &splitQKVNode = opGraph.nodes.at(nodeId++);
    atb::Node &ropeNode = opGraph.nodes.at(nodeId++);

    atb::infer::SplitParam splitQKVParam;
    splitQKVParam.splitDim = 3; // 3: 在第四维上进行切分
    splitQKVParam.splitNum = 3; // 3: 进行三等分
    CreateOperation(splitQKVParam, &splitQKVNode.operation);
    splitQKVNode.inTensorIds = {IN_MIXEDQKV_ID}; // [bs, seq_len, 3 * hidden_size / world_size]
    splitQKVNode.outTensorIds = {INTERMEDIATE_QLAYER_ID, INTERMEDIATE_KLAYER_ID,
                                 OUT_VALUE}; // [bs, seq_len, hidden_size / world_size]
    splitQKVNode.inTensorReshapeFuncs.resize(splitQKVNode.inTensorIds.size());
    splitQKVNode.inTensorReshapeFuncs.at(0) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = oldShape.dimNum + 1;
        newShape.dims[0] = oldShape.dims[0]; // bs
        newShape.dims[1] = oldShape.dims[1]; // seq_len
        newShape.dims[2] = param.headNum;    // 2: 设置张量第三维的大小, headNum = headNum / rankSize
        newShape.dims[3] = oldShape.dims[2] / param.headNum; // 2, 3: 设置张量第四维的大小
    };

    atb::infer::RopeParam ropeParam;
    ropeParam.rotaryCoeff = 2; // 2: 设置旋转系数
    CreateOperation(ropeParam, &ropeNode.operation);
    ropeNode.inTensorIds = {INTERMEDIATE_QLAYER_ID, INTERMEDIATE_KLAYER_ID, IN_COS_TABLE_ID, IN_SIN_TABLE_ID,
                            IN_SEQLEN_ID}; // IN_COS_TABLE_ID: [bs, seq_len, max_seq_len]
    ropeNode.outTensorIds = {OUT_QEMBEDDED_ID, OUT_KEMBEDDED_ID};
    ropeNode.inTensorReshapeFuncs = {&QKReshapeFunc, &QKReshapeFunc, &CosSinReshapeFunc, &CosSinReshapeFunc};

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        outTensorDescs.at(0).shape.dimNum = 4;                                  // 4表示输出维度
        outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0]; // bs
        outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1]; // 1: 设置第一个张量第二维长度, seq_len
        outTensorDescs.at(0).shape.dims[2] = param.headNum;        // 2: 设置第一个张量第三维长度, headNum
        outTensorDescs.at(0).shape.dims[3] =                       // 3: 设置第一个张量第四维长度
            inTensorDescs.at(0).shape.dims[2] / param.headNum / 3; // 2, 3: 设置张量第四维长度
        outTensorDescs.at(1) = outTensorDescs.at(0);               // 1: 设置第二个张量的描述
        outTensorDescs.at(2) = outTensorDescs.at(0);               // 2: 设置第三个张量的描述
        return atb::NO_ERROR;
    };

    atb::CreateOperation(opGraph, operation);
    return atb::NO_ERROR;
}
} // namespace glm130b
} // namespace atb_speed