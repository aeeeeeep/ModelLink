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
#include "rope_fusion_operation.h"

namespace atb_speed {
namespace internlm_20b {
enum RopeFusionTensorId {
    IN_QLAYER_ID = 0,    // [batch, seqLen, hiddenSize], half
    IN_KLAYER_ID,        // [batch, seqLen, hiddenSize], half
    IN_POSITION_IDS_ID,  // [batch, seqLen], int64
    IN_COS_TABLE_ID,     // [1, 1, maxseqLen, headDim], half
    IN_SIN_TABLE_ID,     // [1, 1, maxseqLen, headDim], half
    IN_SEQLEN_ID,        // [batch], int32
    OUT_QEMBEDDED_ID,    // [batch, seqlen, headNum, headDim], half
    OUT_KEMBEDDED_ID,    // [batch, seqlen, headNum, headDim], half
};

static const uint64_t IN_TENSOR_COUNT = 6;
static const uint64_t OUT_TENSOR_COUNT = 2;
static const uint64_t NODE_COUNT = 1;

void RopeReshapeFunc(const atb::Dims &oldShape, atb::Dims &newShape)
{
    newShape.dimNum = oldShape.dimNum - 1;
    newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
    newShape.dims[1] = oldShape.dims[2]; // 2: 设置新张量第二维的长度
}

atb::Status RopeFusionOperation(const RopeFusionParam &param, atb::Operation **operation)
{
    ATB_LOG(INFO) << __func__ << ", headNum: " << param.headNum;
    atb::GraphParam opGraph;
    opGraph.name = "RopeFusionOperation";
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    auto &ropeNode = opGraph.nodes.at(nodeId++);

    atb::infer::RopeParam ropeParam;
    ropeParam.rotaryCoeff = 2; // 设置旋转系数
    CREATE_OPERATION(ropeParam, &ropeNode.operation);
    ropeNode.inTensorIds = {IN_QLAYER_ID, IN_KLAYER_ID, IN_COS_TABLE_ID, IN_SIN_TABLE_ID, IN_SEQLEN_ID};
    ropeNode.outTensorIds = {OUT_QEMBEDDED_ID, OUT_KEMBEDDED_ID};
    ropeNode.inTensorReshapeFuncs = {&RopeReshapeFunc, &RopeReshapeFunc, &RopeReshapeFunc, &RopeReshapeFunc};

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
        atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        outTensorDescs.at(0).shape.dimNum = 4; // 表示输出维度
        outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
        outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1];
        outTensorDescs.at(0).shape.dims[2] = param.headNum;    // 1: 设置张量第二维长度
        outTensorDescs.at(0).shape.dims[3] =                   // 2: 设置张量第三维长度
            inTensorDescs.at(0).shape.dims[2] / param.headNum; // 3: 设置张量第四维长度
        outTensorDescs.at(1) = outTensorDescs.at(0);
        return atb::NO_ERROR;
    };

    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}
} // namespace internlm_20b
} // namespace atb_speed