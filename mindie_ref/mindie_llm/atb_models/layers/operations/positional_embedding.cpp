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
#include "layers/operations/positional_embedding.h"

namespace atb_speed {
namespace common {

enum PositionalEmbeddingGatherTensorIdx : uint32_t {
    IN_POSITION_IDS = 0,
    IN_COS_TABLE,
    IN_SIN_TABLE,
    OUT_COS_EMBEDDING,
    OUT_SIN_EMBEDDING,
};

static const uint64_t IN_TENSOR_COUNT = 3;
static const uint64_t OUT_TENSOR_COUNT = 2;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 0;
static const uint64_t NODE_COUNT = 2;

atb::Status PositionalEmbeddingGather(atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);
    opGraph.name = "PositionalEmbeddingGather";

    size_t nodeId = 0;

    auto &cosEmbeddingNode = opGraph.nodes.at(nodeId++);
    atb::infer::GatherParam cosEmbeddingGatherParam;
    CREATE_OPERATION(cosEmbeddingGatherParam, &cosEmbeddingNode.operation);
    cosEmbeddingNode.inTensorIds = {
        PositionalEmbeddingGatherTensorIdx::IN_COS_TABLE, PositionalEmbeddingGatherTensorIdx::IN_POSITION_IDS
    };
    cosEmbeddingNode.outTensorIds = {PositionalEmbeddingGatherTensorIdx::OUT_COS_EMBEDDING};

    auto &sinEmbeddingNode = opGraph.nodes.at(nodeId++);
    atb::infer::GatherParam sinEmbeddingGatherParam;
    CREATE_OPERATION(sinEmbeddingGatherParam, &sinEmbeddingNode.operation);
    sinEmbeddingNode.inTensorIds = {
        PositionalEmbeddingGatherTensorIdx::IN_SIN_TABLE, PositionalEmbeddingGatherTensorIdx::IN_POSITION_IDS
    };
    sinEmbeddingNode.outTensorIds = {PositionalEmbeddingGatherTensorIdx::OUT_SIN_EMBEDDING};

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(1);
        outTensorDescs.at(0).shape.dimNum = 2;
        outTensorDescs.at(0).shape.dims[0] = 1;
        // unpadInputs=True场景下，for loop只循环一次；unpadInputs=False场景下，for loop循环两次，将bsz和seqLen合轴
        for (uint64_t i = 0; i < inTensorDescs.at(0).shape.dimNum; i++) {
            outTensorDescs.at(0).shape.dims[0] = outTensorDescs.at(0).shape.dims[0] * inTensorDescs.at(0).shape.dims[i];
        }

        outTensorDescs.at(1) = outTensorDescs.at(0);
        return atb::NO_ERROR;
    };

    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}
static const uint64_t POS_EMB_IN_TENSOR_COUNT = 5;
static const uint64_t POS_EMB_OUT_TENSOR_COUNT = 2;
static const uint64_t POS_EMB_INTERMEDIATE_TENSOR_2D_COUNT = 6;
static const uint64_t POS_EMB_INTERMEDIATE_TENSOR_1D_COUNT = 0;
static const uint64_t POS_EMB_NODE_2D_COUNT = 5;
static const uint64_t POS_EMB_NODE_1D_COUNT = 1;

static const uint64_t DIM_NUM_1 = 1;
static const uint64_t DIM_NUM_2 = 2;
static const uint64_t DIM_NUM_3 = 3;
static const uint64_t DIM_NUM_4 = 4;
static const int64_t DIM_LAST = -1;
static const uint64_t DIM_0 = 0;
static const uint64_t DIM_1 = 1;
static const uint64_t DIM_2 = 2;
static const uint64_t DIM_3 = 3;
static const uint64_t SPLIT_NUM_2 = 2;
static const uint64_t SPLIT_NUM_3 = 3;

void unsqueezeDim4ByHeadNum(const atb::Dims &oldShape, atb::Dims &newShape, int headNum, int headDim)
{
    newShape.dimNum = 4;
    newShape.dims[0] = oldShape.dims[0];       // seqLen
    newShape.dims[1] = oldShape.dims[1];
    newShape.dims[2] = headNum;                // numAttentionHeadsPerRank or numKeyValueHeadsPerRank
    newShape.dims[3] = headDim;                // hiddenSizePerAttentionHead
}

void unsqueezeDim2ByHeadNum(const atb::Dims &oldShape, atb::Dims &newShape, int headNum)
{
    newShape.dimNum = 3; // dimNum: 3
    newShape.dims[0] = oldShape.dims[0]; // 0 dim: n tokens
    newShape.dims[1] = headNum; // 1 dim: head num
    if (headNum != 0) {
        newShape.dims[2] = oldShape.dims[1] / headNum; // 1 dim: head size
    } else {
        ATB_LOG(ERROR) << "in unsqueezeDim2ByHeadNum: headNum == 0 be divided!";
    }
}

void SqueezeLinearReshapeFuncPA(const atb::Dims &oldShape, atb::Dims &newShape)
{
    newShape.dimNum = 2;
    newShape.dims[0] = oldShape.dims[0];
    newShape.dims[1] = oldShape.dims[1] * oldShape.dims[2];
}

static void squeezeRopeIntensor(const atb::Dims &oldShape, atb::Dims &newShape)
{
    if (oldShape.dimNum == DIM_NUM_4) {
        newShape.dimNum = DIM_NUM_2;
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
        newShape.dims[1] = oldShape.dims[DIM_2] * oldShape.dims[DIM_3];
    } else if (oldShape.dimNum == DIM_NUM_3) {
        newShape.dimNum = DIM_NUM_2;
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
        newShape.dims[1] = oldShape.dims[DIM_2];
    } else {
        newShape = oldShape;
    }
}
enum class RotaryPositionEmbeddingTensorId : int {
    IN_QUERY = 0,
    IN_KEY,
    IN_ROPE_COS,
    IN_ROPE_SIN,
    IN_SEQLEN,

    OUT_QUERY,
    OUT_KEY,

    INTERMEDIATE_QCHUNK0,
    INTERMEDIATE_QCHUNK1,
    INTERMEDIATE_KCHUNK0,
    INTERMEDIATE_KCHUNK1,
    INTERMEDIATE_QOUT,
    INTERMEDIATE_KOUT,
};
#define POS_EMB_CAST(x) static_cast<int>(RotaryPositionEmbeddingTensorId::x)
atb::Status RotaryPositionEmbedding(const RotaryPositionEmbeddingParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.name = "RotaryPositionEmbedding";
    opGraph.inTensorNum = POS_EMB_IN_TENSOR_COUNT;
    opGraph.outTensorNum = POS_EMB_OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = param.isHalfRotary ?
                                     POS_EMB_INTERMEDIATE_TENSOR_2D_COUNT : POS_EMB_INTERMEDIATE_TENSOR_1D_COUNT;
    int nodeCount = param.isHalfRotary ?
                         POS_EMB_NODE_2D_COUNT : POS_EMB_NODE_1D_COUNT;
    opGraph.nodes.resize(nodeCount);

    size_t nodeId = 0;

    if (param.isHalfRotary) {
        // split q and k to half
        auto &splitQNode = opGraph.nodes[nodeId++];
        atb::infer::SplitParam splitQParam;
        splitQParam.splitDim = DIM_LAST;
        splitQParam.splitNum = SPLIT_NUM_2;
        CREATE_OPERATION(splitQParam, &splitQNode.operation);
        splitQNode.inTensorIds = {POS_EMB_CAST(IN_QUERY)};
        splitQNode.outTensorIds = {POS_EMB_CAST(INTERMEDIATE_QCHUNK0), POS_EMB_CAST(INTERMEDIATE_QCHUNK1)};
        if (param.isFA) {
            splitQNode.inTensorReshapeFuncs = {[=](const atb::Dims &oldShape, atb::Dims &newShape) {
                unsqueezeDim4ByHeadNum(oldShape, newShape,
                    param.headNum, param.headDim);
                }
            };
        } else {
            splitQNode.inTensorReshapeFuncs = {[=](const atb::Dims &oldShape, atb::Dims &newShape) {
                unsqueezeDim2ByHeadNum(oldShape, newShape,
                    param.headNum);
                }
            };
        }

        auto &splitKNode = opGraph.nodes[nodeId++];
        atb::infer::SplitParam splitKParam;
        splitKParam.splitDim = DIM_LAST;
        splitKParam.splitNum = SPLIT_NUM_2;
        CREATE_OPERATION(splitKParam, &splitKNode.operation);
        splitKNode.inTensorIds = {POS_EMB_CAST(IN_KEY)};
        splitKNode.outTensorIds = {POS_EMB_CAST(INTERMEDIATE_KCHUNK0), POS_EMB_CAST(INTERMEDIATE_KCHUNK1)};
        if (param.isFA) {
            splitKNode.inTensorReshapeFuncs = {[=](const atb::Dims &oldShape, atb::Dims &newShape) {
                unsqueezeDim4ByHeadNum(oldShape, newShape,
                    param.headNum, param.headDim);
                }
            };
        } else {
            splitKNode.inTensorReshapeFuncs = {[=](const atb::Dims &oldShape, atb::Dims &newShape) {
                unsqueezeDim2ByHeadNum(oldShape, newShape,
                    param.kvHeadNum);
                }
            };
        }

        auto &ropeNode = opGraph.nodes[nodeId++];
        atb::infer::RopeParam ropeParam;
        ropeParam.rotaryCoeff = param.rotaryCoeff;
        CREATE_OPERATION(ropeParam, &ropeNode.operation);
        ropeNode.inTensorIds = {POS_EMB_CAST(INTERMEDIATE_QCHUNK0), POS_EMB_CAST(INTERMEDIATE_KCHUNK0),
                                POS_EMB_CAST(IN_ROPE_COS), POS_EMB_CAST(IN_ROPE_SIN),
                                POS_EMB_CAST(IN_SEQLEN)};
        ropeNode.outTensorIds = {POS_EMB_CAST(INTERMEDIATE_QOUT), POS_EMB_CAST(INTERMEDIATE_KOUT)};
        ropeNode.inTensorReshapeFuncs.resize(ropeNode.inTensorIds.size());
        if (param.isFA) {
            ropeNode.inTensorReshapeFuncs.at(DIM_2) = &squeezeRopeIntensor;
            ropeNode.inTensorReshapeFuncs.at(DIM_3) = &squeezeRopeIntensor;
        } else {
            ropeNode.inTensorReshapeFuncs.at(0) = &SqueezeLinearReshapeFuncPA;
            ropeNode.inTensorReshapeFuncs.at(1) = &SqueezeLinearReshapeFuncPA;
            ropeNode.inTensorReshapeFuncs.at(2) = &SqueezeLinearReshapeFuncPA; // reshape No.2 input
            ropeNode.inTensorReshapeFuncs.at(3) = &SqueezeLinearReshapeFuncPA; // reshape No.3 input
        }

        auto &cat1Node = opGraph.nodes[nodeId++];
        atb::infer::ConcatParam cat1Param;
        cat1Param.concatDim = DIM_LAST;
        CREATE_OPERATION(cat1Param, &cat1Node.operation);
        cat1Node.inTensorIds = {POS_EMB_CAST(INTERMEDIATE_QOUT), POS_EMB_CAST(INTERMEDIATE_QCHUNK1)};
        cat1Node.outTensorIds = {POS_EMB_CAST(OUT_QUERY)};
        if (!param.isFA) {
            cat1Node.inTensorReshapeFuncs.resize(cat1Node.inTensorIds.size());
            cat1Node.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
                unsqueezeDim2ByHeadNum(oldShape, newShape, param.headNum);
            };
        }

        auto &cat2Node = opGraph.nodes[nodeId++];
        atb::infer::ConcatParam cat2Param;
        cat2Param.concatDim = DIM_LAST;
        CREATE_OPERATION(cat2Param, &cat2Node.operation);
        cat2Node.inTensorIds = {POS_EMB_CAST(INTERMEDIATE_KOUT), POS_EMB_CAST(INTERMEDIATE_KCHUNK1)};
        cat2Node.outTensorIds = {POS_EMB_CAST(OUT_KEY)};
        if (!param.isFA) {
            cat2Node.inTensorReshapeFuncs.resize(cat2Node.inTensorIds.size());
            cat2Node.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
                unsqueezeDim2ByHeadNum(oldShape, newShape, param.kvHeadNum);
            };
        }
    } else {
        auto &ropeNode = opGraph.nodes[nodeId++];
        atb::infer::RopeParam ropeParam;
        ropeParam.rotaryCoeff = param.rotaryCoeff; // 设置旋转系数
        CREATE_OPERATION(ropeParam, &ropeNode.operation);
        ropeNode.inTensorIds = {POS_EMB_CAST(IN_QUERY), POS_EMB_CAST(IN_KEY), POS_EMB_CAST(IN_ROPE_COS),
                                POS_EMB_CAST(IN_ROPE_SIN), POS_EMB_CAST(IN_SEQLEN)};
        ropeNode.outTensorIds = {POS_EMB_CAST(OUT_QUERY), POS_EMB_CAST(OUT_KEY)};
        ropeNode.inTensorReshapeFuncs = {&squeezeRopeIntensor, &squeezeRopeIntensor, &squeezeRopeIntensor,
                                         &squeezeRopeIntensor};
    }
    if (param.isFA) {
        opGraph.inferShapeFunc = [=]
                    (const atb::SVector<atb::TensorDesc> &inTensorDescs, atb::SVector<atb::TensorDesc> &outTensorDescs) {
            outTensorDescs.at(0) = inTensorDescs.at(0);
            outTensorDescs.at(0).shape.dimNum = DIM_NUM_4;
            outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
            outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1];
            outTensorDescs.at(0).shape.dims[DIM_2] = param.headNum;
            outTensorDescs.at(0).shape.dims[DIM_3] = param.headDim;
            outTensorDescs.at(1) = inTensorDescs.at(1);
            outTensorDescs.at(1).shape.dimNum = DIM_NUM_4;
            outTensorDescs.at(1).shape.dims[0] = inTensorDescs.at(1).shape.dims[0];
            outTensorDescs.at(1).shape.dims[1] = inTensorDescs.at(1).shape.dims[1];
            outTensorDescs.at(1).shape.dims[DIM_2] = param.kvHeadNum;
            outTensorDescs.at(1).shape.dims[DIM_3] = param.headDim;
            return atb::NO_ERROR;
        };
    } else {
        opGraph.inferShapeFunc = [=]
                    (const atb::SVector<atb::TensorDesc> &inTensorDescs, atb::SVector<atb::TensorDesc> &outTensorDescs) {
            outTensorDescs.at(0) = inTensorDescs.at(0);
            outTensorDescs.at(0).shape.dimNum = DIM_NUM_3;
            outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
            outTensorDescs.at(0).shape.dims[DIM_1] = param.headNum;
            outTensorDescs.at(0).shape.dims[DIM_2] = param.headDim;
            outTensorDescs.at(1) = inTensorDescs.at(1);
            outTensorDescs.at(1).shape.dimNum = DIM_NUM_3;
            outTensorDescs.at(1).shape.dims[0] = inTensorDescs.at(1).shape.dims[0];
            outTensorDescs.at(1).shape.dims[DIM_1] = param.kvHeadNum;
            outTensorDescs.at(1).shape.dims[DIM_2] = param.headDim;
            return atb::NO_ERROR;
        };
    }

    CREATE_OPERATION(opGraph, operation);

    return atb::NO_ERROR;
}

}  // namespace common
}  // namespace atb_speed