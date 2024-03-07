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

#define POS_EMB_CAST(x) static_cast<int>(PositionEmbeddingTensorId::x)

#include "attention.h"

namespace atb_speed {
namespace common {

int g_headNum = 0;
int g_hiddenSizePerHead = 0;
int g_kvHeadNum = 0;

static const uint64_t DIM_0 = 0;
static const uint64_t DIM_1 = 1;
static const uint64_t DIM_2 = 2;
static const uint64_t DIM_3 = 3;
static const int64_t DIM_LAST = -1;
static const uint64_t DIM_NUM_1 = 1;
static const uint64_t DIM_NUM_2 = 2;
static const uint64_t DIM_NUM_3 = 3;
static const uint64_t DIM_NUM_4 = 4;
static const uint64_t SPLIT_NUM_2 = 2;
static const uint64_t SPLIT_NUM_3 = 3;

static const uint64_t IN_TENSOR_COUNT = 22;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT_GQA = 8;
static const uint64_t INTERMEDIATE_TENSOR_COUNT_MHA = 7;
static const uint64_t NODE_COUNT_BASE = 3;
static const uint64_t NODE_COUNT_ROPE = 1;
static const uint64_t NODE_COUNT_NO_ROPE = 0;
static const uint64_t NODE_COUNT_SPLIT_GQA = 3;
static const uint64_t NODE_COUNT_SPLIT_MHA = 1;

void squeezeRopeIntensor(const atb::Dims &oldShape, atb::Dims &newShape);

void unsqueezeByHeadNum(const atb::Dims &oldShape, atb::Dims &newShape);

void unsqueezeByKVHeadNum(const atb::Dims &oldShape, atb::Dims &newShape);

void unsqueezeMixedQKVByHeadNum(const atb::Dims &oldShape, atb::Dims &newShape);

atb::Status FlashAttentionWithPosEmbedding::FlashAttentionWithPositionEmbeddingLayer(const FTWithROPEParam &param,
                                                                                     atb::Operation **operation)
{
    g_headNum = param.selfAttentionKvCacheParam.headNum;
    g_hiddenSizePerHead = param.selfAttentionKvCacheParam.headDim;
    g_kvHeadNum = param.selfAttentionKvCacheParam.kvHeadNum;

    atb::GraphParam opGraph;
    opGraph.name = "FAWithPositionEmbedding";
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = param.isGroupedQueryAttention ?
                                    INTERMEDIATE_TENSOR_COUNT_GQA : INTERMEDIATE_TENSOR_COUNT_MHA;
    size_t posEmbNodeCount = param.selfAttentionKvCacheParam.isSupportAlibi ? NODE_COUNT_NO_ROPE : NODE_COUNT_ROPE;
    size_t splitNodeCount = param.isGroupedQueryAttention ? NODE_COUNT_SPLIT_GQA : NODE_COUNT_SPLIT_MHA;
    size_t nodeCount = NODE_COUNT_BASE + posEmbNodeCount + splitNodeCount;
    opGraph.nodes.resize(nodeCount);

    size_t nodeId = 0;

    atb::Node &mixedQKVLinearNode = opGraph.nodes.at(nodeId++);
    atb_speed::common::RowParallelLinearV2(param.mixdQkvLinearParam, &mixedQKVLinearNode.operation);
    mixedQKVLinearNode.inTensorIds = {IN_HIDDENSTATES, IN_WEIGHT_MIXEDQKV, IN_BIAS_MIXEDQKV,
                                      IN_DEQSCALE_MIXEDQKV, IN_QKVMIXDWEIGHT_INDEX, IN_QKVMIXDOFFSETX,
                                      IN_QKVMIXDWEIGHT_COMPRESSINFO};
    mixedQKVLinearNode.outTensorIds = {INTERMEDIATE_MIXED_QKV};

    // split mixedQKV
    if (param.isGroupedQueryAttention) {
        auto &sliceQNode = opGraph.nodes[nodeId++];
        atb::infer::SliceParam sliceQNodeParam;
        sliceQNodeParam.offsets = {0, 0, 0};
        sliceQNodeParam.size = {-1, -1, g_headNum * g_hiddenSizePerHead};
        CREATE_OPERATION(sliceQNodeParam, &sliceQNode.operation);
        sliceQNode.inTensorIds = {INTERMEDIATE_MIXED_QKV};
        sliceQNode.outTensorIds = {INTERMEDIATE_QUERY};

        auto &sliceKVNode = opGraph.nodes[nodeId++];
        atb::infer::SliceParam sliceKVNodeParam;
        sliceKVNodeParam.offsets = {0, 0, g_headNum * g_hiddenSizePerHead};
        sliceKVNodeParam.size = {-1, -1, g_kvHeadNum * g_hiddenSizePerHead * 2};
        CREATE_OPERATION(sliceKVNodeParam, &sliceKVNode.operation);
        sliceKVNode.inTensorIds = {INTERMEDIATE_MIXED_QKV};
        sliceKVNode.outTensorIds = {INTERMEDIATE_KV};

        auto &splitKVNode = opGraph.nodes[nodeId++];
        atb::infer::SplitParam splitKVParam;
        splitKVParam.splitDim = DIM_LAST;
        splitKVParam.splitNum = SPLIT_NUM_2;
        CREATE_OPERATION(splitKVParam, &splitKVNode.operation);
        splitKVNode.inTensorIds = {INTERMEDIATE_KV};
        splitKVNode.outTensorIds = {INTERMEDIATE_KEY, INTERMEDIATE_VALUE};
    } else {
        auto &splitMixedQKVNode = opGraph.nodes[nodeId++];
        atb::infer::SplitParam splitMixedQKVParam;
        splitMixedQKVParam.splitDim = DIM_LAST;
        splitMixedQKVParam.splitNum = SPLIT_NUM_3;
        CREATE_OPERATION(splitMixedQKVParam, &splitMixedQKVNode.operation);
        splitMixedQKVNode.inTensorIds = {INTERMEDIATE_MIXED_QKV};
        splitMixedQKVNode.outTensorIds = {INTERMEDIATE_QUERY, INTERMEDIATE_KEY, INTERMEDIATE_VALUE};
        if (param.isCrossedWeight) {
            splitMixedQKVNode.inTensorReshapeFuncs = {&unsqueezeMixedQKVByHeadNum};
        }
    }

    if (!param.selfAttentionKvCacheParam.isSupportAlibi) {
        atb::Node &positionEmbeddingNode = opGraph.nodes.at(nodeId++);
        PositionEmbedding(param, &positionEmbeddingNode.operation);
        positionEmbeddingNode.inTensorIds = {INTERMEDIATE_QUERY, INTERMEDIATE_KEY, IN_ROPE_COS, IN_ROPE_SIN, IN_SEQLEN};
        positionEmbeddingNode.outTensorIds = {INTERMEDIATE_POSITIONEMBED_Q, INTERMEDIATE_POSITIONEMBED_K};
    }

    atb::Node &selfAttentionNode = opGraph.nodes.at(nodeId++);
    CREATE_OPERATION(param.selfAttentionKvCacheParam, &selfAttentionNode.operation);
    selfAttentionNode.inTensorIds = {INTERMEDIATE_POSITIONEMBED_Q,
                                     INTERMEDIATE_POSITIONEMBED_K,
                                     INTERMEDIATE_VALUE,
                                     IN_CACHED_K,
                                     IN_CACHED_V,
                                     IN_ATTENTION_MASK,
                                     IN_TOKEN_OFFSET,
                                     IN_SEQLEN,
                                     IN_LAYER_ID};
    selfAttentionNode.outTensorIds = {INTERMEDIATE_SELFOUT};
    if (param.isGroupedQueryAttention) {
        selfAttentionNode.inTensorReshapeFuncs = {&unsqueezeByHeadNum, &unsqueezeByKVHeadNum, &unsqueezeByKVHeadNum};
    } else {
        selfAttentionNode.inTensorReshapeFuncs = {&unsqueezeByHeadNum, &unsqueezeByHeadNum, &unsqueezeByHeadNum};
    }

    atb::Node &selfOutLinearNode = opGraph.nodes.at(nodeId++);
    atb_speed::common::RowParallelLinearV2(param.selfOutLinearParam, &selfOutLinearNode.operation);
    selfOutLinearNode.inTensorIds = {INTERMEDIATE_SELFOUT, IN_WEIGHT_SELFOUT, IN_BIAS_SELFOUT, IN_DEQSCALE_SELFOUT,
                                     IN_SELFOUTLINEARWEIGHT_INDEX, IN_SELFOUTLINEAROFFSETX,
                                     IN_SELFOUTLINEARWEIGHT_COMPRESSINFO};
    selfOutLinearNode.outTensorIds = {OUT_RESULT_ID};

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        outTensorDescs.at(0).dtype = ACL_FLOAT16;
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

atb::Status FlashAttentionWithPosEmbedding::PositionEmbedding(const FTWithROPEParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.name = "PositionEmbedding";
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
        splitQParam.splitDim = DIM_3;
        splitQParam.splitNum = SPLIT_NUM_2;
        CREATE_OPERATION(splitQParam, &splitQNode.operation);
        splitQNode.inTensorIds = {POS_EMB_CAST(IN_QUERY)};
        splitQNode.outTensorIds = {POS_EMB_CAST(INTERMEDIATE_QCHUNK0), POS_EMB_CAST(INTERMEDIATE_QCHUNK1)};
        splitQNode.inTensorReshapeFuncs = {&unsqueezeByHeadNum};

        auto &splitKNode = opGraph.nodes[nodeId++];
        atb::infer::SplitParam splitKParam;
        splitKParam.splitDim = DIM_3;
        splitKParam.splitNum = SPLIT_NUM_2;
        CREATE_OPERATION(splitKParam, &splitKNode.operation);
        splitKNode.inTensorIds = {POS_EMB_CAST(IN_KEY)};
        splitKNode.outTensorIds = {POS_EMB_CAST(INTERMEDIATE_KCHUNK0), POS_EMB_CAST(INTERMEDIATE_KCHUNK1)};
        splitKNode.inTensorReshapeFuncs = {&unsqueezeByKVHeadNum};

        auto &ropeNode = opGraph.nodes[nodeId++];
        atb::infer::RopeParam ropeParam;
        ropeParam.rotaryCoeff = param.rotaryCoeff;
        CREATE_OPERATION(ropeParam, &ropeNode.operation);
        ropeNode.inTensorIds = {POS_EMB_CAST(INTERMEDIATE_QCHUNK0), POS_EMB_CAST(INTERMEDIATE_KCHUNK0),
                                POS_EMB_CAST(IN_ROPE_COS), POS_EMB_CAST(IN_ROPE_SIN), POS_EMB_CAST(IN_SEQLEN)};
        ropeNode.outTensorIds = {POS_EMB_CAST(INTERMEDIATE_QOUT), POS_EMB_CAST(INTERMEDIATE_KOUT)};
        ropeNode.inTensorReshapeFuncs.resize(ropeNode.inTensorIds.size());
        ropeNode.inTensorReshapeFuncs.at(DIM_2) = &squeezeRopeIntensor;
        ropeNode.inTensorReshapeFuncs.at(DIM_3) = &squeezeRopeIntensor;

        auto &cat1Node = opGraph.nodes[nodeId++];
        atb::infer::ConcatParam cat1Param;
        cat1Param.concatDim = DIM_LAST;
        CREATE_OPERATION(cat1Param, &cat1Node.operation);
        cat1Node.inTensorIds = {POS_EMB_CAST(INTERMEDIATE_QOUT), POS_EMB_CAST(INTERMEDIATE_QCHUNK1)};
        cat1Node.outTensorIds = {POS_EMB_CAST(OUT_QUERY)};

        auto &cat2Node = opGraph.nodes[nodeId++];
        atb::infer::ConcatParam cat2Param;
        cat2Param.concatDim = DIM_LAST;
        CREATE_OPERATION(cat2Param, &cat2Node.operation);
        cat2Node.inTensorIds = {POS_EMB_CAST(INTERMEDIATE_KOUT), POS_EMB_CAST(INTERMEDIATE_KCHUNK1)};
        cat2Node.outTensorIds = {POS_EMB_CAST(OUT_KEY)};
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

    opGraph.inferShapeFunc = []
                (const atb::SVector<atb::TensorDesc> &inTensorDescs, atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        outTensorDescs.at(0).shape.dimNum = DIM_NUM_4;
        outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
        outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1];
        outTensorDescs.at(0).shape.dims[DIM_2] = g_headNum;
        outTensorDescs.at(0).shape.dims[DIM_3] = g_hiddenSizePerHead;
        outTensorDescs.at(1) = inTensorDescs.at(1);
        outTensorDescs.at(1).shape.dimNum = DIM_NUM_4;
        outTensorDescs.at(1).shape.dims[0] = inTensorDescs.at(1).shape.dims[0];
        outTensorDescs.at(1).shape.dims[1] = inTensorDescs.at(1).shape.dims[1];
        outTensorDescs.at(1).shape.dims[DIM_2] = g_kvHeadNum;
        outTensorDescs.at(1).shape.dims[DIM_3] = g_hiddenSizePerHead;
        return atb::NO_ERROR;
    };

    CREATE_OPERATION(opGraph, operation);

    return atb::NO_ERROR;
}

void squeezeRopeIntensor(const atb::Dims &oldShape, atb::Dims &newShape)
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

void unsqueezeByHeadNum(const atb::Dims &oldShape, atb::Dims &newShape)
{
    newShape.dimNum = DIM_NUM_4;
    newShape.dims[0] = oldShape.dims[0];
    newShape.dims[1] = oldShape.dims[1];
    newShape.dims[DIM_2] = g_headNum;
    newShape.dims[DIM_3] = g_hiddenSizePerHead;
}

void unsqueezeByKVHeadNum(const atb::Dims &oldShape, atb::Dims &newShape)
{
    newShape.dimNum = DIM_NUM_4;
    newShape.dims[0] = oldShape.dims[0];
    newShape.dims[1] = oldShape.dims[1];
    newShape.dims[DIM_2] = g_kvHeadNum;
    newShape.dims[DIM_3] = g_hiddenSizePerHead;
}

void unsqueezeMixedQKVByHeadNum(const atb::Dims &oldShape, atb::Dims &newShape)
{
    newShape.dimNum = DIM_NUM_4;
    newShape.dims[0] = oldShape.dims[0];
    newShape.dims[1] = oldShape.dims[1];
    newShape.dims[DIM_2] = g_headNum;
    newShape.dims[DIM_3] = g_hiddenSizePerHead * SPLIT_NUM_3;
}

} // namespace common
} // namespace atb_speed