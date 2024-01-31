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

#include "models/llama_parallel/operation/linear.h"
#include "models/llama_parallel/operation/linear_parallel.h"
#include "models/llama_parallel/operation/attention.h"

namespace atb_speed {
namespace llama_parallel {

enum QKVLinearSplitTensorIdx : uint32_t {
    IN_QKV_INPUT = 0,
    IN_QKV_WEIGHT_0,
    IN_QKV_SCALE_0,
    IN_QKV_OFFSET_0,
    IN_QKV_DESCALE_0,
    IN_QKV_WEIGHT_1,
    IN_QKV_SCALE_1,
    IN_QKV_OFFSET_1,
    IN_QKV_DESCALE_1,
    IN_QKV_WEIGHT_2,
    IN_QKV_SCALE_2,
    IN_QKV_OFFSET_2,
    IN_QKV_DESCALE_2,
    OUT_Q,
    OUT_K,
    OUT_V,
};

static const uint64_t QKV_IN_TENSOR_COUNT = 13;
static const uint64_t QKV_OUT_TENSOR_COUNT = 3;

template <class T>
atb::Status CreateQKVLinearSplit(const FusionAttentionParam &param, atb::Operation **operation, T config)
{
    atb::GraphParam opGraph;
    opGraph.name = param.isPack ? "QKVLinearSplitPack" : "QKVLinearSplitNoPack";
    opGraph.inTensorNum = QKV_IN_TENSOR_COUNT;
    opGraph.outTensorNum = QKV_OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = config.INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(config.NODE_COUNT);

    size_t nodeId = 0;

    atb::Node &linearNode = opGraph.nodes.at(nodeId++);
    atb_speed::llama_parallel::FusionLinearParam qkvLinearParam = param.qkvLinearParam;
    FusionLinear(qkvLinearParam, &linearNode.operation);
    linearNode.inTensorIds = {QKVLinearSplitTensorIdx::IN_QKV_INPUT, QKVLinearSplitTensorIdx::IN_QKV_WEIGHT_0, QKVLinearSplitTensorIdx::IN_QKV_SCALE_0, QKVLinearSplitTensorIdx::IN_QKV_OFFSET_0, QKVLinearSplitTensorIdx::IN_QKV_DESCALE_0};
    linearNode.outTensorIds = {param.isPack ? config.INTERMEDIATE_MIXED_QKV : QKVLinearSplitTensorIdx::OUT_Q};

    if (param.isPack && param.isGroupedQueryAttention) {  // Split GQA
        auto &sliceQNode = opGraph.nodes[nodeId++];
        atb::infer::SliceParam sliceQNodeParam;
        if (param.isFA) {  // FA相比于PA多了一个batchSize维度
            sliceQNodeParam.offsets = {0, 0, 0};
            sliceQNodeParam.size = {-1, -1, param.selfAttentionParam.headNum * param.selfAttentionParam.headDim};
        } else {
            sliceQNodeParam.offsets = {0, 0};
            sliceQNodeParam.size = {-1, param.selfAttentionParam.headNum * param.selfAttentionParam.headDim};
        }
        CreateOperation(sliceQNodeParam, &sliceQNode.operation);
        sliceQNode.inTensorIds = {config.INTERMEDIATE_MIXED_QKV};
        sliceQNode.outTensorIds = {QKVLinearSplitTensorIdx::OUT_Q};

        auto &sliceKVNode = opGraph.nodes[nodeId++];
        atb::infer::SliceParam sliceKVNodeParam;
        if (param.isFA) {  // FA相比于PA多了一个batchSize维度
            sliceKVNodeParam.offsets = {0, 0, param.selfAttentionParam.headNum * param.selfAttentionParam.headDim};
            sliceKVNodeParam.size = {-1, -1, param.selfAttentionParam.kvHeadNum * param.selfAttentionParam.headDim * 2};
        } else {
            sliceKVNodeParam.offsets = {0, param.selfAttentionParam.headNum * param.selfAttentionParam.headDim};
            sliceKVNodeParam.size = {-1, param.selfAttentionParam.kvHeadNum * param.selfAttentionParam.headDim * 2};
        }
        CreateOperation(sliceKVNodeParam, &sliceKVNode.operation);
        sliceKVNode.inTensorIds = {config.INTERMEDIATE_MIXED_QKV};
        sliceKVNode.outTensorIds = {config.INTERMEDIATE_KV};

        auto &splitKVNode = opGraph.nodes[nodeId++];
        atb::infer::SplitParam splitKVParam;
        splitKVParam.splitDim = -1;
        splitKVParam.splitNum = 2;
        CreateOperation(splitKVParam, &splitKVNode.operation);
        splitKVNode.inTensorIds = {config.INTERMEDIATE_KV};
        splitKVNode.outTensorIds = {QKVLinearSplitTensorIdx::OUT_K, QKVLinearSplitTensorIdx::OUT_V};
    } else if (param.isPack && !param.isGroupedQueryAttention) {  // Split MHA
        auto &splitMixedQKVNode = opGraph.nodes[nodeId++];
        atb::infer::SplitParam splitMixedQKVParam;
        splitMixedQKVParam.splitDim = -1;
        splitMixedQKVParam.splitNum = 3;
        CreateOperation(splitMixedQKVParam, &splitMixedQKVNode.operation);
        splitMixedQKVNode.inTensorIds = {config.INTERMEDIATE_MIXED_QKV};
        splitMixedQKVNode.outTensorIds = {QKVLinearSplitTensorIdx::OUT_Q, QKVLinearSplitTensorIdx::OUT_K, QKVLinearSplitTensorIdx::OUT_V};
    } else {  // isPack: false
        atb::Node &kLinearNode = opGraph.nodes.at(nodeId++);
        FusionLinear(qkvLinearParam, &kLinearNode.operation);
        kLinearNode.inTensorIds = {QKVLinearSplitTensorIdx::IN_QKV_INPUT, QKVLinearSplitTensorIdx::IN_QKV_WEIGHT_1, QKVLinearSplitTensorIdx::IN_QKV_SCALE_1, QKVLinearSplitTensorIdx::IN_QKV_OFFSET_1, QKVLinearSplitTensorIdx::IN_QKV_DESCALE_1};
        kLinearNode.outTensorIds = {QKVLinearSplitTensorIdx::OUT_K};

        atb::Node &vLinearNode = opGraph.nodes.at(nodeId++);
        FusionLinear(qkvLinearParam, &vLinearNode.operation);
        vLinearNode.inTensorIds = {QKVLinearSplitTensorIdx::IN_QKV_INPUT, QKVLinearSplitTensorIdx::IN_QKV_WEIGHT_2, QKVLinearSplitTensorIdx::IN_QKV_SCALE_2, QKVLinearSplitTensorIdx::IN_QKV_OFFSET_2, QKVLinearSplitTensorIdx::IN_QKV_DESCALE_2};
        vLinearNode.outTensorIds = {QKVLinearSplitTensorIdx::OUT_V};
    }

    opGraph.inferShapeFunc = [=]
                (const atb::SVector<atb::TensorDesc> &inTensorDescs, atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        outTensorDescs.at(0).shape.dims[inTensorDescs.at(0).shape.dimNum - 1] = param.selfAttentionParam.headNum * param.selfAttentionParam.headDim;

        outTensorDescs.at(1) = outTensorDescs.at(0);
        outTensorDescs.at(1).shape.dims[inTensorDescs.at(0).shape.dimNum - 1] = param.selfAttentionParam.kvHeadNum * param.selfAttentionParam.headDim;

        outTensorDescs.at(2) = outTensorDescs.at(1);
        return atb::NO_ERROR;
    };

    return atb::CreateOperation(opGraph, operation);
}

class QKVLinearSplitNoPackConfig {
public:
    uint64_t INTERMEDIATE_TENSOR_COUNT = 0;
    uint64_t NODE_COUNT = 3;

    enum QKVLinearSplitNoPackTensorIdx : uint32_t {
        INTERMEDIATE_MIXED_QKV = QKVLinearSplitTensorIdx::OUT_V + 1,  // no usage
        INTERMEDIATE_KV,  // no usage
    };
};

class QKVLinearSplitPackMHAConfig {
public:
    uint64_t INTERMEDIATE_TENSOR_COUNT = 1;
    uint64_t NODE_COUNT = 2;

    enum QKVLinearSplitPackTensorIdx : uint32_t {
        INTERMEDIATE_MIXED_QKV = QKVLinearSplitTensorIdx::OUT_V + 1,
        INTERMEDIATE_KV,  // no usage
    };
};

class QKVLinearSplitPackGQAConfig {
public:
    uint64_t INTERMEDIATE_TENSOR_COUNT = 2;
    uint64_t NODE_COUNT = 4;

    enum QKVLinearSplitPackTensorIdx : uint32_t {
        INTERMEDIATE_MIXED_QKV = QKVLinearSplitTensorIdx::OUT_V + 1,
        INTERMEDIATE_KV,
    };
};

atb::Status FusionAttention::QKVLinearSplit(const FusionAttentionParam &param_, atb::Operation **operation)
{
    if (param_.isPack && param_.isGroupedQueryAttention) {  // Pack + GQA
        QKVLinearSplitPackGQAConfig qkvLinearSplitPackGQAConfig;
        return CreateQKVLinearSplit(param_, operation, qkvLinearSplitPackGQAConfig);
    } else if (param_.isPack && !param_.isGroupedQueryAttention) {  // Pack + MHA
        QKVLinearSplitPackMHAConfig qkvLinearSplitPackMHAConfig;
        return CreateQKVLinearSplit(param_, operation, qkvLinearSplitPackMHAConfig);
    } else {  // No Pack
        QKVLinearSplitNoPackConfig qkvLinearSplitNoPackConfig;
        return CreateQKVLinearSplit(param_, operation, qkvLinearSplitNoPackConfig);
    }
}

void unSqueezeBatchSize(const atb::Dims &oldShape, atb::Dims &newShape, int batchSize)
{
    newShape.dimNum = 4;
    newShape.dims[0] = batchSize;
    newShape.dims[1] = oldShape.dims[0] / batchSize;
    newShape.dims[2] = oldShape.dims[1];
    newShape.dims[3] = oldShape.dims[2];
}

void unSqueezeLayerAxis(const atb::Dims &oldShape, atb::Dims &newShape)
{
    newShape.dimNum = 4;
    newShape.dims[0] = 1;  // Layer Axis
    newShape.dims[1] = oldShape.dims[0];
    newShape.dims[2] = oldShape.dims[1];
    newShape.dims[3] = oldShape.dims[2];
}

enum SelfAttentionTensorIdx : uint32_t {
    IN_SELF_ATTENTION_POSITION_EMBED_Q = 0,
    IN_SELF_ATTENTION_POSITION_EMBED_K,
    IN_SELF_ATTENTION_V,
    IN_SELF_ATTENTION_K_CACHE,
    IN_SELF_ATTENTION_V_CACHE,
    IN_SELF_ATTENTION_ATTENTION_MASK,
    IN_SELF_ATTENTION_TOKEN_OFFSET,
    IN_SELF_ATTENTION_SEQ_LEN,
    IN_SELF_ATTENTION_LAYER_ID,
    IN_SELF_ATTENTION_BLOCK_TABLES,
    IN_SELF_ATTENTION_SLOTS,
    OUT_SELF_ATTENTION,
};

static const uint64_t SELF_ATTENTION_IN_TENSOR_COUNT = 11;
static const uint64_t SELF_ATTENTION_OUT_TENSOR_COUNT = 1;
static const uint64_t SELF_ATTENTION_INTERMEDIATE_TENSOR_COUNT = 0;
static const uint64_t SELF_ATTENTION_FA_NODE_COUNT = 1;
static const uint64_t SELF_ATTENTION_PA_NODE_COUNT = 2;

atb::Status FusionAttention::SelfAttention(const FusionAttentionParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.name = param.isFA ? "SelfAttentionFA" : "SelfAttentionPA";
    opGraph.inTensorNum = SELF_ATTENTION_IN_TENSOR_COUNT;
    opGraph.outTensorNum = SELF_ATTENTION_OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = SELF_ATTENTION_INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(param.isFA ? SELF_ATTENTION_FA_NODE_COUNT : SELF_ATTENTION_PA_NODE_COUNT);

    size_t nodeId = 0;

    if (!param.isFA) {  // PA
        atb::Node &reshapeAndCacheNode = opGraph.nodes.at(nodeId++);
        atb::infer::ReshapeAndCacheParam reshapeCacheParm;
        CreateOperation(reshapeCacheParm, &reshapeAndCacheNode.operation);
        reshapeAndCacheNode.inTensorIds = {
            SelfAttentionTensorIdx::IN_SELF_ATTENTION_POSITION_EMBED_K,  // shape: [1, numAttentionHeadsPerRank * hiddenSizePerAttentionHead]
            SelfAttentionTensorIdx::IN_SELF_ATTENTION_V,                 // shape: [1, seqLen, numAttentionHeadsPerRank * hiddenSizePerAttentionHead]
            SelfAttentionTensorIdx::IN_SELF_ATTENTION_K_CACHE,           // shape: [2622, hiddenSizePerAttentionHead, numAttentionHeadsPerRank, hiddenSizePerAttentionHead]
            SelfAttentionTensorIdx::IN_SELF_ATTENTION_V_CACHE,           // shape: [2622, hiddenSizePerAttentionHead, numAttentionHeadsPerRank, hiddenSizePerAttentionHead]
            SelfAttentionTensorIdx::IN_SELF_ATTENTION_SLOTS              // shape: [seqLen]
        };
        reshapeAndCacheNode.outTensorIds = {};
    }

    atb::Node &selfAttentionNode = opGraph.nodes.at(nodeId++);
    if (param.isFA) { // FA
        CreateOperation(param.selfAttentionParam, &selfAttentionNode.operation);
        selfAttentionNode.inTensorIds = {
            SelfAttentionTensorIdx::IN_SELF_ATTENTION_POSITION_EMBED_Q,   // shape: [seqLen, headNum, hiddenSizePerHead]
            SelfAttentionTensorIdx::IN_SELF_ATTENTION_POSITION_EMBED_K,   // shape: [seqLen, headNum, hiddenSizePerHead]
            SelfAttentionTensorIdx::IN_SELF_ATTENTION_V,                  // shape: [batchSize, seqLen, headNum, hiddenSizePerHead]
            SelfAttentionTensorIdx::IN_SELF_ATTENTION_K_CACHE,
            SelfAttentionTensorIdx::IN_SELF_ATTENTION_V_CACHE,
            SelfAttentionTensorIdx::IN_SELF_ATTENTION_ATTENTION_MASK,
            SelfAttentionTensorIdx::IN_SELF_ATTENTION_TOKEN_OFFSET,
            SelfAttentionTensorIdx::IN_SELF_ATTENTION_SEQ_LEN,
            SelfAttentionTensorIdx::IN_SELF_ATTENTION_LAYER_ID
        };
        selfAttentionNode.outTensorIds = {SelfAttentionTensorIdx::OUT_SELF_ATTENTION};
        // Unsqeeze batch size of POSITION_EMBED_Q and POSITION_EMBED_K
        selfAttentionNode.inTensorReshapeFuncs.resize(selfAttentionNode.inTensorIds.size());
        selfAttentionNode.inTensorReshapeFuncs.at(0) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            unSqueezeBatchSize(oldShape, newShape, *batchNumPtr);
        };
        selfAttentionNode.inTensorReshapeFuncs.at(1) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            unSqueezeBatchSize(oldShape, newShape, *batchNumPtr);
        };
        // Unsqueeze layer axis of kv cache
        selfAttentionNode.inTensorReshapeFuncs.at(3) = &unSqueezeLayerAxis;
        selfAttentionNode.inTensorReshapeFuncs.at(4) = &unSqueezeLayerAxis;
    } else if (!param.isFA && param.isPrefill) {  // PA Prefill
        CreateOperation(param.selfAttentionParam, &selfAttentionNode.operation);
        selfAttentionNode.inTensorIds = {SelfAttentionTensorIdx::IN_SELF_ATTENTION_POSITION_EMBED_Q, SelfAttentionTensorIdx::IN_SELF_ATTENTION_POSITION_EMBED_K, SelfAttentionTensorIdx::IN_SELF_ATTENTION_V, SelfAttentionTensorIdx::IN_SELF_ATTENTION_ATTENTION_MASK, SelfAttentionTensorIdx::IN_SELF_ATTENTION_SEQ_LEN};
        selfAttentionNode.outTensorIds = {SelfAttentionTensorIdx::OUT_SELF_ATTENTION};
    } else {  // PA Decode
        if (param.isBF16) {
            selfAttentionNode.inTensorIds = {SelfAttentionTensorIdx::IN_SELF_ATTENTION_POSITION_EMBED_Q, SelfAttentionTensorIdx::IN_SELF_ATTENTION_K_CACHE, SelfAttentionTensorIdx::IN_SELF_ATTENTION_V_CACHE, SelfAttentionTensorIdx::IN_SELF_ATTENTION_BLOCK_TABLES,
                                             SelfAttentionTensorIdx::IN_SELF_ATTENTION_SEQ_LEN, SelfAttentionTensorIdx::IN_SELF_ATTENTION_ATTENTION_MASK};
        } else {
            selfAttentionNode.inTensorIds = {
                SelfAttentionTensorIdx::IN_SELF_ATTENTION_POSITION_EMBED_Q,  // shape: [1, numAttentionHeadsPerRank * hiddenSizePerAttentionHead]
                SelfAttentionTensorIdx::IN_SELF_ATTENTION_K_CACHE,           // shape: [2622, hiddenSizePerAttentionHead, numAttentionHeadsPerRank, hiddenSizePerAttentionHead]
                SelfAttentionTensorIdx::IN_SELF_ATTENTION_V_CACHE,           // shape: [2622, hiddenSizePerAttentionHead, numAttentionHeadsPerRank, hiddenSizePerAttentionHead]
                SelfAttentionTensorIdx::IN_SELF_ATTENTION_BLOCK_TABLES,      // shape: [seqLen, seqLen]
                SelfAttentionTensorIdx::IN_SELF_ATTENTION_SEQ_LEN            // shape: [seqLen]
            };
        }
        CreateOperation(param.pageAttentionParam, &selfAttentionNode.operation);
        selfAttentionNode.outTensorIds = {SelfAttentionTensorIdx::OUT_SELF_ATTENTION};  // shape: [seqLen, numAttentionHeadsPerRank, hiddenSizePerAttentionHead]
    }

    opGraph.inferShapeFunc = [=]
                (const atb::SVector<atb::TensorDesc> &inTensorDescs, atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        if (param.isFA) {
            outTensorDescs.at(0).shape.dimNum = 3;
            outTensorDescs.at(0).shape.dims[0] = *batchNumPtr;  // batchSize
            outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[0] / (*batchNumPtr); // seqLen
            outTensorDescs.at(0).shape.dims[2] = param.selfAttentionParam.headNum * param.selfAttentionParam.headDim;
        } else {
            outTensorDescs.at(0).shape.dimNum = 2;
            outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0]; // seqLen
            outTensorDescs.at(0).shape.dims[1] = param.selfAttentionParam.headNum * param.selfAttentionParam.headDim;
        }
        return atb::NO_ERROR;
    };

    return atb::CreateOperation(opGraph, operation);
}

enum AttentionTensorIdx : uint32_t {
    IN_INPUT = 0,
    IN_WEIGHT_0,  // q or mixed qkv
    IN_SCALE_0,
    IN_OFFSET_0,
    IN_DESCALE_0,
    IN_WEIGHT_1,  // k
    IN_SCALE_1,
    IN_OFFSET_1,
    IN_DESCALE_1,
    IN_WEIGHT_2,  // v
    IN_SCALE_2,
    IN_OFFSET_2,
    IN_DESCALE_2,
    IN_COS_TABLE,
    IN_SIN_TABLE,
    IN_SEQ_LEN,
    IN_K_CACHE,
    IN_V_CACHE,
    IN_ATTENTION_MASK,
    IN_TOKEN_OFFSET,
    IN_LAYER_ID,
    IN_BLOCK_TABLES,
    IN_SLOTS,
    IN_WEIGHT_OUT,  // out
    IN_SCALE_OUT,
    IN_OFFSET_OUT,
    IN_DESCALE_OUT,
    OUT_ATTENTION,
    INTERMIDATE_Q,  // shape: PA: [seqLen, numAttentionHeadsPerRank * hiddenSizePerAttentionHead]
    INTERMIDATE_K,  // shape: PA: [seqLen, numAttentionHeadsPerRank * hiddenSizePerAttentionHead]
    INTERMIDATE_V,  // shape: PA: [seqLen, numAttentionHeadsPerRank * hiddenSizePerAttentionHead]
    INTERMIDATE_POSITION_EMBED_Q,  // shape: PA: [seqLen, numAttentionHeadsPerRank * hiddenSizePerAttentionHead]
    INTERMIDATE_POSITION_EMBED_K,  // shape: PA: [seqLen, numAttentionHeadsPerRank * hiddenSizePerAttentionHead]
    INTERMIDATE_SELF_ATTENTION  // shape: PA: [seqLen, numAttentionHeadsPerRank, hiddenSizePerAttentionHead]
};

static const uint64_t ATTENTION_IN_TENSOR_COUNT = 27;
static const uint64_t ATTENTION_OUT_TENSOR_COUNT = 1;
static const uint64_t ATTENTION_INTERMEDIATE_TENSOR_COUNT = 6;
static const uint64_t ATTENTION_NODE_COUNT = 4;

void unsqueezeByHeadNum(const atb::Dims &oldShape, atb::Dims &newShape, int headNum, int headDim)
{
    if (oldShape.dimNum == 3) {
        newShape.dimNum = 4;
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = oldShape.dims[1];
        newShape.dims[2] = headNum;
        newShape.dims[3] = headDim;
    } else {
        newShape.dimNum = 3;
        newShape.dims[0] = oldShape.dims[0];       // seqLen
        newShape.dims[1] = headNum;                // numAttentionHeadsPerRank
        newShape.dims[2] = headDim;                // hiddenSizePerAttentionHead
    }
}

void unsqueezeByKVHeadNum(const atb::Dims &oldShape, atb::Dims &newShape, int kvHeadNum, int headDim)
{
    if (oldShape.dimNum == 3) {
        newShape.dimNum = 4;
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = oldShape.dims[1];
        newShape.dims[2] = kvHeadNum;
        newShape.dims[3] = headDim;
    } else {
        newShape.dimNum = 3;
        newShape.dims[0] = oldShape.dims[0];       // seqLen
        newShape.dims[1] = kvHeadNum;              // numKeyValueHeadsPerRank
        newShape.dims[2] = headDim;                // hiddenSizePerAttentionHead
    }
}

void squeezeBatchSize(const atb::Dims &oldShape, atb::Dims &newShape)
{
    newShape.dimNum = 2;
    newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
    newShape.dims[1] = oldShape.dims[2];
}

atb::Status FusionAttention::Attention(const FusionAttentionParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.name = "Attention";
    opGraph.inTensorNum = ATTENTION_IN_TENSOR_COUNT;
    opGraph.outTensorNum = ATTENTION_OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = ATTENTION_INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(ATTENTION_NODE_COUNT);

    size_t nodeId = 0;

    atb::Node &qkvLinearSplitNode = opGraph.nodes.at(nodeId++);
    QKVLinearSplit(param, &qkvLinearSplitNode.operation);
    qkvLinearSplitNode.inTensorIds = {
        AttentionTensorIdx::IN_INPUT, AttentionTensorIdx::IN_WEIGHT_0, AttentionTensorIdx::IN_SCALE_0, AttentionTensorIdx::IN_OFFSET_0, AttentionTensorIdx::IN_DESCALE_0,
        AttentionTensorIdx::IN_WEIGHT_1, AttentionTensorIdx::IN_SCALE_1, AttentionTensorIdx::IN_OFFSET_1, AttentionTensorIdx::IN_DESCALE_1,
        AttentionTensorIdx::IN_WEIGHT_2, AttentionTensorIdx::IN_SCALE_2, AttentionTensorIdx::IN_OFFSET_2, AttentionTensorIdx::IN_DESCALE_2,
    };
    qkvLinearSplitNode.outTensorIds = {AttentionTensorIdx::INTERMIDATE_Q, AttentionTensorIdx::INTERMIDATE_K, AttentionTensorIdx::INTERMIDATE_V};

    atb::Node &ropeNode = opGraph.nodes.at(nodeId++);
    atb::infer::RopeParam ropeparam;
    ropeparam.rotaryCoeff = param.rotaryCoeff;
    CreateOperation(ropeparam, &ropeNode.operation);
    ropeNode.inTensorIds = {AttentionTensorIdx::INTERMIDATE_Q, AttentionTensorIdx::INTERMIDATE_K, AttentionTensorIdx::IN_COS_TABLE, AttentionTensorIdx::IN_SIN_TABLE, AttentionTensorIdx::IN_SEQ_LEN};
    ropeNode.outTensorIds = {AttentionTensorIdx::INTERMIDATE_POSITION_EMBED_Q, AttentionTensorIdx::INTERMIDATE_POSITION_EMBED_K};
    if (param.isFA) {
        ropeNode.inTensorReshapeFuncs.resize(ropeNode.inTensorIds.size());
        ropeNode.inTensorReshapeFuncs.at(0) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            *batchNumPtr = oldShape.dims[0];
            squeezeBatchSize(oldShape, newShape);
        };
        ropeNode.inTensorReshapeFuncs.at(1) = &squeezeBatchSize;
    }

    atb::Node &selfAttentionNode = opGraph.nodes.at(nodeId++);
    SelfAttention(param, &selfAttentionNode.operation);
    selfAttentionNode.inTensorIds = {
        AttentionTensorIdx::INTERMIDATE_POSITION_EMBED_Q,
        AttentionTensorIdx::INTERMIDATE_POSITION_EMBED_K,
        AttentionTensorIdx::INTERMIDATE_V,
        AttentionTensorIdx::IN_K_CACHE,
        AttentionTensorIdx::IN_V_CACHE,
        AttentionTensorIdx::IN_ATTENTION_MASK,
        AttentionTensorIdx::IN_TOKEN_OFFSET,
        AttentionTensorIdx::IN_SEQ_LEN,
        AttentionTensorIdx::IN_LAYER_ID,
        AttentionTensorIdx::IN_BLOCK_TABLES,
        AttentionTensorIdx::IN_SLOTS,
    };
    selfAttentionNode.outTensorIds = {AttentionTensorIdx::INTERMIDATE_SELF_ATTENTION};
    selfAttentionNode.inTensorReshapeFuncs.resize(selfAttentionNode.inTensorIds.size());
    selfAttentionNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        unsqueezeByHeadNum(oldShape, newShape, param.selfAttentionParam.headNum, param.selfAttentionParam.headDim);
    };
    selfAttentionNode.inTensorReshapeFuncs[1] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        unsqueezeByHeadNum(oldShape, newShape, param.selfAttentionParam.kvHeadNum, param.selfAttentionParam.headDim);
    };
    selfAttentionNode.inTensorReshapeFuncs[2] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        unsqueezeByHeadNum(oldShape, newShape, param.selfAttentionParam.kvHeadNum, param.selfAttentionParam.headDim);
    };

    atb::Node &selfOutLinearParallelNode = opGraph.nodes.at(nodeId++);
    atb_speed::llama_parallel::LinearParallelParam selfOutLinearParam = param.selfOutLinearParallelParam;
    LinearParallel(selfOutLinearParam, &selfOutLinearParallelNode.operation);
    selfOutLinearParallelNode.inTensorIds = {AttentionTensorIdx::INTERMIDATE_SELF_ATTENTION, AttentionTensorIdx::IN_WEIGHT_OUT, AttentionTensorIdx::IN_SCALE_OUT, AttentionTensorIdx::IN_OFFSET_OUT, AttentionTensorIdx::IN_DESCALE_OUT};
    selfOutLinearParallelNode.outTensorIds = {AttentionTensorIdx::OUT_ATTENTION};

    return atb::CreateOperation(opGraph, operation);
}

} // namespace llama_parallel
} // namespace atb_speed