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

#include "layers/operations/fusion_attention.h"

namespace atb_speed {
namespace common {

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
    atb_speed::common::FusionLinearParam qkvLinearParam = param.qkvLinearParam;
    FusionLinear(qkvLinearParam, &linearNode.operation);
    linearNode.inTensorIds = {
        QKVLinearSplitTensorIdx::IN_QKV_INPUT, QKVLinearSplitTensorIdx::IN_QKV_WEIGHT_0,
        QKVLinearSplitTensorIdx::IN_QKV_SCALE_0, QKVLinearSplitTensorIdx::IN_QKV_OFFSET_0,
        QKVLinearSplitTensorIdx::IN_QKV_DESCALE_0
    };
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
        CREATE_OPERATION(sliceQNodeParam, &sliceQNode.operation);
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
        CREATE_OPERATION(sliceKVNodeParam, &sliceKVNode.operation);
        sliceKVNode.inTensorIds = {config.INTERMEDIATE_MIXED_QKV};
        sliceKVNode.outTensorIds = {config.INTERMEDIATE_KV};

        auto &splitKVNode = opGraph.nodes[nodeId++];
        atb::infer::SplitParam splitKVParam;
        splitKVParam.splitDim = -1;
        splitKVParam.splitNum = 2;
        CREATE_OPERATION(splitKVParam, &splitKVNode.operation);
        splitKVNode.inTensorIds = {config.INTERMEDIATE_KV};
        splitKVNode.outTensorIds = {QKVLinearSplitTensorIdx::OUT_K, QKVLinearSplitTensorIdx::OUT_V};
    } else if (param.isPack && !param.isGroupedQueryAttention) {  // Split MHA
        auto &splitMixedQKVNode = opGraph.nodes[nodeId++];
        atb::infer::SplitParam splitMixedQKVParam;
        splitMixedQKVParam.splitDim = -1;
        splitMixedQKVParam.splitNum = 3;
        CREATE_OPERATION(splitMixedQKVParam, &splitMixedQKVNode.operation);
        splitMixedQKVNode.inTensorIds = {config.INTERMEDIATE_MIXED_QKV};
        splitMixedQKVNode.outTensorIds = {
            QKVLinearSplitTensorIdx::OUT_Q, QKVLinearSplitTensorIdx::OUT_K,
            QKVLinearSplitTensorIdx::OUT_V
        };
    } else {  // isPack: false
        atb::Node &kLinearNode = opGraph.nodes.at(nodeId++);
        FusionLinear(qkvLinearParam, &kLinearNode.operation);
        kLinearNode.inTensorIds = {
            QKVLinearSplitTensorIdx::IN_QKV_INPUT, QKVLinearSplitTensorIdx::IN_QKV_WEIGHT_1,
            QKVLinearSplitTensorIdx::IN_QKV_SCALE_1, QKVLinearSplitTensorIdx::IN_QKV_OFFSET_1,
            QKVLinearSplitTensorIdx::IN_QKV_DESCALE_1
        };
        kLinearNode.outTensorIds = {QKVLinearSplitTensorIdx::OUT_K};

        atb::Node &vLinearNode = opGraph.nodes.at(nodeId++);
        FusionLinear(qkvLinearParam, &vLinearNode.operation);
        vLinearNode.inTensorIds = {
            QKVLinearSplitTensorIdx::IN_QKV_INPUT, QKVLinearSplitTensorIdx::IN_QKV_WEIGHT_2,
            QKVLinearSplitTensorIdx::IN_QKV_SCALE_2, QKVLinearSplitTensorIdx::IN_QKV_OFFSET_2,
            QKVLinearSplitTensorIdx::IN_QKV_DESCALE_2
        };
        vLinearNode.outTensorIds = {QKVLinearSplitTensorIdx::OUT_V};
    }

    opGraph.inferShapeFunc = [=]
                (const atb::SVector<atb::TensorDesc> &inTensorDescs, atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        outTensorDescs.at(0).shape.dims[inTensorDescs.at(0).shape.dimNum - 1] \
            = param.selfAttentionParam.headNum * param.selfAttentionParam.headDim;

        outTensorDescs.at(1) = outTensorDescs.at(0);
        outTensorDescs.at(1).shape.dims[inTensorDescs.at(0).shape.dimNum - 1] \
            = param.selfAttentionParam.kvHeadNum * param.selfAttentionParam.headDim;

        outTensorDescs.at(2) = outTensorDescs.at(1);
        return atb::NO_ERROR;
    };

    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
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
        CREATE_OPERATION(reshapeCacheParm, &reshapeAndCacheNode.operation);
        reshapeAndCacheNode.inTensorIds = {
            SelfAttentionTensorIdx::IN_SELF_ATTENTION_POSITION_EMBED_K,
            SelfAttentionTensorIdx::IN_SELF_ATTENTION_V,
            SelfAttentionTensorIdx::IN_SELF_ATTENTION_K_CACHE,
            SelfAttentionTensorIdx::IN_SELF_ATTENTION_V_CACHE,
            SelfAttentionTensorIdx::IN_SELF_ATTENTION_SLOTS
        };
        reshapeAndCacheNode.outTensorIds = {};
    }

    atb::Node &selfAttentionNode = opGraph.nodes.at(nodeId++);
    if (param.isFA) { // FA
        CREATE_OPERATION(param.selfAttentionParam, &selfAttentionNode.operation);
        selfAttentionNode.inTensorIds = {
            SelfAttentionTensorIdx::IN_SELF_ATTENTION_POSITION_EMBED_Q,
            SelfAttentionTensorIdx::IN_SELF_ATTENTION_POSITION_EMBED_K,
            SelfAttentionTensorIdx::IN_SELF_ATTENTION_V,
            SelfAttentionTensorIdx::IN_SELF_ATTENTION_K_CACHE,
            SelfAttentionTensorIdx::IN_SELF_ATTENTION_V_CACHE,
            SelfAttentionTensorIdx::IN_SELF_ATTENTION_ATTENTION_MASK,
            SelfAttentionTensorIdx::IN_SELF_ATTENTION_TOKEN_OFFSET,
            SelfAttentionTensorIdx::IN_SELF_ATTENTION_SEQ_LEN,
            SelfAttentionTensorIdx::IN_SELF_ATTENTION_LAYER_ID
        };
        selfAttentionNode.outTensorIds = {SelfAttentionTensorIdx::OUT_SELF_ATTENTION};
    } else if (!param.isFA && param.isPrefill) {  // PA Prefill
        CREATE_OPERATION(param.selfAttentionParam, &selfAttentionNode.operation);
        selfAttentionNode.inTensorIds = {
            SelfAttentionTensorIdx::IN_SELF_ATTENTION_POSITION_EMBED_Q, SelfAttentionTensorIdx::IN_SELF_ATTENTION_POSITION_EMBED_K,
            SelfAttentionTensorIdx::IN_SELF_ATTENTION_V, SelfAttentionTensorIdx::IN_SELF_ATTENTION_ATTENTION_MASK,
            SelfAttentionTensorIdx::IN_SELF_ATTENTION_SEQ_LEN
        };
        selfAttentionNode.outTensorIds = {SelfAttentionTensorIdx::OUT_SELF_ATTENTION};
    } else {  // PA Decode
        if (param.isBF16) {
            selfAttentionNode.inTensorIds = {
                SelfAttentionTensorIdx::IN_SELF_ATTENTION_POSITION_EMBED_Q, SelfAttentionTensorIdx::IN_SELF_ATTENTION_K_CACHE,
                SelfAttentionTensorIdx::IN_SELF_ATTENTION_V_CACHE, SelfAttentionTensorIdx::IN_SELF_ATTENTION_BLOCK_TABLES,
                SelfAttentionTensorIdx::IN_SELF_ATTENTION_SEQ_LEN, SelfAttentionTensorIdx::IN_SELF_ATTENTION_ATTENTION_MASK
            };
        } else {
            selfAttentionNode.inTensorIds = {
                SelfAttentionTensorIdx::IN_SELF_ATTENTION_POSITION_EMBED_Q,
                SelfAttentionTensorIdx::IN_SELF_ATTENTION_K_CACHE,
                SelfAttentionTensorIdx::IN_SELF_ATTENTION_V_CACHE,
                SelfAttentionTensorIdx::IN_SELF_ATTENTION_BLOCK_TABLES,
                SelfAttentionTensorIdx::IN_SELF_ATTENTION_SEQ_LEN
            };
        }
        CREATE_OPERATION(param.pageAttentionParam, &selfAttentionNode.operation);
        selfAttentionNode.outTensorIds = {SelfAttentionTensorIdx::OUT_SELF_ATTENTION};
    }

    opGraph.inferShapeFunc = [=]
                (const atb::SVector<atb::TensorDesc> &inTensorDescs, atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum - 1;
        outTensorDescs.at(0).shape.dims[inTensorDescs.at(0).shape.dimNum - 2] \
            = param.selfAttentionParam.headNum * param.selfAttentionParam.headDim;
        return atb::NO_ERROR;
    };

    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
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
    // shape: PA: [seqLen, numAttentionHeadsPerRank * hiddenSizePerAttentionHead]
    // shape: FA: [batchSize, seqLen, numAttentionHeadsPerRank * hiddenSizePerAttentionHead]
    INTERMIDATE_Q,
    // shape: PA: [seqLen, numKeyValueHeadsPerRank * hiddenSizePerAttentionHead]
    // shape: FA: [batchSize, seqLen, numKeyValueHeadsPerRank * hiddenSizePerAttentionHead]
    INTERMIDATE_K,
    INTERMIDATE_V,  // same as INTERMIDATE_K
    // shape: PA: [unpadSeqLen, numAttentionHeadsPerRank * hiddenSizePerAttentionHead]
    // shape: FA: [batchSize * seqLen, numAttentionHeadsPerRank * hiddenSizePerAttentionHead]
    INTERMIDATE_POSITION_EMBED_Q,
    INTERMIDATE_POSITION_EMBED_K,  // same as INTERMIDATE_POSITION_EMBED_Q
    INTERMIDATE_SELF_ATTENTION  // shape: PA: [seqLen, numAttentionHeadsPerRank * hiddenSizePerAttentionHead]
};

static const uint64_t ATTENTION_IN_TENSOR_COUNT = 27;
static const uint64_t ATTENTION_OUT_TENSOR_COUNT = 1;
static const uint64_t ATTENTION_INTERMEDIATE_TENSOR_COUNT = 6;
static const uint64_t ATTENTION_NODE_COUNT = 4;

void unsqueezeByHeadNum(const atb::Dims &oldShape, atb::Dims &newShape, int headNum, int headDim)
{
    newShape.dimNum = 3;
    newShape.dims[0] = oldShape.dims[0];       // seqLen
    newShape.dims[1] = headNum;                // numAttentionHeadsPerRank or numKeyValueHeadsPerRank
    newShape.dims[2] = headDim;                // hiddenSizePerAttentionHead
}

void unsqueezeByHeadNumAndBatchSize(const atb::Dims &oldShape, atb::Dims &newShape,
                                    int batchSize, int headNum, int headDim)
{
    if (oldShape.dimNum == 3) {
        newShape.dimNum = 4;
        newShape.dims[0] = oldShape.dims[0];  // batchSize
        newShape.dims[1] = oldShape.dims[1];  // seqLen
        newShape.dims[2] = headNum;
        newShape.dims[3] = headDim;
    } else {
        newShape.dimNum = 4;
        newShape.dims[0] = batchSize;
        newShape.dims[1] = oldShape.dims[0] / batchSize;
        newShape.dims[2] = headNum;
        newShape.dims[3] = headDim;
    }
}

void squeezeBatchSize(const atb::Dims &oldShape, atb::Dims &newShape)
{
    newShape.dimNum = 2;
    newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
    newShape.dims[1] = oldShape.dims[2];
}

void unSqueezeLayerAxis(const atb::Dims &oldShape, atb::Dims &newShape)
{
    newShape.dimNum = oldShape.dimNum + 1;
    newShape.dims[0] = 1;  // Layer Axis
    for (auto i = 0; i < oldShape.dimNum; i++) {
        newShape.dims[i + 1] = oldShape.dims[i];
    }
}

atb::Status FusionAttention::Attention(const FusionAttentionParam &param, atb::Operation **operation)
{
    std::shared_ptr<int64_t> batchSizePtr = std::make_shared<int64_t>(0);
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
        AttentionTensorIdx::IN_INPUT, AttentionTensorIdx::IN_WEIGHT_0, AttentionTensorIdx::IN_SCALE_0,
        AttentionTensorIdx::IN_OFFSET_0, AttentionTensorIdx::IN_DESCALE_0,
        AttentionTensorIdx::IN_WEIGHT_1, AttentionTensorIdx::IN_SCALE_1, AttentionTensorIdx::IN_OFFSET_1,
        AttentionTensorIdx::IN_DESCALE_1, AttentionTensorIdx::IN_WEIGHT_2, AttentionTensorIdx::IN_SCALE_2,
        AttentionTensorIdx::IN_OFFSET_2, AttentionTensorIdx::IN_DESCALE_2,
    };
    qkvLinearSplitNode.outTensorIds = {
        AttentionTensorIdx::INTERMIDATE_Q, AttentionTensorIdx::INTERMIDATE_K, AttentionTensorIdx::INTERMIDATE_V
    };

    atb::Node &ropeNode = opGraph.nodes.at(nodeId++);
    atb::infer::RopeParam ropeparam;
    ropeparam.rotaryCoeff = param.rotaryCoeff;
    CREATE_OPERATION(ropeparam, &ropeNode.operation);
    ropeNode.inTensorIds = {
        AttentionTensorIdx::INTERMIDATE_Q, AttentionTensorIdx::INTERMIDATE_K, AttentionTensorIdx::IN_COS_TABLE,
        AttentionTensorIdx::IN_SIN_TABLE, AttentionTensorIdx::IN_SEQ_LEN
    };
    ropeNode.outTensorIds = {
        AttentionTensorIdx::INTERMIDATE_POSITION_EMBED_Q, AttentionTensorIdx::INTERMIDATE_POSITION_EMBED_K
    };
    if (param.isFA) {
        ropeNode.inTensorReshapeFuncs.resize(ropeNode.inTensorIds.size());
        ropeNode.inTensorReshapeFuncs.at(0) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            *batchSizePtr = oldShape.dims[0];
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
    if (param.isFA) {
        selfAttentionNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            unsqueezeByHeadNumAndBatchSize(oldShape, newShape, (*batchSizePtr),
                                           param.selfAttentionParam.headNum, param.selfAttentionParam.headDim);
        };
        selfAttentionNode.inTensorReshapeFuncs[1] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            unsqueezeByHeadNumAndBatchSize(oldShape, newShape, (*batchSizePtr),
                                           param.selfAttentionParam.kvHeadNum, param.selfAttentionParam.headDim);
        };
        selfAttentionNode.inTensorReshapeFuncs[2] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            unsqueezeByHeadNumAndBatchSize(oldShape, newShape, (*batchSizePtr),
                                           param.selfAttentionParam.kvHeadNum, param.selfAttentionParam.headDim);
        };
        // Unsqueeze layer axis of kv cache
        selfAttentionNode.inTensorReshapeFuncs.at(3) = &unSqueezeLayerAxis;
        selfAttentionNode.inTensorReshapeFuncs.at(4) = &unSqueezeLayerAxis;
    } else {
        selfAttentionNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            unsqueezeByHeadNum(oldShape, newShape,
                               param.selfAttentionParam.headNum, param.selfAttentionParam.headDim);
        };
        selfAttentionNode.inTensorReshapeFuncs[1] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            unsqueezeByHeadNum(oldShape, newShape,
                               param.selfAttentionParam.kvHeadNum, param.selfAttentionParam.headDim);
        };
        selfAttentionNode.inTensorReshapeFuncs[2] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            unsqueezeByHeadNum(oldShape, newShape,
                               param.selfAttentionParam.kvHeadNum, param.selfAttentionParam.headDim);
        };
    }

    atb::Node &selfOutLinearParallelNode = opGraph.nodes.at(nodeId++);
    atb_speed::common::LinearParallelParam selfOutLinearParam = param.selfOutLinearParallelParam;
    LinearParallel(selfOutLinearParam, &selfOutLinearParallelNode.operation);
    selfOutLinearParallelNode.inTensorIds = {
        AttentionTensorIdx::INTERMIDATE_SELF_ATTENTION, AttentionTensorIdx::IN_WEIGHT_OUT,
        AttentionTensorIdx::IN_SCALE_OUT, AttentionTensorIdx::IN_OFFSET_OUT, AttentionTensorIdx::IN_DESCALE_OUT
    };
    selfOutLinearParallelNode.outTensorIds = {AttentionTensorIdx::OUT_ATTENTION};

    opGraph.inferShapeFunc = [=]
                (const atb::SVector<atb::TensorDesc> &inTensorDescs, atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
    };

    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}

} // namespace common
} // namespace atb_speed