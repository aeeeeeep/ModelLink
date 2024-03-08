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
#include "positional_embedding.h"
namespace atb_speed {
namespace common {

enum QKVLinearSplitTensorIdx : uint32_t {
    IN_QKV_INPUT = 0,
    IN_QKV_NORM_WEIGHT,
    IN_QKV_NORM_BIAS,
    IN_QKV_NORM_NEW_WEIGHT,
    IN_QKV_NORM_NEW_BIAS,
    IN_QKV_WEIGHT_0,
    IN_QKV_SCALE_0,
    IN_QKV_OFFSET_0,
    IN_QKV_DESCALE_0,
    IN_QKV_BIAS_0,
    IN_QKV_WEIGHT_1,
    IN_QKV_SCALE_1,
    IN_QKV_OFFSET_1,
    IN_QKV_DESCALE_1,
    IN_QKV_BIAS_1,
    IN_QKV_WEIGHT_2,
    IN_QKV_SCALE_2,
    IN_QKV_OFFSET_2,
    IN_QKV_DESCALE_2,
    IN_QKV_BIAS_2,
    OUT_Q,
    OUT_K,
    OUT_V,
    INTERMEDIATE_MIXED_QKV,
    INTERMEDIATE_KV,
};

static const uint64_t QKV_IN_TENSOR_COUNT = 20;
static const uint64_t QKV_OUT_TENSOR_COUNT = 3;
static const uint64_t QKV_NO_PACK_INTERMEDIATE_TENSOR_COUNT = 0;
static const uint64_t QKV_NO_PACK_NODE_COUNT = 3;
static const uint64_t QKV_PACK_MHA_INTERMEDIATE_TENSOR_COUNT = 1;
static const uint64_t QKV_PACK_MHA_NODE_COUNT = 2;
static const uint64_t QKV_PACK_GQA_INTERMEDIATE_TENSOR_COUNT = 2;
static const uint64_t QKV_PACK_GQA_NODE_COUNT = 4;

template <typename NormParamType>
atb::Status QKVLinearSplit(const FusionAttentionParam<NormParamType> &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.name = param.isPack ? "QKVLinearSplitPack" : "QKVLinearSplitNoPack";
    opGraph.inTensorNum = QKV_IN_TENSOR_COUNT;
    opGraph.outTensorNum = QKV_OUT_TENSOR_COUNT;
    if (param.isPack && param.isGroupedQueryAttention) {  // Pack + GQA
        opGraph.internalTensorNum = QKV_PACK_GQA_INTERMEDIATE_TENSOR_COUNT;
        opGraph.nodes.resize(QKV_PACK_GQA_NODE_COUNT);
    } else if (param.isPack && !param.isGroupedQueryAttention) {  // Pack + MHA
        opGraph.internalTensorNum = QKV_PACK_MHA_INTERMEDIATE_TENSOR_COUNT;
        opGraph.nodes.resize(QKV_PACK_MHA_NODE_COUNT);
    } else {  // No Pack
        opGraph.internalTensorNum = QKV_NO_PACK_INTERMEDIATE_TENSOR_COUNT;
        opGraph.nodes.resize(QKV_NO_PACK_NODE_COUNT);
    }

    size_t nodeId = 0;

    atb::Node &qNormLinearNode = opGraph.nodes.at(nodeId++);
    atb_speed::common::NormLinearParam<NormParamType> qNormLinearParam;
    qNormLinearParam.isAntiOutlier = param.isAntiOutlier;
    if (param.packQuantType == atb_speed::common::ALL_W8A16) {
        qNormLinearParam.fusionLinearParam.quantType = W8A16;
    } else {
        qNormLinearParam.fusionLinearParam.quantType \
            = param.layerLinearQuantType[0] == atb_speed::common::LinearType::FP ? NO_QUANT : NORM_QUANT_LINEAR_DEQUANT;
    }
    qNormLinearParam.fusionLinearParam.isBF16 = param.isBF16;
    qNormLinearParam.fusionLinearParam.hasBias = param.qkvHasBias;
    qNormLinearParam.normParamType = param.normParamType;
    qNormLinearParam.normQuantParamType = param.normQuantParamType;
    NormLinear<NormParamType>(qNormLinearParam, &qNormLinearNode.operation);
    qNormLinearNode.inTensorIds = {
        QKVLinearSplitTensorIdx::IN_QKV_INPUT, QKVLinearSplitTensorIdx::IN_QKV_NORM_WEIGHT, QKVLinearSplitTensorIdx::IN_QKV_NORM_BIAS,
        QKVLinearSplitTensorIdx::IN_QKV_NORM_NEW_WEIGHT, QKVLinearSplitTensorIdx::IN_QKV_NORM_NEW_BIAS,
        QKVLinearSplitTensorIdx::IN_QKV_WEIGHT_0,
        QKVLinearSplitTensorIdx::IN_QKV_SCALE_0, QKVLinearSplitTensorIdx::IN_QKV_OFFSET_0,
        QKVLinearSplitTensorIdx::IN_QKV_DESCALE_0, QKVLinearSplitTensorIdx::IN_QKV_BIAS_0
    };
    qNormLinearNode.outTensorIds = {param.isPack ? QKVLinearSplitTensorIdx::INTERMEDIATE_MIXED_QKV : QKVLinearSplitTensorIdx::OUT_Q};

    if (param.isPack && param.isGroupedQueryAttention) {  // Split GQA
        auto &sliceQNode = opGraph.nodes[nodeId++];
        atb::infer::SliceParam sliceQNodeParam;
        if (param.isFA) {  // FA相比于PA多了一个batchSize维度
            sliceQNodeParam.offsets = {0, 0, 0};
            sliceQNodeParam.size = {-1, -1, param.selfAttentionParam.headNum * param.headDim};
        } else {
            sliceQNodeParam.offsets = {0, 0};
            sliceQNodeParam.size = {-1, param.selfAttentionParam.headNum * param.headDim};
        }
        CREATE_OPERATION(sliceQNodeParam, &sliceQNode.operation);
        sliceQNode.inTensorIds = {QKVLinearSplitTensorIdx::INTERMEDIATE_MIXED_QKV};
        sliceQNode.outTensorIds = {QKVLinearSplitTensorIdx::OUT_Q};

        auto &sliceKVNode = opGraph.nodes[nodeId++];
        atb::infer::SliceParam sliceKVNodeParam;
        if (param.isFA) {  // FA相比于PA多了一个batchSize维度
            sliceKVNodeParam.offsets = {0, 0, param.selfAttentionParam.headNum * param.headDim};
            sliceKVNodeParam.size = {-1, -1, param.selfAttentionParam.kvHeadNum * param.headDim * 2};
        } else {
            sliceKVNodeParam.offsets = {0, param.selfAttentionParam.headNum * param.headDim};
            sliceKVNodeParam.size = {-1, param.selfAttentionParam.kvHeadNum * param.headDim * 2};
        }
        CREATE_OPERATION(sliceKVNodeParam, &sliceKVNode.operation);
        sliceKVNode.inTensorIds = {QKVLinearSplitTensorIdx::INTERMEDIATE_MIXED_QKV};
        sliceKVNode.outTensorIds = {QKVLinearSplitTensorIdx::INTERMEDIATE_KV};

        auto &splitKVNode = opGraph.nodes[nodeId++];
        atb::infer::SplitParam splitKVParam;
        splitKVParam.splitDim = -1;
        splitKVParam.splitNum = 2;
        CREATE_OPERATION(splitKVParam, &splitKVNode.operation);
        splitKVNode.inTensorIds = {QKVLinearSplitTensorIdx::INTERMEDIATE_KV};
        splitKVNode.outTensorIds = {QKVLinearSplitTensorIdx::OUT_K, QKVLinearSplitTensorIdx::OUT_V};
    } else if (param.isPack && !param.isGroupedQueryAttention) {  // Split MHA
        auto &splitMixedQKVNode = opGraph.nodes[nodeId++];
        atb::infer::SplitParam splitMixedQKVParam;
        splitMixedQKVParam.splitDim = -1;
        splitMixedQKVParam.splitNum = 3;
        CREATE_OPERATION(splitMixedQKVParam, &splitMixedQKVNode.operation);
        splitMixedQKVNode.inTensorIds = {QKVLinearSplitTensorIdx::INTERMEDIATE_MIXED_QKV};
        splitMixedQKVNode.outTensorIds = {
            QKVLinearSplitTensorIdx::OUT_Q, QKVLinearSplitTensorIdx::OUT_K,
            QKVLinearSplitTensorIdx::OUT_V
        };
    } else {  // isPack: false
        atb::Node &kNormLinearNode = opGraph.nodes.at(nodeId++);
        atb_speed::common::NormLinearParam<NormParamType> kNormLinearParam;
        kNormLinearParam.isAntiOutlier = param.isAntiOutlier;
        if (param.packQuantType == atb_speed::common::ALL_W8A16) {
            kNormLinearParam.fusionLinearParam.quantType = W8A16;
        } else {
            kNormLinearParam.fusionLinearParam.quantType \
                = param.layerLinearQuantType[1] == atb_speed::common::LinearType::FP ? NO_QUANT : NORM_QUANT_LINEAR_DEQUANT;
        }
        kNormLinearParam.fusionLinearParam.isBF16 = param.isBF16;
        kNormLinearParam.fusionLinearParam.hasBias = param.qkvHasBias;
        kNormLinearParam.normParamType = param.normParamType;
        kNormLinearParam.normQuantParamType = param.normQuantParamType;
        NormLinear<NormParamType>(kNormLinearParam, &kNormLinearNode.operation);
        kNormLinearNode.inTensorIds = {
            QKVLinearSplitTensorIdx::IN_QKV_INPUT, QKVLinearSplitTensorIdx::IN_QKV_NORM_WEIGHT, QKVLinearSplitTensorIdx::IN_QKV_NORM_BIAS,
            QKVLinearSplitTensorIdx::IN_QKV_NORM_NEW_WEIGHT, QKVLinearSplitTensorIdx::IN_QKV_NORM_NEW_BIAS,
            QKVLinearSplitTensorIdx::IN_QKV_WEIGHT_1,
            QKVLinearSplitTensorIdx::IN_QKV_SCALE_1, QKVLinearSplitTensorIdx::IN_QKV_OFFSET_1,
            QKVLinearSplitTensorIdx::IN_QKV_DESCALE_1, QKVLinearSplitTensorIdx::IN_QKV_BIAS_1
        };
        kNormLinearNode.outTensorIds = {QKVLinearSplitTensorIdx::OUT_K};

        atb::Node &vNormLinearNode = opGraph.nodes.at(nodeId++);
        atb_speed::common::NormLinearParam<NormParamType> vNormLinearParam;
        vNormLinearParam.isAntiOutlier = param.isAntiOutlier;
        if (param.packQuantType == atb_speed::common::ALL_W8A16) {
            vNormLinearParam.fusionLinearParam.quantType = W8A16;
        } else {
            vNormLinearParam.fusionLinearParam.quantType \
                = param.layerLinearQuantType[2] == atb_speed::common::LinearType::FP ? NO_QUANT : NORM_QUANT_LINEAR_DEQUANT;
        }
        vNormLinearParam.fusionLinearParam.isBF16 = param.isBF16;
        vNormLinearParam.fusionLinearParam.hasBias = param.qkvHasBias;
        vNormLinearParam.normParamType = param.normParamType;
        vNormLinearParam.normQuantParamType = param.normQuantParamType;
        NormLinear<NormParamType>(vNormLinearParam, &vNormLinearNode.operation);
        vNormLinearNode.inTensorIds = {
            QKVLinearSplitTensorIdx::IN_QKV_INPUT, QKVLinearSplitTensorIdx::IN_QKV_NORM_WEIGHT, QKVLinearSplitTensorIdx::IN_QKV_NORM_BIAS,
            QKVLinearSplitTensorIdx::IN_QKV_NORM_NEW_WEIGHT, QKVLinearSplitTensorIdx::IN_QKV_NORM_NEW_BIAS,
            QKVLinearSplitTensorIdx::IN_QKV_WEIGHT_2,
            QKVLinearSplitTensorIdx::IN_QKV_SCALE_2, QKVLinearSplitTensorIdx::IN_QKV_OFFSET_2,
            QKVLinearSplitTensorIdx::IN_QKV_DESCALE_2, QKVLinearSplitTensorIdx::IN_QKV_BIAS_2
        };
        vNormLinearNode.outTensorIds = {QKVLinearSplitTensorIdx::OUT_V};
    }

    opGraph.inferShapeFunc = [=]
                (const atb::SVector<atb::TensorDesc> &inTensorDescs, atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        outTensorDescs.at(0).shape.dims[inTensorDescs.at(0).shape.dimNum - 1] \
            = param.selfAttentionParam.headNum * param.headDim;

        outTensorDescs.at(1) = outTensorDescs.at(0);
        outTensorDescs.at(1).shape.dims[inTensorDescs.at(0).shape.dimNum - 1] \
            = param.selfAttentionParam.kvHeadNum * param.headDim;

        outTensorDescs.at(2) = outTensorDescs.at(1);
        return atb::NO_ERROR;
    };

    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
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

template <typename NormParamType>
atb::Status SelfAttention(const FusionAttentionParam<NormParamType> &param, atb::Operation **operation)
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
        reshapeAndCacheNode.outTensorIds = {
            SelfAttentionTensorIdx::IN_SELF_ATTENTION_K_CACHE,
            SelfAttentionTensorIdx::IN_SELF_ATTENTION_V_CACHE
        };
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
            = param.selfAttentionParam.headNum * param.headDim;
        return atb::NO_ERROR;
    };

    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}

enum AttentionTensorIdx : uint32_t {
    IN_INPUT = 0,
    IN_NORM_WEIGHT,
    IN_NORM_BIAS,
    IN_NORM_NEW_WEIGHT,
    IN_NORM_NEW_BIAS,
    IN_WEIGHT_0,  // q or mixed qkv
    IN_SCALE_0,
    IN_OFFSET_0,
    IN_DESCALE_0,
    IN_BIAS_0,
    IN_WEIGHT_1,  // k
    IN_SCALE_1,
    IN_OFFSET_1,
    IN_DESCALE_1,
    IN_BIAS_1,
    IN_WEIGHT_2,  // v
    IN_SCALE_2,
    IN_OFFSET_2,
    IN_DESCALE_2,
    IN_BIAS_2,
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
    IN_BIAS_OUT,
    OUT_ATTENTION,
    // shape: PA: [seqLen, numAttentionHeadsPerRank * hiddenSizePerAttentionHead]
    // shape: FA: [batchSize, seqLen, numAttentionHeadsPerRank * hiddenSizePerAttentionHead]
    INTERMIDATE_Q,
    // shape: PA: [seqLen, numKeyValueHeadsPerRank * hiddenSizePerAttentionHead]
    // shape: FA: [batchSize, seqLen, numKeyValueHeadsPerRank * hiddenSizePerAttentionHead]
    INTERMIDATE_K,
    INTERMIDATE_V,  // same as INTERMIDATE_K
    INTERMIDATE_SELF_ATTENTION,  // shape: PA: [seqLen, numAttentionHeadsPerRank * hiddenSizePerAttentionHead]
    // shape: PA: [unpadSeqLen, numAttentionHeadsPerRank * hiddenSizePerAttentionHead]
    // shape: FA: [batchSize * seqLen, numAttentionHeadsPerRank * hiddenSizePerAttentionHead]
    INTERMIDATE_POSITION_EMBED_Q,
    INTERMIDATE_POSITION_EMBED_K,  // same as INTERMIDATE_POSITION_EMBED_Q
};

static const uint64_t ATTENTION_IN_TENSOR_COUNT = 35;
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
    for (uint64_t i = 0; i < oldShape.dimNum; i++) {
        newShape.dims[i + 1] = oldShape.dims[i];
    }
}

template <typename NormParamType>
atb::Status Attention(const FusionAttentionParam<NormParamType> &param, atb::Operation **operation)
{
    std::shared_ptr<int64_t> batchSizePtr = std::make_shared<int64_t>(0);
    atb::GraphParam opGraph;
    opGraph.name = "Attention";
    opGraph.inTensorNum = ATTENTION_IN_TENSOR_COUNT;
    opGraph.outTensorNum = ATTENTION_OUT_TENSOR_COUNT;
    if (param.needRope) {
        opGraph.internalTensorNum = ATTENTION_INTERMEDIATE_TENSOR_COUNT;
        opGraph.nodes.resize(ATTENTION_NODE_COUNT);
    } else {
        opGraph.internalTensorNum = ATTENTION_INTERMEDIATE_TENSOR_COUNT - 2;
        opGraph.nodes.resize(ATTENTION_NODE_COUNT - 1);
    }

    size_t nodeId = 0;

    atb::Node &qkvLinearSplitNode = opGraph.nodes.at(nodeId++);
    QKVLinearSplit(param, &qkvLinearSplitNode.operation);
    qkvLinearSplitNode.inTensorIds = {
        AttentionTensorIdx::IN_INPUT, AttentionTensorIdx::IN_NORM_WEIGHT, AttentionTensorIdx::IN_NORM_BIAS,
        AttentionTensorIdx::IN_NORM_NEW_WEIGHT,  AttentionTensorIdx::IN_NORM_NEW_BIAS,
        AttentionTensorIdx::IN_WEIGHT_0, AttentionTensorIdx::IN_SCALE_0, AttentionTensorIdx::IN_OFFSET_0,
        AttentionTensorIdx::IN_DESCALE_0, AttentionTensorIdx::IN_BIAS_0,
        AttentionTensorIdx::IN_WEIGHT_1, AttentionTensorIdx::IN_SCALE_1, AttentionTensorIdx::IN_OFFSET_1,
        AttentionTensorIdx::IN_DESCALE_1, AttentionTensorIdx::IN_BIAS_1,
        AttentionTensorIdx::IN_WEIGHT_2, AttentionTensorIdx::IN_SCALE_2, AttentionTensorIdx::IN_OFFSET_2,
        AttentionTensorIdx::IN_DESCALE_2, AttentionTensorIdx::IN_BIAS_2
    };
    qkvLinearSplitNode.outTensorIds = {
        AttentionTensorIdx::INTERMIDATE_Q, AttentionTensorIdx::INTERMIDATE_K, AttentionTensorIdx::INTERMIDATE_V
    };
    if (param.needRope) {
        atb::Node &ropeNode = opGraph.nodes.at(nodeId++);
        RotaryPositionEmbeddingParam ropeParam;
        ropeParam.isHalfRotary = param.isHalfRotary;
        ropeParam.isFA = param.isFA;
        ropeParam.headDim = param.headDim;
        ropeParam.headNum = param.selfAttentionParam.headNum;
        ropeParam.kvHeadNum = param.selfAttentionParam.kvHeadNum;
        ropeParam.rotaryCoeff = param.rotaryCoeff;

        RotaryPositionEmbedding(ropeParam, &ropeNode.operation);

        ropeNode.inTensorIds = {
            AttentionTensorIdx::INTERMIDATE_Q, AttentionTensorIdx::INTERMIDATE_K, AttentionTensorIdx::IN_COS_TABLE,
            AttentionTensorIdx::IN_SIN_TABLE, AttentionTensorIdx::IN_SEQ_LEN
        };
        ropeNode.outTensorIds = {
            AttentionTensorIdx::INTERMIDATE_POSITION_EMBED_Q, AttentionTensorIdx::INTERMIDATE_POSITION_EMBED_K
        };
        if (param.isFA && !param.isHalfRotary) {
            ropeNode.inTensorReshapeFuncs.resize(ropeNode.inTensorIds.size());
            ropeNode.inTensorReshapeFuncs.at(0) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
                *batchSizePtr = oldShape.dims[0];
                squeezeBatchSize(oldShape, newShape);
            };
            ropeNode.inTensorReshapeFuncs.at(1) = &squeezeBatchSize;
        }
    }

    atb::Node &selfAttentionNode = opGraph.nodes.at(nodeId++);
    SelfAttention(param, &selfAttentionNode.operation);
    selfAttentionNode.inTensorIds = {
        param.needRope ? AttentionTensorIdx::INTERMIDATE_POSITION_EMBED_Q : AttentionTensorIdx::INTERMIDATE_Q,
        param.needRope ? AttentionTensorIdx::INTERMIDATE_POSITION_EMBED_K : AttentionTensorIdx::INTERMIDATE_K,
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
            if (*batchSizePtr == 0) {
                *batchSizePtr = oldShape.dims[0];
            }
            unsqueezeByHeadNumAndBatchSize(oldShape, newShape, (*batchSizePtr),
                                           param.selfAttentionParam.headNum, param.headDim);
        };
        selfAttentionNode.inTensorReshapeFuncs[1] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            unsqueezeByHeadNumAndBatchSize(oldShape, newShape, (*batchSizePtr),
                                           param.selfAttentionParam.kvHeadNum, param.headDim);
        };
        selfAttentionNode.inTensorReshapeFuncs[2] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            unsqueezeByHeadNumAndBatchSize(oldShape, newShape, (*batchSizePtr),
                                           param.selfAttentionParam.kvHeadNum, param.headDim);
        };
        // Unsqueeze layer axis of kv cache
        selfAttentionNode.inTensorReshapeFuncs.at(3) = &unSqueezeLayerAxis;
        selfAttentionNode.inTensorReshapeFuncs.at(4) = &unSqueezeLayerAxis;
    } else {
        selfAttentionNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            unsqueezeByHeadNum(oldShape, newShape,
                               param.selfAttentionParam.headNum, param.headDim);
        };
        selfAttentionNode.inTensorReshapeFuncs[1] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            unsqueezeByHeadNum(oldShape, newShape,
                               param.selfAttentionParam.kvHeadNum, param.headDim);
        };
        selfAttentionNode.inTensorReshapeFuncs[2] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            unsqueezeByHeadNum(oldShape, newShape,
                               param.selfAttentionParam.kvHeadNum, param.headDim);
        };
    }

    atb::Node &selfOutLinearParallelNode = opGraph.nodes.at(nodeId++);
    atb_speed::common::LinearParallelParam selfOutLinearParam;
    selfOutLinearParam.parallelType = atb_speed::common::ROW_PARALLEL;
    if (param.packQuantType == atb_speed::common::ALL_W8A16) {
        selfOutLinearParam.fusionLinearParam.quantType = W8A16;
    } else {
        selfOutLinearParam.fusionLinearParam.quantType \
            = param.layerLinearQuantType[3] == atb_speed::common::LinearType::FP ? \
            atb_speed::common::LinearQuantType::NO_QUANT : atb_speed::common::LinearQuantType::LINEAR_QUANT;
    }
    selfOutLinearParam.biasAfterSync = param.selfOutLinearTensorParallelInfo.worldSize > 1 \
        && selfOutLinearParam.fusionLinearParam.quantType == atb_speed::common::LinearQuantType::NO_QUANT \
        && param.selfAttnHasBias;
    selfOutLinearParam.fusionLinearParam.isBF16 = param.isBF16;
    selfOutLinearParam.fusionLinearParam.hasBias = param.selfAttnHasBias && !selfOutLinearParam.biasAfterSync;
    selfOutLinearParam.tensorParallelInfo = param.selfOutLinearTensorParallelInfo;
    selfOutLinearParam.supportLcoc = param.supportLcoc;
    LinearParallel(selfOutLinearParam, &selfOutLinearParallelNode.operation);
    selfOutLinearParallelNode.inTensorIds = {
        AttentionTensorIdx::INTERMIDATE_SELF_ATTENTION, AttentionTensorIdx::IN_WEIGHT_OUT,
        AttentionTensorIdx::IN_SCALE_OUT, AttentionTensorIdx::IN_OFFSET_OUT,
        AttentionTensorIdx::IN_DESCALE_OUT, AttentionTensorIdx::IN_BIAS_OUT
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

template atb::Status QKVLinearSplit(const FusionAttentionParam<atb::infer::RmsNormParam> &param, atb::Operation **operation);
template atb::Status SelfAttention(const FusionAttentionParam<atb::infer::RmsNormParam> &param, atb::Operation **operation);
template atb::Status Attention(const FusionAttentionParam<atb::infer::RmsNormParam> &param, atb::Operation **operation);

template atb::Status QKVLinearSplit(const FusionAttentionParam<atb::infer::LayerNormParam> &param, atb::Operation **operation);
template atb::Status SelfAttention(const FusionAttentionParam<atb::infer::LayerNormParam> &param, atb::Operation **operation);
template atb::Status Attention(const FusionAttentionParam<atb::infer::LayerNormParam> &param, atb::Operation **operation);

} // namespace common
} // namespace atb_speed