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
#include "paged_attention_layer.h"
#include "layers/mlp_gate_v2.h"
#include "layers/parallel_layer_v2.h"

namespace atb_speed {
namespace star_coder {
static const uint64_t IN_TENSOR_COUNT = 20;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 11;
static const uint64_t NODE_COUNT = 12;
static const uint64_t LAYER_NORM_AXIS_COUNT = 2;
static const uint64_t ATTENTION_DIM_3 = 3;
static const uint64_t KV_SPLIT_DIM = 2;
static const uint64_t KV_SPLIT_NUM = 2;

enum PALayerTensorId : int {
    IN_HIDDENSTATES = 0,
    IN_LN_1_WEIGTH,
    IN_LN_1_BIAS,
    IN_C_ATTN_WEIGHT,
    IN_C_ATTN_BIAS,
    IN_C_PROJ_WEIGHT,
    IN_C_PROJ_BIAS,
    IN_LN_2_WEIGHT,
    IN_LN_2_BIAS,
    IN_MLP_FC_WEIGHT,
    IN_MLP_FC_BIAS,
    IN_MLP_PROJ_WEIGHT,
    IN_MLP_PROJ_BIAS,
    IN_ATTENTIONMASK,
    IN_BLOCK_TABLES,
    IN_SLOTS,
    IN_INPUT_LENGTHS,   // 16
    IN_HOLDER,          // HOLDER
    IN_PASTK,
    IN_PASTV,
    OUT_LAYEROUT,
    INTERMIDATE_INPUTNORMOUT,
    INTERMIDATE_QKV,
    INTERNAL_Q,
    INTERNAL_KV,
    INTERNAL_K,
    INTERNAL_V,
    INTERMIDATE_SELFOUT,
    INTERMIDATE_SELF_LINEAR_OUT,
    INTERMIDATE_SELF_RESIDUAL_ADD_OUT,
    INTERMEDIATE_LN_2_NORM_OUT,
    INTERMIDATE_MLP_OUT,
};

// [bs, seq, kv_hidden_size] -> [n_tokens=bs*seq, head_Num, head_dim] -> [n_tokens, 1, 128]
void reshapeHeads(const atb::Dims &oldShape, atb::Dims &newShape, int headNum)
{
    newShape.dimNum = ATTENTION_DIM_3;                      // dimNum: 3
    newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1]; // 0 dim: n tokens
    newShape.dims[1] = headNum;                             // 1 dim: head num
    newShape.dims[2] = oldShape.dims[2] / headNum;          // 2 dim: head size
}

atb::Status PALayer(const PALayerParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);
    if (param.isPrefill) {
        opGraph.name = "StarCoder_Prefill_PA_layer";
    } else {
        opGraph.name = "StarCoder_Decoder_PA_layer";
    }

    size_t nodeId = 0;
    atb::Node &inputLayerNormNode = opGraph.nodes.at(nodeId++);
    atb::Node &cAttnLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &qPassSliceNode = opGraph.nodes.at(nodeId++);
    atb::Node &kVPassSliceNode = opGraph.nodes.at(nodeId++);
    atb::Node &splitKVNode = opGraph.nodes.at(nodeId++);
    atb::Node &reshapeAndCacheNode = opGraph.nodes.at(nodeId++);
    atb::Node &attentionNode = opGraph.nodes.at(nodeId++);
    atb::Node &cProjLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfResidualAddNode = opGraph.nodes.at(nodeId++);
    atb::Node &postAttnLayerNormNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpNode = opGraph.nodes.at(nodeId++);
    atb::Node &attnResidualAddNode = opGraph.nodes.at(nodeId++);

    atb::SVector<int64_t> sliceOffsetQ = {0, 0, 0};
    atb::SVector<int64_t> sliceSizeQ = {-1, -1, param.headNum * param.dk};
    atb::SVector<int64_t> sliceOffsetKV = {0, 0, param.headNum * param.dk};
    atb::SVector<int64_t> sliceSizeKV = {-1, -1, 2 * param.dk};

    // [n_tokens, hidden_size + k_dim + v_dim] = [n_tokens,6144 + 128 + 128] = [n_tokens, 6400]
    atb::infer::LayerNormParam layerNormParam;
    layerNormParam.layerType = atb::infer::LayerNormParam::LayerNormType::LAYER_NORM_NORM;
    layerNormParam.normParam.epsilon = param.layerNormEps;
    layerNormParam.normParam.beginNormAxis = LAYER_NORM_AXIS_COUNT;
    layerNormParam.normParam.beginParamsAxis = LAYER_NORM_AXIS_COUNT;
    CreateOperation(layerNormParam, &inputLayerNormNode.operation);
    inputLayerNormNode.inTensorIds = {IN_HIDDENSTATES, IN_LN_1_WEIGTH, IN_LN_1_BIAS};
    inputLayerNormNode.outTensorIds = {INTERMIDATE_INPUTNORMOUT};

    atb::infer::LinearParam linearBiasParam = { false, false, true };
    CreateOperation(linearBiasParam, &cAttnLinearNode.operation);
    cAttnLinearNode.inTensorIds = { INTERMIDATE_INPUTNORMOUT, IN_C_ATTN_WEIGHT, IN_C_ATTN_BIAS };
    cAttnLinearNode.outTensorIds = { INTERMIDATE_QKV };

    atb::infer::SliceParam slicePassParam;
    slicePassParam.offsets = sliceOffsetQ;
    slicePassParam.size = sliceSizeQ;
    CreateOperation(slicePassParam, &qPassSliceNode.operation);
    qPassSliceNode.inTensorIds = {INTERMIDATE_QKV};
    qPassSliceNode.outTensorIds = {INTERNAL_Q};

    atb::infer::SliceParam slicePassKVParam;
    slicePassKVParam.offsets = sliceOffsetKV;
    slicePassKVParam.size = sliceSizeKV;
    CreateOperation(slicePassKVParam, &kVPassSliceNode.operation);
    kVPassSliceNode.inTensorIds = {INTERMIDATE_QKV};
    kVPassSliceNode.outTensorIds = {INTERNAL_KV};

    atb::infer::SplitParam splitParam;
    splitParam.splitDim = KV_SPLIT_DIM;
    splitParam.splitNum = KV_SPLIT_NUM;
    CreateOperation(splitParam, &splitKVNode.operation);
    splitKVNode.inTensorIds = {INTERNAL_KV};
    splitKVNode.outTensorIds = {INTERNAL_K, INTERNAL_V};

    atb::infer::ReshapeAndCacheParam reshapeCacheParm;
    CreateOperation(reshapeCacheParm, &reshapeAndCacheNode.operation);
    reshapeAndCacheNode.inTensorIds = {INTERNAL_K, INTERNAL_V,
                                       IN_PASTK, IN_PASTV, IN_SLOTS};
    reshapeAndCacheNode.outTensorIds = {};
    reshapeAndCacheNode.inTensorReshapeFuncs.resize(reshapeAndCacheNode.inTensorIds.size());
    reshapeAndCacheNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        reshapeHeads(oldShape, newShape, param.kvHead);
    };
    reshapeAndCacheNode.inTensorReshapeFuncs[1] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        reshapeHeads(oldShape, newShape, param.kvHead);
    };

    if (param.isPrefill) {
        atb::infer::SelfAttentionParam faEnParam;
        faEnParam.headDim = param.dk;               // 128
        faEnParam.headNum = param.headNum;          // 48
        faEnParam.qScale = 1.0 / sqrt(param.dk);
        faEnParam.kvHeadNum = param.kvHead;         // 1
        faEnParam.isEncoder = true;
        faEnParam.isFusion = true;
        CreateOperation(faEnParam, &attentionNode.operation);
        attentionNode.inTensorIds = {INTERNAL_Q, INTERNAL_K, INTERNAL_V, IN_ATTENTIONMASK, IN_INPUT_LENGTHS};
        attentionNode.outTensorIds = {INTERMIDATE_SELFOUT};
        attentionNode.inTensorReshapeFuncs.resize(attentionNode.inTensorIds.size());
        attentionNode.inTensorReshapeFuncs.at(0) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            reshapeHeads(oldShape, newShape, faEnParam.headNum);
        };
        attentionNode.inTensorReshapeFuncs.at(1) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            reshapeHeads(oldShape, newShape, faEnParam.kvHeadNum);
        };
        attentionNode.inTensorReshapeFuncs.at(2) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            reshapeHeads(oldShape, newShape, faEnParam.kvHeadNum);
        };
    } else {
        atb::infer::PagedAttentionParam paDeParam;
        paDeParam.headNum = param.headNum;
        paDeParam.qkScale = 1.0 / sqrt(param.dk);
        paDeParam.kvHeadNum = param.kvHead;
        CreateOperation(paDeParam, &attentionNode.operation);
        attentionNode.inTensorIds = {INTERNAL_Q, IN_PASTK, IN_PASTV, IN_BLOCK_TABLES, IN_INPUT_LENGTHS};
        attentionNode.outTensorIds = {INTERMIDATE_SELFOUT};
        attentionNode.inTensorReshapeFuncs.resize(attentionNode.inTensorIds.size());
        attentionNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            reshapeHeads(oldShape, newShape, paDeParam.headNum);
        };
    }

    atb_speed::common::ParallelParamV2 cProjLinearParam;
    cProjLinearParam.commParam.rank = param.rank;
    cProjLinearParam.commParam.rankSize = param.rankSize;
    cProjLinearParam.commParam.backend = param.backend;
    cProjLinearParam.isBias = true;
    atb_speed::common::RowParallelLinearV2(cProjLinearParam, &cProjLinearNode.operation);
    cProjLinearNode.inTensorIds = {INTERMIDATE_SELFOUT, IN_C_PROJ_WEIGHT, IN_C_PROJ_BIAS,
                                    IN_HOLDER, IN_HOLDER, IN_HOLDER, IN_HOLDER};
    cProjLinearNode.outTensorIds = {INTERMIDATE_SELF_LINEAR_OUT};
    cProjLinearNode.inTensorReshapeFuncs.resize(cProjLinearNode.inTensorIds.size());
    cProjLinearNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = ATTENTION_DIM_3;                      // dim num
        newShape.dims[0] = 1;
        newShape.dims[1] = oldShape.dims[0];                    // 1: dim 1, n tokens
        newShape.dims[2] = oldShape.dims[1] * oldShape.dims[2]; // 2: hidden size: old 1, head num , old 2 head size
    };

    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CreateOperation(addParam, &selfResidualAddNode.operation);
    selfResidualAddNode.inTensorIds = { IN_HIDDENSTATES, INTERMIDATE_SELF_LINEAR_OUT };
    selfResidualAddNode.outTensorIds = { INTERMIDATE_SELF_RESIDUAL_ADD_OUT };

    CreateOperation(layerNormParam, &postAttnLayerNormNode.operation);
    postAttnLayerNormNode.inTensorIds = {INTERMIDATE_SELF_RESIDUAL_ADD_OUT, IN_LN_2_WEIGHT, IN_LN_2_BIAS};
    postAttnLayerNormNode.outTensorIds = {INTERMEDIATE_LN_2_NORM_OUT};

    atb_speed::common::MlpGateParamV2 mlpParam;
    mlpParam.commDownParam.rank = param.rank;
    mlpParam.commDownParam.rankSize = param.rankSize;
    mlpParam.commDownParam.backend = param.backend;
    mlpParam.activationType = atb::infer::ActivationType::ACTIVATION_GELU;
    mlpParam.transposeB = false;
    mlpParam.isBias = true;
    mlpParam.isPack = false;
    mlpParam.noGate = true;
    atb_speed::common::MlpGateLayerV2(mlpParam, &mlpNode.operation);
    mlpNode.inTensorIds = {INTERMEDIATE_LN_2_NORM_OUT,
                            IN_MLP_FC_WEIGHT, IN_HOLDER, IN_MLP_PROJ_WEIGHT,
                            IN_HOLDER, IN_HOLDER, IN_HOLDER,
                            IN_MLP_FC_BIAS, IN_HOLDER, IN_MLP_PROJ_BIAS,
                            IN_HOLDER, IN_HOLDER, IN_HOLDER,
                            IN_HOLDER, IN_HOLDER, IN_HOLDER,
                            IN_HOLDER, IN_HOLDER, IN_HOLDER};
    mlpNode.outTensorIds = {INTERMIDATE_MLP_OUT};

    CreateOperation(addParam, &attnResidualAddNode.operation);
    attnResidualAddNode.inTensorIds = {INTERMIDATE_SELF_RESIDUAL_ADD_OUT, INTERMIDATE_MLP_OUT};
    attnResidualAddNode.outTensorIds = {OUT_LAYEROUT};

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
        atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
    };

    atb::CreateOperation(opGraph, operation);
    return atb::NO_ERROR;
}

StarCoderPAFlashAttentionHostBinder::StarCoderPAFlashAttentionHostBinder() {}

StarCoderPAFlashAttentionHostBinder::~StarCoderPAFlashAttentionHostBinder() {}

void StarCoderPAFlashAttentionHostBinder::ParseParam(const nlohmann::json &paramJson)
{
    seqLen_.clear();
    for (auto item : paramJson["seqLen"]) {
        seqLen_.push_back(item.get<int32_t>());
    }
}

void StarCoderPAFlashAttentionHostBinder::BindTensor(atb::VariantPack &variantPack)
{
    variantPack.inTensors.at(IN_INPUT_LENGTHS).hostData = seqLen_.data();
}

} // namespace star_coder
} // namespace atb_speed