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
#include "paged_attention_quant_layer.h"
#include "layers/mlp_gate_v2.h"
#include "layers/parallel_layer_v2.h"

namespace atb_speed {
namespace star_coder {
static const uint64_t IN_TENSOR_COUNT = 27;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 11;
static const uint64_t NODE_COUNT = 12;
static const uint64_t LAYER_NORM_AXIS_COUNT = 2;
static const uint64_t ATTENTION_DIM_3 = 3;
static const uint64_t KV_SPLIT_DIM = 2;
static const uint64_t KV_SPLIT_NUM = 2;

enum PAQuantLayerTensorId : int {
    IN_HIDDENSTATES = 0,
    
    IN_LAYERNORM_1_WEIGTH,
    IN_LAYERNORM_1_BIAS,
    IN_QKV_WEIGHT,
    IN_QKV_BIAS,
    IN_QKV_DEQSCALE,
    IN_SELFOUT_LINEAR_WEIGHT,
    IN_SELFOUT_LINEAR_BIAS,
    IN_SELFOUT_LINEAR_DEQSCALE,
    IN_LAYERNORM_2_WEIGHT,
    IN_LAYERNORM_2_BIAS,
    IN_MLP_UP_WEIGHT,
    IN_MLP_UP_BIAS,
    IN_MLP_UP_DEQSCALE,
    IN_MLP_GATE_WEIGHT,
    IN_MLP_GATE_DEQSCALE,
    IN_MLP_GATE_BIAS,
    IN_MLP_DOWN_WEIGHT,
    IN_MLP_DOWN_BIAS,
    IN_MLP_DOWN_DEQSCALE,

    IN_ATTENTIONMASK,
    IN_BLOCK_TABLES,
    IN_SLOTS,
    IN_INPUT_LENGTHS,
    IN_HOLDER,
    IN_K_CACHE,
    IN_V_CACHE,

    OUT_LAYEROUT,

    INTERMEDIATE_INPUTNORM_OUT,
    INTERMEDIATE_INPUTNORM_OUT_QUANT,
    INTERMEDIATE_QKV,
    INTERMEDIATE_Q,
    INTERMEDIATE_KV,
    INTERMEDIATE_K,
    INTERMEDIATE_V,
    INTERMEDIATE_SELF_OUT,
    INTERMEDIATE_SELF_LINEAR_OUT,
    INTERMEDIATE_SELF_RESIDUAL_ADD_OUT,
    INTERMEDIATE_LAYERNORM_2_OUT,
    INTERMEDIATE_MLP_OUT,
};

// [bs, seq, kv_hidden_size] -> [n_tokens=bs*seq, head_Num, head_dim] -> [n_tokens, 1, 128]
void reshapeQuantHeads(const atb::Dims &oldShape, atb::Dims &newShape, int headNum)
{
    newShape.dimNum = ATTENTION_DIM_3;
    newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
    newShape.dims[1] = headNum;
    newShape.dims[2] = oldShape.dims[2] / headNum; // 1 dim: head size
}

atb::Status PAQuantLayer(const PAQuantLayerParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);
    if (param.isPrefill) {
        opGraph.name = "StarCoder_Prefill_PA_Quant_layer";
    } else {
        opGraph.name = "StarCoder_Decoder_PA_Quant_layer";
    }

    size_t nodeId = 0;
    atb::Node &inputLayerNormNode = opGraph.nodes.at(nodeId++);
    atb::Node &qkvLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &qPassSliceNode = opGraph.nodes.at(nodeId++);
    atb::Node &kVPassSliceNode = opGraph.nodes.at(nodeId++);
    atb::Node &splitKVNode = opGraph.nodes.at(nodeId++);
    atb::Node &reshapeAndCacheNode = opGraph.nodes.at(nodeId++);
    atb::Node &attentionNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfOutLinearNode = opGraph.nodes.at(nodeId++);
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
    layerNormParam.normParam.quantType = atb::infer::QUANT_INT8;
    CreateOperation(layerNormParam, &inputLayerNormNode.operation);
    inputLayerNormNode.inTensorIds = {IN_HIDDENSTATES, IN_LAYERNORM_1_WEIGTH, IN_LAYERNORM_1_BIAS};
    inputLayerNormNode.outTensorIds = {INTERMEDIATE_INPUTNORM_OUT, INTERMEDIATE_INPUTNORM_OUT_QUANT};

    atb::infer::LinearParam qkvLinearParam;
    qkvLinearParam.linearType = atb::infer::LinearType::LINEAR_INT8INT8_INT32_FP16;
    CreateOperation(qkvLinearParam, &qkvLinearNode.operation);
    qkvLinearNode.inTensorIds = {INTERMEDIATE_INPUTNORM_OUT_QUANT, IN_QKV_WEIGHT, IN_QKV_BIAS, IN_QKV_DEQSCALE};
    qkvLinearNode.outTensorIds = {INTERMEDIATE_QKV};

    atb::infer::SliceParam slicePassParam;
    slicePassParam.offsets = sliceOffsetQ;
    slicePassParam.size = sliceSizeQ;
    CreateOperation(slicePassParam, &qPassSliceNode.operation);
    qPassSliceNode.inTensorIds = {INTERMEDIATE_QKV};
    qPassSliceNode.outTensorIds = {INTERMEDIATE_Q};

    atb::infer::SliceParam slicePassKVParam;
    slicePassKVParam.offsets = sliceOffsetKV;
    slicePassKVParam.size = sliceSizeKV;
    CreateOperation(slicePassKVParam, &kVPassSliceNode.operation);
    kVPassSliceNode.inTensorIds = {INTERMEDIATE_QKV};
    kVPassSliceNode.outTensorIds = {INTERMEDIATE_KV};

    atb::infer::SplitParam splitParam;
    splitParam.splitDim = KV_SPLIT_DIM;
    splitParam.splitNum = KV_SPLIT_NUM;
    CreateOperation(splitParam, &splitKVNode.operation);
    splitKVNode.inTensorIds = {INTERMEDIATE_KV};
    splitKVNode.outTensorIds = {INTERMEDIATE_K, INTERMEDIATE_V};

    atb::infer::ReshapeAndCacheParam reshapeCacheParm;
    CreateOperation(reshapeCacheParm, &reshapeAndCacheNode.operation);
    reshapeAndCacheNode.inTensorIds = {INTERMEDIATE_K, INTERMEDIATE_V, IN_K_CACHE, IN_V_CACHE, IN_SLOTS};
    reshapeAndCacheNode.outTensorIds = {IN_K_CACHE, IN_V_CACHE};
    reshapeAndCacheNode.inTensorReshapeFuncs.resize(reshapeAndCacheNode.inTensorIds.size());
    reshapeAndCacheNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        reshapeQuantHeads(oldShape, newShape, param.kvHead);
    };
    reshapeAndCacheNode.inTensorReshapeFuncs[1] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        reshapeQuantHeads(oldShape, newShape, param.kvHead);
    };

    if (param.isPrefill) {
        atb::infer::SelfAttentionParam faEnParam;
        faEnParam.headNum = param.headNum;
        faEnParam.qScale = 1.0 / sqrt(param.dk);
        faEnParam.kvHeadNum = param.kvHead;
        faEnParam.calcType = atb::infer::SelfAttentionParam::PA_ENCODER;
        CreateOperation(faEnParam, &attentionNode.operation);
        attentionNode.inTensorIds = {INTERMEDIATE_Q, INTERMEDIATE_K, INTERMEDIATE_V, IN_ATTENTIONMASK, IN_INPUT_LENGTHS};
        attentionNode.outTensorIds = {INTERMEDIATE_SELF_OUT};
        attentionNode.inTensorReshapeFuncs.resize(attentionNode.inTensorIds.size());
        attentionNode.inTensorReshapeFuncs.at(0) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            reshapeQuantHeads(oldShape, newShape, faEnParam.headNum);
        };
        attentionNode.inTensorReshapeFuncs.at(1) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            reshapeQuantHeads(oldShape, newShape, faEnParam.kvHeadNum);
        };
        attentionNode.inTensorReshapeFuncs.at(2) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            reshapeQuantHeads(oldShape, newShape, faEnParam.kvHeadNum);
        };
    } else {
        atb::infer::PagedAttentionParam paDeParam;
        paDeParam.headNum = param.headNum;
        paDeParam.qkScale = 1.0 / sqrt(param.dk);
        paDeParam.kvHeadNum = param.kvHead;
        CreateOperation(paDeParam, &attentionNode.operation);
        attentionNode.inTensorIds = {INTERMEDIATE_Q, IN_K_CACHE, IN_V_CACHE, IN_BLOCK_TABLES, IN_INPUT_LENGTHS}; // 增量直接从kv cache里取值
        attentionNode.outTensorIds = {INTERMEDIATE_SELF_OUT};
        attentionNode.inTensorReshapeFuncs.resize(attentionNode.inTensorIds.size());
        attentionNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            reshapeQuantHeads(oldShape, newShape, paDeParam.headNum);
        };
    }

    atb_speed::common::ParallelParamV2 selfOutLinearParam;
    selfOutLinearParam.commParam.rank = param.rank;
    selfOutLinearParam.commParam.rankSize = param.rankSize;
    selfOutLinearParam.commParam.backend = param.backend;
    selfOutLinearParam.isBias = true;
    atb_speed::common::RowParallelLinearV2(selfOutLinearParam, &selfOutLinearNode.operation);
    selfOutLinearNode.inTensorIds = {INTERMEDIATE_SELF_OUT, IN_SELFOUT_LINEAR_WEIGHT, IN_SELFOUT_LINEAR_BIAS,
                                    IN_SELFOUT_LINEAR_DEQSCALE, IN_HOLDER, IN_HOLDER, IN_HOLDER};
    selfOutLinearNode.outTensorIds = {INTERMEDIATE_SELF_LINEAR_OUT};
    selfOutLinearNode.inTensorReshapeFuncs.resize(selfOutLinearNode.inTensorIds.size());
    selfOutLinearNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = ATTENTION_DIM_3;                      // dim num
        newShape.dims[0] = 1;
        newShape.dims[1] = oldShape.dims[0];                    // 1: dim 1, n tokens
        newShape.dims[2] = oldShape.dims[1] * oldShape.dims[2]; // 2: hidden size: old 1, head num , old 2 head size
    };

    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CreateOperation(addParam, &selfResidualAddNode.operation);
    selfResidualAddNode.inTensorIds = {IN_HIDDENSTATES, INTERMEDIATE_SELF_LINEAR_OUT};
    selfResidualAddNode.outTensorIds = {INTERMEDIATE_SELF_RESIDUAL_ADD_OUT};

    CreateOperation(layerNormParam, &postAttnLayerNormNode.operation);
    postAttnLayerNormNode.inTensorIds = {INTERMEDIATE_SELF_RESIDUAL_ADD_OUT, IN_LAYERNORM_2_WEIGHT, IN_LAYERNORM_2_BIAS};
    postAttnLayerNormNode.outTensorIds = {INTERMEDIATE_LAYERNORM_2_OUT};

    atb_speed::common::MlpGateParamV2 mlpParam;
    mlpParam.isBias=true;
    mlpParam.isPack=false;
    mlpParam.isQuant=true;
    mlpParam.transposeB=true;
    mlpParam.commDownParam.rank = param.rank;
    mlpParam.commDownParam.rankSize = param.rankSize;
    mlpParam.commDownParam.backend = param.backend;
    mlpParam.quantUpParam.quantType = atb::infer::QUANT_INT8;
    mlpParam.quantUpParam.isQuantOp = false;
    mlpParam.quantGateParam.quantType = atb::infer::QUANT_INT8;
    mlpParam.quantGateParam.isQuantOp = false;
    mlpParam.activationType = atb::infer::ActivationType::ACTIVATION_GELU;
    mlpParam.transposeB = true;
    mlpParam.quantDownParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_QUANT;
    mlpParam.quantDownParam.inputScale = param.mlpOutInputScale;
    mlpParam.quantDownParam.inputOffset = param.mlpOutInputOffset;

    atb_speed::common::MlpGateLayerV2(mlpParam, &mlpNode.operation);
    mlpNode.inTensorIds = {INTERMEDIATE_LAYERNORM_2_OUT,
                            IN_MLP_UP_WEIGHT, IN_MLP_GATE_WEIGHT, IN_MLP_DOWN_WEIGHT,
                            IN_MLP_UP_DEQSCALE, IN_MLP_GATE_DEQSCALE, IN_MLP_DOWN_DEQSCALE,
                            IN_MLP_UP_BIAS, IN_MLP_GATE_BIAS, IN_MLP_DOWN_BIAS,
                            IN_HOLDER, IN_HOLDER, IN_HOLDER,
                            IN_HOLDER, IN_HOLDER, IN_HOLDER,
                            IN_HOLDER, IN_HOLDER, IN_HOLDER};
    mlpNode.outTensorIds = {INTERMEDIATE_MLP_OUT};

    CreateOperation(addParam, &attnResidualAddNode.operation);
    attnResidualAddNode.inTensorIds = {INTERMEDIATE_SELF_RESIDUAL_ADD_OUT, INTERMEDIATE_MLP_OUT};
    attnResidualAddNode.outTensorIds = {OUT_LAYEROUT};

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
        atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
    };

    atb::CreateOperation(opGraph, operation);
    return atb::NO_ERROR;
}

StarCoderPAQuantHostBinder::StarCoderPAQuantHostBinder() {}

StarCoderPAQuantHostBinder::~StarCoderPAQuantHostBinder() {}

void StarCoderPAQuantHostBinder::ParseParam(const nlohmann::json &paramJson)
{
    seqLen_.clear();
    for (auto item : paramJson["seqLen"]) {
        seqLen_.push_back(item.get<int32_t>());
    }
}

void StarCoderPAQuantHostBinder::BindTensor(atb::VariantPack &variantPack)
{
    variantPack.inTensors.at(IN_INPUT_LENGTHS).hostData = seqLen_.data();
}

} // namespace star_coder
} // namespace atb_speed