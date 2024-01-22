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
#include "flash_attention_quant_layer.h"
#include "models/chatglm2/6b/operation/LinearQuantParallel.h"
#include "layers/mlp_gate_v2.h"
#include "layers/parallel_layer_v2.h"
namespace atb_speed {
namespace star_coder {
enum FlashAttentionQuantLayerTensorId : int {
    IN_HIDDENSTATES = 0,
    IN_LAYERNORN_1_WEIGTH,
    IN_LAYERNORN_1_BIAS,

    IN_QKV_WEIGHT,
    IN_QKV_BIAS,
    IN_QKV_DEQSCALE,
    
    IN_SELFOUT_LINEAR_WEIGHT,
    IN_SELFOUT_LINEAR_BIAS,
    IN_SELFOUT_LINEAR_DEQSCALE,

    IN_LAYERNORN_2_WEIGHT,
    IN_LAYERNORN_2_BIAS,

    IN_MLP_UP_WEIGHT,
    IN_MLP_UP_BIAS,
    IN_MLP_UP_DEQSCALE,

    IN_MLP_DOWN_WEIGHT,
    IN_MLP_DOWN_BIAS,
    IN_MLP_DOWN_DEQSCALE,

    IN_ATTENTIONMASK,
    IN_CACHEK,
    IN_CACHEV,
    IN_TOKENOFFSET, // 20
    IN_SEQLEN,      // 21
    IN_HOLDER,
    IN_LAYERID,

    OUT_LAYEROUT,

    INTERMIDATE_INPUTNORM_OUT,
    INTERMIDATE_INPUTNORM_OUT_QUANT,
    INTERMIDATE_QKV,
    INTERNAL_Q,
    INTERNAL_KV,
    INTERNAL_K,
    INTERNAL_V,
    INTERMIDATE_SELFOUT,
    INTERMIDATE_SELF_LINEAR_OUT,
    INTERMIDATE_SELF_RESIDUAL_ADD_OUT,
    INTERMEDIATE_LN_2_NORM_OUT,
    INTERMEDIATE_LN_2_NORM_OUT_QUANT,
    INTERMIDATE_MLP_OUT,
};

static const uint64_t IN_TENSOR_COUNT = 24;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 13;
static const uint64_t NODE_COUNT = 11;
static const uint64_t LAYER_NORM_AXIS_COUNT = 2;
static const uint64_t ATTENTION_DIM_3 = 3;

atb::Status FlashAttentionQuantLayer(const FlashAttentionQuantLayerParam &param, atb::Operation **operation)
{
    ATB_LOG(INFO) << "Enter FlashAttentionQuantLayer";
    atb::GraphParam opGraph;
    opGraph.name = "StarCoder_FAQA_layer";
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    atb::Node &inputLayerNormNode = opGraph.nodes.at(nodeId++);
    atb::Node &qkvLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &qPassSliceNode = opGraph.nodes.at(nodeId++);
    atb::Node &kVPassSliceNode = opGraph.nodes.at(nodeId++);
    atb::Node &splitKVNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfAttentionFaNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfoutLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfResidualAddNode = opGraph.nodes.at(nodeId++);
    atb::Node &postAttnLayerNormNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpNode = opGraph.nodes.at(nodeId++);
    atb::Node &attnResidualAddNode = opGraph.nodes.at(nodeId++);

    atb::SVector<int64_t> sliceOffsetQ = {0, 0, 0};
    atb::SVector<int64_t> sliceSizeQ = {-1, -1, param.headNum * param.dk};
    atb::SVector<int64_t> sliceOffsetKV = {0, 0, param.headNum * param.dk};
    atb::SVector<int64_t> sliceSizeKV = {-1, -1, 2 * param.dk};

    // NORM量化
    atb::infer::LayerNormParam layerNormParam;
    layerNormParam.layerType = atb::infer::LayerNormParam::LAYER_NORM_NORM;
    layerNormParam.normParam.epsilon = param.layerNormEps;
    layerNormParam.normParam.beginNormAxis = LAYER_NORM_AXIS_COUNT;
    layerNormParam.normParam.beginParamsAxis = LAYER_NORM_AXIS_COUNT;
    layerNormParam.normParam.quantType = atb::infer::QUANT_INT8;
    layerNormParam.normParam.quantInputScale = param.qkvInputScale;
    layerNormParam.normParam.quantInputOffset = param.qkvInputOffset;
    CreateOperation(layerNormParam, &inputLayerNormNode.operation);
    inputLayerNormNode.inTensorIds = {IN_HIDDENSTATES, IN_LAYERNORN_1_WEIGTH, IN_LAYERNORN_1_BIAS};
    inputLayerNormNode.outTensorIds = {INTERMIDATE_INPUTNORM_OUT, INTERMIDATE_INPUTNORM_OUT_QUANT};

    // QKV LINEAR量化
    atb::infer::LinearQuantParam qkvLinearParam;
    CreateOperation(qkvLinearParam, &qkvLinearNode.operation);
    qkvLinearNode.inTensorIds = {INTERMIDATE_INPUTNORM_OUT_QUANT, IN_QKV_WEIGHT, IN_QKV_BIAS, IN_QKV_DEQSCALE};
    qkvLinearNode.outTensorIds = {INTERMIDATE_QKV};

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
    splitParam.splitDim = 2; // 最后一维
    splitParam.splitNum = 2; // split to k and v
    CreateOperation(splitParam, &splitKVNode.operation);
    splitKVNode.inTensorIds = {INTERNAL_KV};
    splitKVNode.outTensorIds = {INTERNAL_K, INTERNAL_V};

    atb::infer::SelfAttentionParam selfAttentionParam;
    selfAttentionParam.headDim = param.dk;
    selfAttentionParam.headNum = param.headNum;
    selfAttentionParam.qScale = 1.0 / sqrt(param.dk);
    selfAttentionParam.kvHeadNum = param.kvHead;
    selfAttentionParam.isFusion = true;
    if (param.isEncoder) {
        selfAttentionParam.coderType = atb::infer::SelfAttentionParam::ENCODER;
    } else {
        selfAttentionParam.coderType = atb::infer::SelfAttentionParam::DECODER;
    }
    CreateOperation(selfAttentionParam, &selfAttentionFaNode.operation);
    selfAttentionFaNode.inTensorIds = {INTERNAL_Q,
                                                  INTERNAL_K,
                                                  INTERNAL_V,
                                                  IN_CACHEK,
                                                  IN_CACHEV,
                                                  IN_ATTENTIONMASK,
                                                  IN_TOKENOFFSET,
                                                  IN_SEQLEN,
                                                  IN_LAYERID};
    selfAttentionFaNode.outTensorIds = {INTERMIDATE_SELFOUT};
    selfAttentionFaNode.inTensorReshapeFuncs.resize(selfAttentionFaNode.inTensorIds.size());
    selfAttentionFaNode.inTensorReshapeFuncs.at(0) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 4;
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = oldShape.dims[1];
        newShape.dims[2] = param.headNum;
        newShape.dims[ATTENTION_DIM_3] = oldShape.dims[2] / param.headNum;
    };
    selfAttentionFaNode.inTensorReshapeFuncs.at(1) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 4;
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = oldShape.dims[1];
        newShape.dims[2] = param.kvHead;
        newShape.dims[ATTENTION_DIM_3] = oldShape.dims[2];
    };
    selfAttentionFaNode.inTensorReshapeFuncs.at(2) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 4;
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = oldShape.dims[1];
        newShape.dims[2] = param.kvHead;
        newShape.dims[ATTENTION_DIM_3] = oldShape.dims[2];
    };

    // SelfAttention输出量化
    atb_speed::common::ParallelParamV2 selfoutLinearParam;
    selfoutLinearParam.commParam.rank = param.rank;
    selfoutLinearParam.commParam.rankSize = param.rankSize;
    selfoutLinearParam.isBias = true;
    selfoutLinearParam.isQuant = true;
    selfoutLinearParam.transposeB = true;
    selfoutLinearParam.quantParam.quantType = atb::infer::QUANT_INT8;
    selfoutLinearParam.quantParam.isQuantOp = true;
    selfoutLinearParam.quantParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_QUANT;
    selfoutLinearParam.quantParam.inputScale = param.denseInputScale;
    selfoutLinearParam.quantParam.inputOffset = param.denseInputOffset;
    atb_speed::common::RowParallelLinearV2(selfoutLinearParam, &selfoutLinearNode.operation);
    selfoutLinearNode.inTensorIds = {INTERMIDATE_SELFOUT, IN_SELFOUT_LINEAR_WEIGHT,
                                    IN_SELFOUT_LINEAR_BIAS, IN_SELFOUT_LINEAR_DEQSCALE,
                                    IN_HOLDER, IN_HOLDER, IN_HOLDER};
    selfoutLinearNode.outTensorIds = {INTERMIDATE_SELF_LINEAR_OUT};

    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CreateOperation(addParam, &selfResidualAddNode.operation);
    selfResidualAddNode.inTensorIds = { IN_HIDDENSTATES, INTERMIDATE_SELF_LINEAR_OUT };
    selfResidualAddNode.outTensorIds = { INTERMIDATE_SELF_RESIDUAL_ADD_OUT };

    // NORM量化
    atb::infer::LayerNormParam selfNormParam;
    selfNormParam.layerType = atb::infer::LayerNormParam::LAYER_NORM_NORM;
    selfNormParam.normParam.epsilon = param.layerNormEps;
    selfNormParam.normParam.beginNormAxis = LAYER_NORM_AXIS_COUNT;
    selfNormParam.normParam.beginParamsAxis = LAYER_NORM_AXIS_COUNT;
    selfNormParam.normParam.quantType = atb::infer::QUANT_INT8;
    selfNormParam.normParam.quantInputScale = param.selfLnInputScale;
    selfNormParam.normParam.quantInputOffset = param.selfLnInputOffset;
    CreateOperation(selfNormParam, &postAttnLayerNormNode.operation);
    postAttnLayerNormNode.inTensorIds = {INTERMIDATE_SELF_RESIDUAL_ADD_OUT, IN_LAYERNORN_2_WEIGHT, IN_LAYERNORN_2_BIAS};
    postAttnLayerNormNode.outTensorIds = {INTERMEDIATE_LN_2_NORM_OUT, INTERMEDIATE_LN_2_NORM_OUT_QUANT};

    // MLP量化
    atb_speed::common::MlpGateParamV2 mlpParam;
    mlpParam.activationType = atb::infer::ActivationType::ACTIVATION_GELU;
    mlpParam.isBias = true;

    mlpParam.commDownParam.rank = param.rank;
    mlpParam.commDownParam.rankSize = param.rankSize;

    mlpParam.isQuant = param.quantmodel;
    if (param.quantmodel) {
        mlpParam.transposeB = true;
        mlpParam.noGate = true;
        mlpParam.quantUpParam.quantType = atb::infer::QUANT_INT8;
        mlpParam.quantUpParam.isQuantOp = false;
        mlpParam.quantDownParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_QUANT;
        mlpParam.quantDownParam.inputScale = param.mlpOutInputScale;
        mlpParam.quantDownParam.inputOffset = param.mlpOutInputOffset;
    }
    atb_speed::common::MlpGateLayerV2(mlpParam, &mlpNode.operation);
    mlpNode.inTensorIds = {INTERMEDIATE_LN_2_NORM_OUT_QUANT, IN_MLP_UP_WEIGHT, IN_HOLDER, IN_MLP_DOWN_WEIGHT,
                           IN_MLP_UP_DEQSCALE, IN_HOLDER, IN_MLP_DOWN_DEQSCALE,
                           IN_MLP_UP_BIAS, IN_HOLDER, IN_MLP_DOWN_BIAS,
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
    ATB_LOG(INFO) << "End FlashAttentionQuantLayer";
    return atb::NO_ERROR;
}
FlashAttentionQuantHostBinder::FlashAttentionQuantHostBinder() {}

FlashAttentionQuantHostBinder::~FlashAttentionQuantHostBinder() {}

void FlashAttentionQuantHostBinder::ParseParam(const nlohmann::json &paramJson)
{
    ATB_LOG(INFO) << "FlashAttentionQuantHostBinder layer ParseParam start";

    tokenOffset_.clear();
    for (auto item : paramJson["tokenOffset"]) {
        tokenOffset_.push_back(item.get<int32_t>());
    }

    seqLen_.clear();
    for (auto item : paramJson["seqLen"]) {
        seqLen_.push_back(item.get<int32_t>());
    }
    ATB_LOG(INFO) << "FlashAttentionQuantHostBinder layer ParseParam end";
}

void FlashAttentionQuantHostBinder::BindTensor(atb::VariantPack &variantPack)
{
    ATB_LOG(INFO) << "FlashAttentionQuantHostBinder layer start";
    variantPack.inTensors.at(IN_TOKENOFFSET).hostData = tokenOffset_.data();
    variantPack.inTensors.at(IN_SEQLEN).hostData = seqLen_.data();
    ATB_LOG(INFO) << "FlashAttentionQuantHostBinder layer end";
}

} // namespace star_coder
} // namespace atb_speed