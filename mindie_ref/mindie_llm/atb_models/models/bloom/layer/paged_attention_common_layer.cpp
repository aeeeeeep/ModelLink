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

#include "layers/operations/linear.h"
#include "layers/operations/linear_parallel.h"
#include "layers/operations/norm_linear.h"
#include "layers/operations/fusion_attention.h"
#include "layers/operations/mlp.h"
#include "layers/operations/mlp_swiglu.h"
#include "models/bloom/layer/paged_attention_common_layer.h"

namespace atb_speed {
namespace bloom_7b {

static const uint64_t IN_TENSOR_COUNT = 55;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 3;
static const uint64_t NODE_COUNT = 4;

atb::Status DecoderLayer(const DecoderLayerParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);
    opGraph.name = param.isPrefill ? "Prefill_layer" : "Decoder_layer";

    size_t nodeId = 0;
    atb::Node &attentionNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfResidualAddNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpParallelNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpResidualAddNode = opGraph.nodes.at(nodeId++);

    atb_speed::common::FusionAttentionParam<atb::infer::LayerNormParam> fusionAttentionParam;
    // QKV linear param
    fusionAttentionParam.packQuantType = param.packQuantType[0];
    fusionAttentionParam.layerLinearQuantType = param.linearQuantType;
    fusionAttentionParam.isGroupedQueryAttention = param.numAttentionHeadsPerRank != param.numKeyValueHeadsPerRank;
    fusionAttentionParam.isBF16 = param.isBF16;
    fusionAttentionParam.normHasBias = true;
    fusionAttentionParam.qkvHasBias = true;
    fusionAttentionParam.selfAttnHasBias = true;
    fusionAttentionParam.splitWithStride = true;
    
    fusionAttentionParam.layerLinearQuantType = param.linearQuantType;
    fusionAttentionParam.packQuantType = param.packQuantType[0];
    fusionAttentionParam.supportLcoc = param.supportLcoc;
    atb::infer::LayerNormParam attenLayerNormParam;
    attenLayerNormParam.layerType = atb::infer::LayerNormParam::LAYER_NORM_NORM;
    const int32_t beginParamsAxis = param.isFA ? 2 : 1;
    attenLayerNormParam.normParam.epsilon = param.layerNormEps;
    attenLayerNormParam.normParam.beginNormAxis = beginParamsAxis;
    attenLayerNormParam.normParam.beginParamsAxis = 1;
    fusionAttentionParam.normParamType = attenLayerNormParam;
    atb::infer::LayerNormParam attenLayerNormQuantParam;
    attenLayerNormQuantParam.layerType = atb::infer::LayerNormParam::LAYER_NORM_NORM;
    attenLayerNormQuantParam.normParam.epsilon = param.layerNormEps;
    attenLayerNormQuantParam.normParam.beginNormAxis = beginParamsAxis;
    attenLayerNormQuantParam.normParam.beginParamsAxis = 1;
    attenLayerNormQuantParam.normParam.quantType = atb::infer::QUANT_INT8;
    fusionAttentionParam.normQuantParamType = attenLayerNormQuantParam;
    // rope param
    fusionAttentionParam.rotaryType = atb_speed::common::RotaryType::NO_ROTARY;
    // self attention param
    fusionAttentionParam.isFA = param.isFA;
    fusionAttentionParam.isPrefill = param.isPrefill;
    fusionAttentionParam.headDim = param.hiddenSizePerAttentionHead;
    fusionAttentionParam.selfAttentionParam.headNum = param.numAttentionHeadsPerRank;
    fusionAttentionParam.selfAttentionParam.kvHeadNum = param.numKeyValueHeadsPerRank;
    if (param.hiddenSizePerAttentionHead == 0) {
        return atb::ERROR_INVALID_GRAPH;
    }
    fusionAttentionParam.selfAttentionParam.qkScale = 1.0 / sqrt(param.hiddenSizePerAttentionHead);
    fusionAttentionParam.selfAttentionParam.isTriuMask = param.isPrefill ? 1 : 0;
    if (param.isFA) {
        fusionAttentionParam.selfAttentionParam.calcType = param.isPrefill ? \
            atb::infer::SelfAttentionParam::CalcType::ENCODER : atb::infer::SelfAttentionParam::CalcType::DECODER;
    } else {
        fusionAttentionParam.selfAttentionParam.calcType = atb::infer::SelfAttentionParam::CalcType::PA_ENCODER;
    }
    fusionAttentionParam.pageAttentionParam.headNum = param.numAttentionHeadsPerRank;
    fusionAttentionParam.pageAttentionParam.kvHeadNum = param.numKeyValueHeadsPerRank;
    fusionAttentionParam.pageAttentionParam.qkScale = 1.0 / sqrt(param.hiddenSizePerAttentionHead);
    fusionAttentionParam.pageAttentionParam.maskType = atb::infer::PagedAttentionParam::MaskType::MASK_TYPE_ALIBI;
    fusionAttentionParam.selfAttentionParam.maskType = atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_ALIBI;
    fusionAttentionParam.selfOutLinearTensorParallelInfo = {param.rank, param.worldSize, param.backend};
    Attention(fusionAttentionParam, &attentionNode.operation);
    attentionNode.inTensorIds = {
        IN_HIDDEN_STATES,
        IN_INPUT_NORM_WEIGHT,
        IN_INPUT_NORM_BIAS,
        IN_INPUT_NORM_NEW_WEIGHT,
        IN_INPUT_NORM_NEW_BIAS,
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
        IN_ATTENTION_OUT_WEIGHT,
        IN_ATTENTION_OUT_SCALE,
        IN_ATTENTION_OUT_OFFSET,
        IN_ATTENTION_OUT_DESCALE,
        IN_ATTENTION_OUT_BIAS,
    };
    attentionNode.outTensorIds = {INTERMEDIATE_ATTENTION_OUT};

    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CREATE_OPERATION(addParam, &selfResidualAddNode.operation);
    selfResidualAddNode.inTensorIds = {
        IN_HIDDEN_STATES,
        INTERMEDIATE_ATTENTION_OUT
    };
    selfResidualAddNode.outTensorIds = {INTERMEDIATE_RESIDUAL_ADD_OUT};

    atb_speed::common::MlpParam<atb::infer::LayerNormParam> mlpParam;
    mlpParam.isBF16 = param.isBF16;
    mlpParam.normHasBias = true;
    mlpParam.gateUpHasBias = true;
    mlpParam.downHasBias = true;
    mlpParam.packQuantType = param.packQuantType[1];
    mlpParam.layerLinearQuantType = param.linearQuantType;
    mlpParam.mlpPackType = atb_speed::common::UP_WEIGHT_ONLY;
    atb::infer::LayerNormParam mlpLayerNormParam;
    mlpLayerNormParam.layerType = atb::infer::LayerNormParam::LAYER_NORM_NORM;
    mlpLayerNormParam.normParam.epsilon = param.layerNormEps;
    mlpLayerNormParam.normParam.beginNormAxis = beginParamsAxis;
    mlpLayerNormParam.normParam.beginParamsAxis = 1;
    mlpParam.normParamType = mlpLayerNormParam;
    atb::infer::LayerNormParam mlpLayerNormQuantParam;
    mlpLayerNormQuantParam.layerType = atb::infer::LayerNormParam::LAYER_NORM_NORM;
    mlpLayerNormQuantParam.normParam.epsilon = param.layerNormEps;
    mlpLayerNormQuantParam.normParam.beginNormAxis = beginParamsAxis;
    mlpLayerNormQuantParam.normParam.beginParamsAxis = 1;
    mlpLayerNormQuantParam.normParam.quantType = atb::infer::QUANT_INT8;
    mlpParam.normQuantParamType = mlpLayerNormQuantParam;
    // down
    mlpParam.downLinearTensorParallelInfo = {param.rank, param.worldSize, param.backend};
    mlpParam.supportLcoc = param.supportLcoc;
    if (param.supportSwiGLU) {
        mlpParam.activationParam.activationType = atb::infer::ActivationType::ACTIVATION_SWIGLU_FORWARD;
        mlpParam.activationParam.dim = -1;
        MlpSwiGLU(mlpParam, &mlpParallelNode.operation);
    } else {
        mlpParam.activationParam.activationType = atb::infer::ActivationType::ACTIVATION_GELU;
        Mlp(mlpParam, &mlpParallelNode.operation);
    }

    mlpParallelNode.inTensorIds = {
        INTERMEDIATE_RESIDUAL_ADD_OUT,
        IN_ATTENTION_NORM_WEIGHT,
        IN_ATTENTION_NORM_BIAS,
        IN_ATTENTION_NORM_NEW_WEIGHT,
        IN_ATTENTION_NORM_NEW_BIAS,
        IN_MLP_WEIGHT_0,
        IN_MLP_SCALE_0,
        IN_MLP_OFFSET_0,
        IN_MLP_DESCALE_0,
        IN_MLP_BIAS_0,
        IN_MLP_WEIGHT_1,
        IN_MLP_SCALE_1,
        IN_MLP_OFFSET_1,
        IN_MLP_DESCALE_1,
        IN_MLP_BIAS_1,
        IN_MLP_DOWN_WEIGHT,
        IN_MLP_DOWN_SCALE,
        IN_MLP_DOWN_OFFSET,
        IN_MLP_DOWN_DESCALE,
        IN_MLP_DOWN_BIAS,
    };
    mlpParallelNode.outTensorIds = {INTERMEDIATE_MLP_OUT};

    CREATE_OPERATION(addParam, &mlpResidualAddNode.operation);
    mlpResidualAddNode.inTensorIds = {
        INTERMEDIATE_RESIDUAL_ADD_OUT,
        INTERMEDIATE_MLP_OUT
    };
    mlpResidualAddNode.outTensorIds = {OUT_DECODER_LAYER};

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
    };

    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}

} // namespace bloom_7b
} // namespace atb_speed