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
#include "models/llama_parallel/layer/decoder_layer.h"

namespace atb_speed {
namespace llama_parallel {

static const uint64_t IN_TENSOR_COUNT = 62;
static const uint64_t OUT_TENSOR_COUNT = 2;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 1;
static const uint64_t NODE_COUNT = 2;

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
    atb::Node &mlpParallelNode = opGraph.nodes.at(nodeId++);

    atb::infer::RmsNormParam attenRmsNormParam;
    if (param.layerId == 0) {
        attenRmsNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
        attenRmsNormParam.normParam.epsilon = param.rmsNormEps;
    } else {
        attenRmsNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_PRENORM;
        attenRmsNormParam.preNormParam.epsilon = param.rmsNormEps;
    }

    atb::infer::RmsNormParam attenRmsNormQuantParam;
    if (param.layerId == 0) {
        attenRmsNormQuantParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
        attenRmsNormQuantParam.normParam.epsilon = param.rmsNormEps;
        attenRmsNormQuantParam.normParam.quantType = atb::infer::QUANT_INT8;
    } else {
        attenRmsNormQuantParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_PRENORM;
        attenRmsNormQuantParam.preNormParam.epsilon = param.rmsNormEps;
        attenRmsNormQuantParam.preNormParam.quantType = atb::infer::QUANT_INT8;
    }

    atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam> fusionAttentionParam;
    // QKV linear param
    fusionAttentionParam.isGroupedQueryAttention = param.numAttentionHeadsPerRank != param.numKeyValueHeadsPerRank;
    fusionAttentionParam.isBF16 = param.isBF16;
    fusionAttentionParam.layerLinearQuantType = param.linearQuantType;
    fusionAttentionParam.packQuantType = param.packQuantType[0];
    fusionAttentionParam.normParamType = attenRmsNormParam;
    fusionAttentionParam.normQuantParamType = attenRmsNormQuantParam;
    fusionAttentionParam.addNormType = param.layerId == 0 ? \
        atb_speed::common::AddNormType::NORM_ONLY : atb_speed::common::AddNormType::FUSION_ADD_NORM;
    // rope param
    fusionAttentionParam.rotaryType = atb_speed::common::RotaryType::ALL_ROTARY;
    fusionAttentionParam.ropeParam.rotaryCoeff = 2;
    // self attention param
    fusionAttentionParam.isFA = param.isFA;
    fusionAttentionParam.isPrefill = param.isPrefill;
    fusionAttentionParam.headDim = param.hiddenSizePerAttentionHead;
    fusionAttentionParam.selfAttentionParam.headNum = param.numAttentionHeadsPerRank;
    fusionAttentionParam.selfAttentionParam.kvHeadNum = param.numKeyValueHeadsPerRank;
    fusionAttentionParam.selfAttentionParam.qkScale = 1.0 / sqrt(param.hiddenSizePerAttentionHead);
    if (param.isFA) {
        fusionAttentionParam.selfAttentionParam.calcType = param.isPrefill ? \
            atb::infer::SelfAttentionParam::CalcType::ENCODER : atb::infer::SelfAttentionParam::CalcType::DECODER;
    } else {
        fusionAttentionParam.selfAttentionParam.isTriuMask = param.isPrefill ? 1 : 0;
        fusionAttentionParam.selfAttentionParam.calcType = atb::infer::SelfAttentionParam::CalcType::PA_ENCODER;
    }
    fusionAttentionParam.selfAttentionParam.maskType = atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_NORM;
    fusionAttentionParam.pageAttentionParam.headNum = param.numAttentionHeadsPerRank;
    fusionAttentionParam.pageAttentionParam.kvHeadNum = param.numKeyValueHeadsPerRank;
    fusionAttentionParam.pageAttentionParam.qkScale = 1.0 / sqrt(param.hiddenSizePerAttentionHead);
    // self attention dense
    fusionAttentionParam.supportLcoc = param.supportLcoc;
    fusionAttentionParam.selfOutLinearTensorParallelInfo = param.tensorParallelInfo;
    Attention(fusionAttentionParam, &attentionNode.operation);
    attentionNode.inTensorIds = {
        IN_RESIDUAL_ADD_OUT,
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
        IN_QKV_COMPRESS_IDX_0,
        IN_QKV_WEIGHT_1,
        IN_QKV_SCALE_1,
        IN_QKV_OFFSET_1,
        IN_QKV_DESCALE_1,
        IN_QKV_BIAS_1,
        IN_QKV_COMPRESS_IDX_1,
        IN_QKV_WEIGHT_2,
        IN_QKV_SCALE_2,
        IN_QKV_OFFSET_2,
        IN_QKV_DESCALE_2,
        IN_QKV_BIAS_2,
        IN_QKV_COMPRESS_IDX_2,
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
        IN_ATTENTION_OUT_COMPRESS_IDX
    };
    attentionNode.outTensorIds = {IN_RESIDUAL_ADD_OUT, INTERMEDIATE_ATTENTION_OUT};

    atb::infer::RmsNormParam mlpRmsNormParam;
    mlpRmsNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_PRENORM;
    mlpRmsNormParam.preNormParam.epsilon = param.rmsNormEps;

    atb::infer::RmsNormParam mlpRmsNormQuantParam;
    mlpRmsNormQuantParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_PRENORM;
    mlpRmsNormQuantParam.preNormParam.epsilon = param.rmsNormEps;
    mlpRmsNormQuantParam.preNormParam.quantType = atb::infer::QUANT_INT8;

    atb_speed::common::MlpParam<atb::infer::RmsNormParam> mlpParam;
    mlpParam.isBF16 = param.isBF16;
    mlpParam.layerLinearQuantType = param.linearQuantType;
    mlpParam.packQuantType = param.packQuantType[1];
    // gate up
    mlpParam.mlpPackType = atb_speed::common::GetMlpPackType(param.packQuantType[1], false);
    mlpParam.normParamType = mlpRmsNormParam;
    mlpParam.normQuantParamType = mlpRmsNormQuantParam;
    mlpParam.addNormType = atb_speed::common::AddNormType::FUSION_ADD_NORM;
    // down
    mlpParam.downLinearTensorParallelInfo = param.tensorParallelInfo;
    mlpParam.supportLcoc = param.supportLcoc;
    if (param.supportSwiGLU) {
        mlpParam.activationParam.activationType = atb::infer::ActivationType::ACTIVATION_SWIGLU_FORWARD;
        mlpParam.activationParam.dim = -1;
        MlpSwiGLU(mlpParam, &mlpParallelNode.operation);
    } else {
        mlpParam.activationParam.activationType = atb::infer::ActivationType::ACTIVATION_SWISH;
        Mlp(mlpParam, &mlpParallelNode.operation);
    }

    mlpParallelNode.inTensorIds = {
        param.layerId == 0 ? IN_HIDDEN_STATES : IN_RESIDUAL_ADD_OUT,
        INTERMEDIATE_ATTENTION_OUT,
        IN_ATTENTION_NORM_WEIGHT,
        IN_ATTENTION_NORM_BIAS,
        IN_ATTENTION_NORM_NEW_WEIGHT,
        IN_ATTENTION_NORM_NEW_BIAS,
        IN_MLP_WEIGHT_0,
        IN_MLP_SCALE_0,
        IN_MLP_OFFSET_0,
        IN_MLP_DESCALE_0,
        IN_MLP_BIAS_0,
        IN_MLP_COMPRESS_IDX_0,
        IN_MLP_WEIGHT_1,
        IN_MLP_SCALE_1,
        IN_MLP_OFFSET_1,
        IN_MLP_DESCALE_1,
        IN_MLP_BIAS_1,
        IN_MLP_COMPRESS_IDX_1,
        IN_MLP_DOWN_WEIGHT,
        IN_MLP_DOWN_SCALE,
        IN_MLP_DOWN_OFFSET,
        IN_MLP_DOWN_DESCALE,
        IN_MLP_DOWN_BIAS,
        IN_MLP_DOWN_COMPRESS_IDX
    };
    mlpParallelNode.outTensorIds = {OUT_ATTENTION_RESIDUAL_ADD, OUT_MLP};

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(IN_HIDDEN_STATES);
        outTensorDescs.at(1) = inTensorDescs.at(IN_HIDDEN_STATES);
        return atb::NO_ERROR;
    };

    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}

} // namespace llama_parallel
} // namespace atb_speed