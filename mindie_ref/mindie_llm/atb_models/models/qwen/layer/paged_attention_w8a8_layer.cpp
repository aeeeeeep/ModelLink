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
#include "models/qwen/layer/paged_attention_w8a8_layer.h"
#include "atb_speed/log.h"

namespace atb_speed {
namespace qwen_14b {
enum PAW8A8LayerW8A8TensorId : int {
    IN_RESIDUAL_ADD_OUT = 0,
    IN_HIDDEN_STATES,

    IN_NORM_WEIGHT,  // weight
    IN_NORM_BIAS,  // bias
    IN_NORM_NEW_WEIGHT,  // new weight
    IN_NORM_NEW_BIAS,  // new bias

    IN_Q_WEIGHT,  // weight
    IN_Q_BIAS,  // bias
    IN_Q_DEQSCALE,  // deq_scale
    IN_Q_OFFSET,  // offset
    IN_Q_SCALE,  // scale

    IN_K_WEIGHT,  // weight
    IN_K_BIAS,  // bias
    IN_K_DEQSCALE,  // deq_scale
    IN_K_OFFSET,  // offset
    IN_K_SCALE,  // scale

    IN_V_WEIGHT,  // weight
    IN_V_BIAS,  // bias
    IN_V_DEQSCALE,  // deq_scale
    IN_V_OFFSET,  // offset
    IN_V_SCALE,  // scale

    IN_ATTENTION_OUT_WEIGHT,  // weight
    IN_ATTENTION_OUT_BIAS,  // bias
    IN_ATTENTION_OUT_DEQSCALE,  // deq_scale
    IN_ATTENTION_OUT_OFFSET,  // offset
    IN_ATTENTION_OUT_SCALE,  // scale

    IN_SELFOUT_NORM_WEIGHT,  // weight
    IN_SELFOUT_NORM_BIAS,  // bias
    IN_SELFOUT_NORM_NEW_WEIGHT,  // new weight
    IN_SELFOUT_NORM_NEW_BIAS,  // new bias

    IN_MLP_W2_WEIGHT,  // weight
    IN_MLP_W2_BIAS,  // bias
    IN_MLP_W2_DEQSCALE,  // deq_scale
    IN_MLP_W2_OFFSET,  // offset
    IN_MLP_W2_SCALE,  // scale

    IN_MLP_W1_WEIGHT,  // weight
    IN_MLP_W1_BIAS,  // bias
    IN_MLP_W1_DEQSCALE,  // deq_scale
    IN_MLP_W1_OFFSET,  // offset
    IN_MLP_W1_SCALE,  // scale

    IN_MLP_CPROJ_WEIGHT,  // weight
    IN_MLP_CPROJ_BIAS,  // bias
    IN_MLP_CPROJ_DEQSCALE,  // deq_scale
    IN_MLP_CPROJ_OFFSET,  // offset
    IN_MLP_CPROJ_SCALE,  // scale

    IN_COSEMBED,
    IN_SINEMBED,
    IN_ATTENTIONMASK,
    IN_K_CACHE,
    IN_V_CACHE,
    IN_BLOCK_TABLES,
    IN_SLOTS,
    IN_INPUT_LENGTHS,
    IN_PLACEHOLDER,

    OUT_ATTENTION_RESIDUAL_ADD,
    OUT_MLP,

    INTERNAL_ATTENTIONOUT,
};

static const uint64_t IN_TENSOR_COUNT = 54;
static const uint64_t OUT_TENSOR_COUNT = 2;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 1;
static const uint64_t NODE_COUNT = 2;

atb::Status PAW8A8Layer(const PAW8A8LayerParam &param, atb::Operation **operation)
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

    //attention
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
    fusionAttentionParam.qkvHasBias = true;
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
    fusionAttentionParam.selfAttentionParam.isTriuMask = param.isPrefill ? 1 : 0;
    if (param.isFA) {
        fusionAttentionParam.selfAttentionParam.calcType = param.isPrefill ? \
            atb::infer::SelfAttentionParam::CalcType::ENCODER : atb::infer::SelfAttentionParam::CalcType::DECODER;
    } else {
        fusionAttentionParam.selfAttentionParam.calcType = atb::infer::SelfAttentionParam::CalcType::PA_ENCODER;
    }
    fusionAttentionParam.selfAttentionParam.maskType = atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_NORM;
    fusionAttentionParam.pageAttentionParam.headNum = param.numAttentionHeadsPerRank;
    fusionAttentionParam.pageAttentionParam.kvHeadNum = param.numKeyValueHeadsPerRank;
    fusionAttentionParam.pageAttentionParam.qkScale = 1.0 / sqrt(param.hiddenSizePerAttentionHead);
    // self attention dense
    fusionAttentionParam.supportLcoc = param.supportLcoc;
    fusionAttentionParam.selfOutLinearTensorParallelInfo = {param.rank, param.worldSize, param.backend};
    Attention(fusionAttentionParam, &attentionNode.operation);
    attentionNode.inTensorIds = {
        IN_RESIDUAL_ADD_OUT,  // IN_RESIDUAL_ADD_OUT
        IN_HIDDEN_STATES,  // IN_HIDDEN_STATES
        IN_NORM_WEIGHT,  // IN_INPUT_NORM_WEIGHT
        IN_NORM_BIAS,  // IN_INPUT_NORM_BIAS
        IN_NORM_NEW_WEIGHT,  // IN_INPUT_NORM_NEW_WEIGHT
        IN_NORM_NEW_BIAS,  // IN_INPUT_NORM_NEW_BIAS
        IN_Q_WEIGHT,  // IN_QKV_WEIGHT_0
        IN_Q_SCALE,  // IN_QKV_SCALE_0
        IN_Q_OFFSET,  // IN_QKV_OFFSET_0
        IN_Q_DEQSCALE,  // IN_QKV_DESCALE_0
        IN_Q_BIAS,  // IN_QKV_DEOFFSET_0（quant场景下为quant_bias，非quant场景下为bias）
        IN_K_WEIGHT,  // IN_QKV_WEIGHT_1
        IN_K_SCALE,  // IN_QKV_SCALE_1
        IN_K_OFFSET,  // IN_QKV_OFFSET_1
        IN_K_DEQSCALE,  // IN_QKV_DESCALE_1
        IN_K_BIAS,  // IN_QKV_DEOFFSET_1（quant场景下为quant_bias，非quant场景下为bias）
        IN_V_WEIGHT,  // IN_QKV_WEIGHT_2
        IN_V_SCALE,  // IN_QKV_SCALE_2
        IN_V_OFFSET,  // IN_QKV_OFFSET_2
        IN_V_DEQSCALE,  // IN_QKV_DESCALE_2
        IN_V_BIAS,  // IN_QKV_DEOFFSET_2（quant场景下为quant_bias，非quant场景下为bias）
        IN_COSEMBED,  // IN_COS_TABLE
        IN_SINEMBED,  // IN_SIN_TABLE
        IN_INPUT_LENGTHS,  // IN_SEQ_LEN
        IN_K_CACHE,  // IN_K_CACHE
        IN_V_CACHE,  // IN_V_CACHE
        IN_ATTENTIONMASK,  // IN_ATTENTION_MASK
        IN_PLACEHOLDER,  // IN_TOKEN_OFFSET
        IN_PLACEHOLDER,  // IN_LAYER_ID
        IN_BLOCK_TABLES,  // IN_BLOCK_TABLES
        IN_SLOTS,  // IN_SLOTS
        IN_ATTENTION_OUT_WEIGHT,  // IN_ATTENTION_OUT_WEIGHT
        IN_ATTENTION_OUT_SCALE,  // IN_ATTENTION_OUT_SCALE
        IN_ATTENTION_OUT_OFFSET,  // IN_ATTENTION_OUT_OFFSET
        IN_ATTENTION_OUT_DEQSCALE,  // IN_ATTENTION_OUT_DESCALE
        IN_ATTENTION_OUT_BIAS,  // IN_ATTENTION_OUT_DEOFFSET（quant场景下为quant_bias，非quant场景下为bias）
    };
    attentionNode.outTensorIds = {IN_RESIDUAL_ADD_OUT, INTERNAL_ATTENTIONOUT};
    ATB_LOG(INFO) << "[+] attentionNode";

    // mlp
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
    // w2_w1(gate_up)
    mlpParam.mlpPackType = atb_speed::common::GATE_UP_WEIGHT_PACK;
    mlpParam.normParamType = mlpRmsNormParam;
    mlpParam.normQuantParamType = mlpRmsNormQuantParam;
    mlpParam.addNormType = atb_speed::common::AddNormType::FUSION_ADD_NORM;
    // c_proj(down)
    mlpParam.downLinearTensorParallelInfo = {param.rank, param.worldSize, param.backend};
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
        INTERNAL_ATTENTIONOUT,  // INTERMEDIATE_ATTENTION_OUT
        IN_SELFOUT_NORM_WEIGHT,  // IN_ATTENTION_NORM_WEIGHT
        IN_SELFOUT_NORM_BIAS,  // IN_ATTENTION_NORM_BIAS
        IN_SELFOUT_NORM_NEW_WEIGHT,  // IN_ATTENTION_NORM_NEW_WEIGHT
        IN_SELFOUT_NORM_NEW_BIAS,  // IN_ATTENTION_NORM_NEW_BIAS
        IN_MLP_W2_WEIGHT,  // IN_MLP_WEIGHT_0
        IN_MLP_W2_SCALE,  // IN_MLP_SCALE_0
        IN_MLP_W2_OFFSET,  // IN_MLP_OFFSET_0
        IN_MLP_W2_DEQSCALE,  // IN_MLP_DESCALE_0
        IN_MLP_W2_BIAS,  // IN_MLP_DEOFFSET_0
        IN_MLP_W1_WEIGHT,  // IN_MLP_WEIGHT_1
        IN_MLP_W1_SCALE,  // IN_MLP_SCALE_1
        IN_MLP_W1_OFFSET,  // IN_MLP_OFFSET_1
        IN_MLP_W1_DEQSCALE,  // IN_MLP_DESCALE_1
        IN_MLP_W1_BIAS,  // IN_MLP_DEOFFSET_1
        IN_MLP_CPROJ_WEIGHT,  // IN_MLP_DOWN_WEIGHT
        IN_MLP_CPROJ_SCALE,  // IN_MLP_DOWN_SCALE
        IN_MLP_CPROJ_OFFSET,  // IN_MLP_DOWN_OFFSET
        IN_MLP_CPROJ_DEQSCALE,  // IN_MLP_DOWN_DESCALE
        IN_MLP_CPROJ_BIAS,  // IN_MLP_DOWN_DEOFFSET
    };
    mlpParallelNode.outTensorIds = {OUT_ATTENTION_RESIDUAL_ADD, OUT_MLP};
    ATB_LOG(INFO) << "[+] mlpParallelNode";

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(IN_HIDDEN_STATES);
        outTensorDescs.at(1) = inTensorDescs.at(IN_HIDDEN_STATES);
        return atb::NO_ERROR;
    };

    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}

PAW8A8LayerBinder::PAW8A8LayerBinder() {}

PAW8A8LayerBinder::~PAW8A8LayerBinder() {}

void PAW8A8LayerBinder::ParseParam(const nlohmann::json &paramJson)
{
    ATB_LOG(INFO) << "enter PAW8A8LayerBinder ParseParam tokenOffset";
    seqLen_.clear();
    for (auto item : paramJson["seqLen"]) {
        seqLen_.push_back(item.get<int>());
    }
}

void PAW8A8LayerBinder::BindTensor(atb::VariantPack &variantPack)
{
    ATB_LOG(INFO) << "enter PAW8A8LayerOperation BindTensor";
    variantPack.inTensors.at(IN_INPUT_LENGTHS).hostData = seqLen_.data();
}

} // namespace qwen_14b
} // namespace atb_speed