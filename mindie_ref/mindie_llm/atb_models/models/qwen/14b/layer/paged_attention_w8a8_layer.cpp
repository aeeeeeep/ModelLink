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
#include "models/qwen/14b/layer/paged_attention_w8a8_layer.h"

namespace atb_speed {
namespace qwen_14b {
enum PAW8A8LayerW8A8TensorId : int {
    IN_HIDDEN_STATES = 0,

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

    OUT_LAYEROUT,

    INTERNAL_ATTENTIONOUT,
    INTERNAL_ATTENTIONRESIDUALADDOUT,
    INTERNAL_MLPOUT,
};

static const uint64_t IN_TENSOR_COUNT = 53;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 3;
static const uint64_t NODE_COUNT = 4;

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
    atb::Node &selfResidualAddNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpParallelNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpResidualAddNode = opGraph.nodes.at(nodeId++);

    // attention
    atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam> fusionAttentionParam;
    // QKV linear param
    fusionAttentionParam.isAntiOutlier = param.packQuantType[0] == atb_speed::common::MIX_W8A8_ANTI || param.packQuantType[0] == atb_speed::common::ALL_W8A8_ANTI;
    // fusionAttentionParam.isPack = param.packQuantType[0] != atb_speed::common::MIX_W8A8 && param.packQuantType[0] != atb_speed::common::MIX_W8A8_ANTI;
    fusionAttentionParam.isPack = true;
    // fusionAttentionParam.isGroupedQueryAttention = param.numAttentionHeadsPerRank != param.numKeyValueHeadsPerRank;
    fusionAttentionParam.isGroupedQueryAttention = false;
    fusionAttentionParam.isBF16 = param.isBF16;
    fusionAttentionParam.layerLinearQuantType = param.linearQuantType;
    atb::infer::RmsNormParam attenRmsNormParam;
    attenRmsNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    attenRmsNormParam.normParam.epsilon = param.rmsNormEps;
    fusionAttentionParam.normParamType = attenRmsNormParam;
    atb::infer::RmsNormParam attenRmsNormQuantParam;
    attenRmsNormQuantParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    attenRmsNormQuantParam.normParam.epsilon = param.rmsNormEps;
    attenRmsNormQuantParam.normParam.quantType = atb::infer::QUANT_INT8;
    fusionAttentionParam.normQuantParamType = attenRmsNormQuantParam;
    // rope param
    fusionAttentionParam.rotaryCoeff = 2;
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
    if (param.isFA) {
        fusionAttentionParam.selfAttentionParam.isTriuMask = param.isPrefill ? 1 : 0;
        fusionAttentionParam.selfAttentionParam.calcType = param.isPrefill ? \
            atb::infer::SelfAttentionParam::CalcType::ENCODER : atb::infer::SelfAttentionParam::CalcType::DECODER;
    } else {
        fusionAttentionParam.selfAttentionParam.calcType = atb::infer::SelfAttentionParam::CalcType::PA_ENCODER;
    }
    fusionAttentionParam.pageAttentionParam.headNum = param.numAttentionHeadsPerRank;
    fusionAttentionParam.pageAttentionParam.kvHeadNum = param.numKeyValueHeadsPerRank;
    fusionAttentionParam.pageAttentionParam.qkScale = 1.0 / sqrt(param.hiddenSizePerAttentionHead);
    if (param.isBF16) {
        fusionAttentionParam.pageAttentionParam.maskType = atb::infer::PagedAttentionParam::MaskType::MASK_TYPE_ALIBI;
    } else {
        fusionAttentionParam.pageAttentionParam.maskType = atb::infer::PagedAttentionParam::MaskType::UNDEFINED;
    }
    fusionAttentionParam.selfOutLinearTensorParallelInfo = {param.rank, param.worldSize, param.backend};
    Attention(fusionAttentionParam, &attentionNode.operation);
    attentionNode.inTensorIds = {
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
    attentionNode.outTensorIds = {INTERNAL_ATTENTIONOUT};

    // residual
    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CREATE_OPERATION(addParam, &selfResidualAddNode.operation);
    selfResidualAddNode.inTensorIds = {
        IN_HIDDEN_STATES,
        INTERNAL_ATTENTIONOUT
    };
    selfResidualAddNode.outTensorIds = {INTERNAL_ATTENTIONRESIDUALADDOUT};

    // mlp
    atb_speed::common::MlpParam<atb::infer::RmsNormParam> mlpParam;
    mlpParam.isBF16 = param.isBF16;
    mlpParam.isAntiOutlier = param.packQuantType[1] == atb_speed::common::MIX_W8A8_ANTI || param.packQuantType[1] == atb_speed::common::ALL_W8A8_ANTI;
    mlpParam.layerLinearQuantType = param.linearQuantType;
    // w2_w1(gate_up)
    mlpParam.mlpPackType = atb_speed::common::GATE_UP_WEIGHT_PACK;
    atb::infer::RmsNormParam mlpRmsNormParam;
    mlpRmsNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    mlpRmsNormParam.normParam.epsilon = param.rmsNormEps;
    mlpParam.normParamType = mlpRmsNormParam;
    atb::infer::RmsNormParam mlpRmsNormQuantParam;
    mlpRmsNormQuantParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    mlpRmsNormQuantParam.normParam.epsilon = param.rmsNormEps;
    mlpRmsNormQuantParam.normParam.quantType = atb::infer::QUANT_INT8;
    mlpParam.normQuantParamType = mlpRmsNormQuantParam;
    // c_proj(down)
    mlpParam.downLinearTensorParallelInfo = {param.rank, param.worldSize, param.backend};
    if (param.supportSwiGLU) {
        MlpSwiGLU(mlpParam, &mlpParallelNode.operation);
    } else {
        Mlp(mlpParam, &mlpParallelNode.operation);
    }
    mlpParallelNode.inTensorIds = {
        INTERNAL_ATTENTIONRESIDUALADDOUT,  // INTERMEDIATE_RESIDUAL_ADD_OUT
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
    mlpParallelNode.outTensorIds = {INTERNAL_MLPOUT};

    // residual
    CREATE_OPERATION(addParam, &mlpResidualAddNode.operation);
    mlpResidualAddNode.inTensorIds = {
        INTERNAL_ATTENTIONRESIDUALADDOUT,
        INTERNAL_MLPOUT
    };
    mlpResidualAddNode.outTensorIds = {OUT_LAYEROUT};

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
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