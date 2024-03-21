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
#include "layers/operations/linear.h"
#include "layers/operations/linear.h"
#include "layers/operations/linear_parallel.h"
#include "layers/operations/norm_linear.h"
#include "layers/operations/fusion_attention.h"
#include "layers/operations/mlp.h"
#include "layers/operations/mlp_swiglu.h"

namespace atb_speed {
static const uint64_t IN_TENSOR_COUNT = 62;
static const uint64_t OUT_TENSOR_COUNT = 2;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 1;
static const uint64_t NODE_COUNT = 2;

void from_json(const nlohmann::json &paramJson, PAQuantLayerParam &param)
{
    paramJson.at("rmsNormEps").get_to(param.rmsNormEps);
    paramJson.at("headNum").get_to(param.headNum);
    paramJson.at("dk").get_to(param.dk);
    if (paramJson.contains("rank")) {
        paramJson.at("rank").get_to(param.rank);
    }
    if (paramJson.contains("rankSize")) {
        paramJson.at("rankSize").get_to(param.rankSize);
    }
    if (paramJson.contains("transposedWeight")) {
        paramJson.at("transposedWeight").get_to(param.transposedWeight);
    }
    if (paramJson.contains("isPrefill")) {
        paramJson.at("isPrefill").get_to(param.isPrefill);
    }
    if (paramJson.contains("backend")) {
        paramJson.at("backend").get_to(param.backend);
    }
}

atb::Status PAQuantLayer(const PAQuantLayerParam &param, atb::Operation **operation)
{
    ATB_LOG(INFO) << __func__ << " called, headNum: " << param.headNum;
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);
    if (param.isPrefill) {
        opGraph.name = "Prefill_transformer_layer";
    } else {
        opGraph.name = "Decoder_transformer_layer";
    }

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
    fusionAttentionParam.packQuantType = param.packQuantType[0];
    fusionAttentionParam.layerLinearQuantType = param.linearQuantType;
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
    if (param.hiddenSizePerAttentionHead == 0) {
        return atb::ERROR_INVALID_GRAPH;
    }
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
    fusionAttentionParam.supportLcoc = param.supportLcoc;
    fusionAttentionParam.selfOutLinearTensorParallelInfo = {param.rank, param.rankSize, param.backend};
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
        IN_QKV_DEOFFSET_0,
        IN_QKV_COMPRESS_IDX_0,
        IN_QKV_WEIGHT_1,
        IN_QKV_SCALE_1,
        IN_QKV_OFFSET_1,
        IN_QKV_DESCALE_1,
        IN_QKV_DEOFFSET_1,
        IN_QKV_COMPRESS_IDX_1,
        IN_QKV_WEIGHT_2,
        IN_QKV_SCALE_2,
        IN_QKV_OFFSET_2,
        IN_QKV_DESCALE_2,
        IN_QKV_DEOFFSET_2,
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
        IN_ATTENTION_OUT_DEOFFSET,
        IN_ATTENTION_OUT_COMPRESS_IDX,
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
    mlpParam.packQuantType = param.packQuantType[1];
    mlpParam.layerLinearQuantType = param.linearQuantType;
    mlpParam.supportLcoc = param.supportLcoc;
    // gate up
    mlpParam.mlpPackType = atb_speed::common::GetMlpPackType(param.packQuantType[1], false);
    mlpParam.normParamType = mlpRmsNormParam;
    mlpParam.normQuantParamType = mlpRmsNormQuantParam;
    mlpParam.addNormType = atb_speed::common::AddNormType::FUSION_ADD_NORM;
    // down
    mlpParam.downLinearTensorParallelInfo = {param.rank, param.rankSize, param.backend};

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
        IN_MLP_DEOFFSET_0,
        IN_MLP_COMPRESS_IDX_0,
        IN_MLP_WEIGHT_1,
        IN_MLP_SCALE_1,
        IN_MLP_OFFSET_1,
        IN_MLP_DESCALE_1,
        IN_MLP_DEOFFSET_1,
        IN_MLP_COMPRESS_IDX_1,
        IN_MLP_DOWN_WEIGHT,
        IN_MLP_DOWN_SCALE,
        IN_MLP_DOWN_OFFSET,
        IN_MLP_DOWN_DESCALE,
        IN_MLP_DOWN_DEOFFSET,
        IN_MLP_DOWN_0,
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

PagedAttentionQuantHostBinder::PagedAttentionQuantHostBinder() = default;

PagedAttentionQuantHostBinder::~PagedAttentionQuantHostBinder() = default;

void PagedAttentionQuantHostBinder::ParseParam(const nlohmann::json &paramJson)
{
    ATB_LOG(INFO) << "enter DecoderLayerBinder ParseParam tokenOffset";
    seqLen_.clear();
    for (const auto &item : paramJson["seqLen"]) {
        seqLen_.push_back(item.get<int32_t>());
    }

     tokenOffset_.clear();
    for (auto item : paramJson["tokenOffset"]) {
        tokenOffset_.push_back(item.get<int>());
    }

    layerId_ = paramJson["layerId"].get<int>();
}

void PagedAttentionQuantHostBinder::BindTensor(atb::VariantPack &variantPack)
{
    variantPack.inTensors.at(IN_SEQ_LEN).hostData = seqLen_.data();
    variantPack.inTensors.at(IN_TOKEN_OFFSET).hostData = tokenOffset_.data();
    variantPack.inTensors.at(IN_LAYER_ID).hostData = &layerId_;
}
} // namespace baichuan2_7b
} // namespace atb_speed
