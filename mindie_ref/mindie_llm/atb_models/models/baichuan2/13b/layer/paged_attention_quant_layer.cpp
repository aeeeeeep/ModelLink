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
#include "layers/operations/linear.h"
#include "layers/operations/linear_parallel.h"
#include "layers/operations/norm_linear.h"
#include "layers/operations/fusion_attention.h"
#include "layers/operations/mlp.h"
#include "layers/operations/mlp_swiglu.h"
#include "models/baichuan2/13b/layer/paged_attention_quant_layer.h"

namespace atb_speed {
namespace baichuan2_13b {
static const uint64_t IN_TENSOR_COUNT = 51;  // 部分冗余tensor预先保留，
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERNAL_TENSOR_COUNT = 3;
static const uint64_t NODE_COUNT = 4;


atb::Status PAQuantLayer(const PAQuantLayerParam &param, atb::Operation **operation)
{
    // 算子图
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERNAL_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);

    if (param.isPrefill) {
        opGraph.name = "Prefill_transformer_layer";
    } else {
        opGraph.name = "Decoder_transformer_layer";
    }

    size_t nodeId = 0;
    atb::Node &attentionNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfResidualAddNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpParallelNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpResidualAddNode = opGraph.nodes.at(nodeId++);

    // 步骤1：.attentionNode。功能：rmsNorm+qkv+attention+o_proj(对于baichuan而言，没有rope。位置编码信息从python传来)。注意：attention根据不同芯片类型shape有所区分
    atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam> fusionAttentionParam;
    // QKV linear param
    fusionAttentionParam.isGroupedQueryAttention = param.numAttentionHeadsPerRank != param.numKeyValueHeadsPerRank;
    fusionAttentionParam.isBF16 = param.isBF16;
    fusionAttentionParam.packQuantType = param.packQuantType[0];
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
    // rope这一块如何设置
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
    fusionAttentionParam.selfAttentionParam.isTriuMask = param.isPrefill ? 1 : 0;
    fusionAttentionParam.selfAttentionParam.qkScale = 1.0 / sqrt(param.hiddenSizePerAttentionHead);
    if (param.isFA) {
        fusionAttentionParam.selfAttentionParam.calcType = param.isPrefill ? \
            atb::infer::SelfAttentionParam::CalcType::ENCODER : atb::infer::SelfAttentionParam::CalcType::DECODER;
    } else {
        fusionAttentionParam.selfAttentionParam.calcType = atb::infer::SelfAttentionParam::CalcType::PA_ENCODER;
    }
    fusionAttentionParam.selfAttentionParam.maskType = atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_ALIBI;
    fusionAttentionParam.pageAttentionParam.headNum = param.numAttentionHeadsPerRank;
    fusionAttentionParam.pageAttentionParam.kvHeadNum = param.numKeyValueHeadsPerRank;
    fusionAttentionParam.pageAttentionParam.qkScale = 1.0 / sqrt(param.hiddenSizePerAttentionHead);
    fusionAttentionParam.pageAttentionParam.maskType = atb::infer::PagedAttentionParam::MaskType::MASK_TYPE_ALIBI;
    
    fusionAttentionParam.selfOutLinearTensorParallelInfo = {param.rank, param.worldSize, param.backend};
    Attention(fusionAttentionParam, &attentionNode.operation);
    attentionNode.inTensorIds = {
        LayerQuantPATensorId::IN_HIDDEN_STATES,
        IN_INPUT_NORM_WEIGHT,
        IN_INPUT_NORM_BIAS,
        IN_INPUT_NORM_NEW_WEIGHT,
        IN_INPUT_NORM_NEW_BIAS,
        IN_QKV_WEIGHT_0,
        IN_QKV_SCALE_0,
        IN_QKV_OFFSET_0,
        IN_QKV_DESCALE_0,
        IN_QKV_DEOFFSET_0,
        IN_QKV_WEIGHT_1,
        IN_QKV_SCALE_1,
        IN_QKV_OFFSET_1,
        IN_QKV_DESCALE_1,
        IN_QKV_DEOFFSET_1,
        IN_QKV_WEIGHT_2,
        IN_QKV_SCALE_2,
        IN_QKV_OFFSET_2,
        IN_QKV_DESCALE_2,
        IN_QKV_DEOFFSET_2,
        IN_PLACE_HOLDER,
        IN_PLACE_HOLDER,
        IN_INPUT_LENGTHS,
        IN_K_CACHE,
        IN_V_CACHE,
        IN_ATTENTION_MASK,
        IN_PLACE_HOLDER,
        IN_PLACE_HOLDER,
        IN_BLOCK_TABLES,
        IN_SLOTS,
        IN_ATTENTION_OUT_WEIGHT,
        IN_ATTENTION_OUT_SCALE,
        IN_ATTENTION_OUT_OFFSET,
        IN_ATTENTION_OUT_DESCALE,
        IN_ATTENTION_OUT_DEOFFSET,
    };
    attentionNode.outTensorIds = {INTERMEDIATE_ATTENTION_OUT};
    // 步骤2：残差结构
    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CREATE_OPERATION(addParam, &selfResidualAddNode.operation);
    selfResidualAddNode.inTensorIds = {
        IN_HIDDEN_STATES,
        INTERMEDIATE_ATTENTION_OUT
    };
    selfResidualAddNode.outTensorIds = {INTERMEDIATE_RESIDUAL_ADD_OUT};
    // 步骤3：mlp操作
    atb_speed::common::MlpParam<atb::infer::RmsNormParam> mlpParam;
    mlpParam.isBF16 = param.isBF16;
    mlpParam.packQuantType = param.packQuantType[1];
    mlpParam.layerLinearQuantType = param.linearQuantType;
    // gate up
    if (param.packQuantType[1] == atb_speed::common::MIX_W8A8 || param.packQuantType[1] == atb_speed::common::MIX_W8A8_ANTI) {
        mlpParam.mlpPackType = atb_speed::common::GATE_UP_WEIGHT_NO_PACK;
    } else {
        mlpParam.mlpPackType = atb_speed::common::GATE_UP_WEIGHT_PACK;
    }
    atb::infer::RmsNormParam mlpRmsNormParam;
    mlpRmsNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    mlpRmsNormParam.normParam.epsilon = param.rmsNormEps; 
    mlpParam.normParamType = mlpRmsNormParam;
    atb::infer::RmsNormParam mlpRmsNormQuantParam;
    mlpRmsNormQuantParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    mlpRmsNormQuantParam.normParam.epsilon = param.rmsNormEps;
    mlpRmsNormQuantParam.normParam.quantType = atb::infer::QUANT_INT8;
    mlpParam.normQuantParamType = mlpRmsNormQuantParam;
    // down（与attention后的linear一致）
    mlpParam.downLinearTensorParallelInfo = {param.rank, param.worldSize, param.backend};

    if (param.supportSwiGLU) {
        mlpParam.activationParam.activationType = atb::infer::ActivationType::ACTIVATION_SWIGLU_FORWARD;
        mlpParam.activationParam.dim = -1;
        MlpSwiGLU(mlpParam, &mlpParallelNode.operation);
    } else {
        mlpParam.activationParam.activationType = atb::infer::ActivationType::ACTIVATION_SWISH;
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
        IN_MLP_DEOFFSET_0,
        IN_MLP_WEIGHT_1,
        IN_MLP_SCALE_1,
        IN_MLP_OFFSET_1,
        IN_MLP_DESCALE_1,
        IN_MLP_DEOFFSET_1,
        IN_MLP_DOWN_WEIGHT,
        IN_MLP_DOWN_SCALE,
        IN_MLP_DOWN_OFFSET,
        IN_MLP_DOWN_DESCALE,
        IN_MLP_DOWN_DEOFFSET,
    };
    mlpParallelNode.outTensorIds = {INTERMEDIATE_MLP_OUT};
    // 步骤4：残差结构
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

PAQuantLayerHostBinder::PAQuantLayerHostBinder() = default;
PAQuantLayerHostBinder::~PAQuantLayerHostBinder() = default;

void PAQuantLayerHostBinder::ParseParam(const nlohmann::json &paramJson)
{
    ATB_LOG(INFO) << "enter PAQuantLayerHostBinder ParseParam tokenOffset";
    seqLen_.clear();
    for (auto item : paramJson["seqLen"]) {
        seqLen_.push_back(item.get<int>());
    }
}

void PAQuantLayerHostBinder::BindTensor(atb::VariantPack &variantPack)
{
    ATB_LOG(INFO) << "enter DecoderLayerOperation BindTensor";
    variantPack.inTensors.at(IN_INPUT_LENGTHS).hostData = seqLen_.data();
}
} // namespace baichuan2_13b
} // namespace atb_speed