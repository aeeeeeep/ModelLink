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
#include "models/mixtral/7b_dense/layer/decoder_layer.h"
#include "mixtral/7b_dense/operation/mixtral_dense_moe.h"

namespace atb_speed {
namespace mixtralDense {

static const uint64_t IN_TENSOR_COUNT = 61;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT_WITH_EXPERT = 5;
static const uint64_t NODE_COUNT_WITH_EXPERT = 6;

atb::Status DecoderLayer(const DecoderLayerParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT_WITH_EXPERT;
    opGraph.nodes.resize(NODE_COUNT_WITH_EXPERT);
    opGraph.name = param.isPrefill ? "Prefill_layer" : "Decoder_layer";

    size_t nodeId = 0;
    atb::Node &attentionNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfResidualAddNode = opGraph.nodes.at(nodeId++);

    atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam> fusionAttentionParam;
    // QKV linear param
    fusionAttentionParam.isGroupedQueryAttention = param.numAttentionHeadsPerRank != param.numKeyValueHeadsPerRank;
    fusionAttentionParam.isBF16 = param.isBF16;
    fusionAttentionParam.layerLinearQuantType = param.linearQuantType;
    fusionAttentionParam.packQuantType = param.packQuantType[0];
    fusionAttentionParam.supportLcoc = param.supportLcoc;
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
    fusionAttentionParam.pageAttentionParam.maskType = param.isBF16 ? \
        atb::infer::PagedAttentionParam::MaskType::MASK_TYPE_ALIBI : atb::infer::PagedAttentionParam::MaskType::UNDEFINED;
    fusionAttentionParam.selfOutLinearTensorParallelInfo = {param.rank, param.worldSize, param.backend, param.rankTableFile};
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
        IN_ATTENTION_OUT_COMPRESS_IDX,
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

    atb::Node &selfNormNode = opGraph.nodes.at(nodeId++);
    atb::infer::RmsNormParam selfNormParam;
    selfNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    selfNormParam.normParam.epsilon = param.rmsNormEps;
    CreateOperation(selfNormParam, &selfNormNode.operation);
    if (selfNormNode.operation == nullptr) {
        ATB_LOG(ERROR) << "selfNormNode op is nullptr: ";
    }
    selfNormNode.inTensorIds = {INTERMEDIATE_RESIDUAL_ADD_OUT, IN_SELFATTENTION_OUT_NORM_WEIGHT};
    selfNormNode.outTensorIds = {INTERMIDATE_SELFATTENTION_NORM_OUT};
    ATB_LOG(INFO) << "create post rmsnorm";

    atb::Node &moeNode = opGraph.nodes.at(nodeId++);
    atb_speed::mixtralDense::MixtralDenseMoeParam mixtralDenseMoeParam;
    mixtralDenseMoeParam.transpose = param.transpose;
    mixtralDenseMoeParam.numOfExperts = param.numOfExperts;
    mixtralDenseMoeParam.num = param.numOfSelectedExperts;
    mixtralDenseMoeParam.expertParallelDegree = param.expertParallelDegree;
    mixtralDenseMoeParam.maskStartIdx = param.maskStartIdx;
    mixtralDense::CreateMixtralDenseMoeOperation(mixtralDenseMoeParam, &moeNode.operation);
    if (moeNode.operation == nullptr) {
        ATB_LOG(ERROR) << "MixtralDenseMoe op is nullptr: ";
    }
    moeNode.inTensorIds = {
        INTERMIDATE_SELFATTENTION_NORM_OUT,
        IN_BLOCK_SPARSE_MOE_GATE_WEIGHT,
        IN_MLP_GATEUP_WEIGHT_EXPERT_ZERO,
        IN_MLP_DOWN_WEIGHT_EXPERT_ZERO,
        IN_MLP_GATEUP_WEIGHT_EXPERT_ONE,
        IN_MLP_DOWN_WEIGHT_EXPERT_ONE,
        IN_MLP_GATEUP_WEIGHT_EXPERT_TWO,
        IN_MLP_DOWN_WEIGHT_EXPERT_TWO,
        IN_MLP_GATEUP_WEIGHT_EXPERT_THREE,
        IN_MLP_DOWN_WEIGHT_EXPERT_THREE,
        IN_MLP_GATEUP_WEIGHT_EXPERT_FOUR,
        IN_MLP_DOWN_WEIGHT_EXPERT_FOUR,
        IN_MLP_GATEUP_WEIGHT_EXPERT_FIVE,
        IN_MLP_DOWN_WEIGHT_EXPERT_FIVE,
        IN_MLP_GATEUP_WEIGHT_EXPERT_SIX,
        IN_MLP_DOWN_WEIGHT_EXPERT_SIX,
        IN_MLP_GATEUP_WEIGHT_EXPERT_SEVEN,
        IN_MLP_DOWN_WEIGHT_EXPERT_SEVEN,
        IN_ONE_HOT_ONE,
        IN_ONE_HOT_ZERO,
        IN_FINAL_HIDDEN_STATE};
    moeNode.outTensorIds = {INTERMIDATE_MOE_OUT};
    ATB_LOG(INFO) << "Moe Dense calculation success";

    atb::Node &moeAllReduceNode = opGraph.nodes.at(nodeId++);
    atb::infer::AllReduceParam allReduceParam;
    allReduceParam.rank = param.rank;
    allReduceParam.rankSize = param.worldSize;
    allReduceParam.backend = param.backend;
    allReduceParam.rankTableFile = param.rankTableFile;
    CreateOperation(allReduceParam, &moeAllReduceNode.operation);
    if (moeAllReduceNode.operation == nullptr) {
        ATB_LOG(ERROR) << "moeAllReduceNode op is nullptr: ";
    }
    moeAllReduceNode.inTensorIds = {INTERMIDATE_MOE_OUT};
    moeAllReduceNode.outTensorIds = {INTERMEDIATE_MLP_OUT};
    ATB_LOG(INFO) << "create all reduce";

    atb::Node &mlpResidualAddNode = opGraph.nodes.at(nodeId++);
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
    ATB_LOG(INFO) << "decoder layer: residule create opgraph";

    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}

DecoderLayerBinder::DecoderLayerBinder() {}

DecoderLayerBinder::~DecoderLayerBinder() {}

} // namespace mixtralDense
} // namespace atb_speed