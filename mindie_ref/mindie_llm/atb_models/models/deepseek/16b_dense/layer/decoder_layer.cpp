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
#include "models/deepseek/16b_dense/layer/decoder_layer.h"
#include "deepseek/16b_dense/operation/deepseek_dense_moe.h"
#include "deepseek/16b_dense/operation/deepseek_dense_mlp_without_expert.h"

namespace atb_speed {
namespace deepseekDense {

static const uint64_t IN_TENSOR_COUNT = 171;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT_NO_EXPERT = 5;
static const uint64_t NODE_COUNT_NO_EXPERT = 6;
static const uint64_t INTERMEDIATE_TENSOR_COUNT_WITH_EXPERT = 7;
static const uint64_t NODE_COUNT_WITH_EXPERT = 8;

atb::Status DecoderLayer(const DecoderLayerParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    if (param.layerId == 0) {
        opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT_NO_EXPERT;
        opGraph.nodes.resize(NODE_COUNT_NO_EXPERT);
    } else {
        opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT_WITH_EXPERT;
        opGraph.nodes.resize(NODE_COUNT_WITH_EXPERT);
    }
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

    if (param.layerId == 0) {
        atb::Node &mlpExpertNode = opGraph.nodes.at(nodeId++);
        atb_speed::deepseekDense::DeepseekDenseMlpWithoutExpertParam mlpExpertParam;
        mlpExpertParam.transpose = param.transpose;
        deepseekDense::CreateDeepseekDenseMlpWithoutExpertOperation(
            mlpExpertParam, &mlpExpertNode.operation);
        mlpExpertNode.inTensorIds = {INTERMIDATE_SELFATTENTION_NORM_OUT,
                                    IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT,
                                    IN_MLP_DOWN_WEIGHT_EXPERT_SHARED_EXPERT};
        mlpExpertNode.outTensorIds = {INTERMIDATE_MOE_OUT_ALL};
        ATB_LOG(INFO) << "shared expert calculation success";
    } else {
        atb::Node &moeNode = opGraph.nodes.at(nodeId++);
        atb_speed::deepseekDense::DeepseekDenseMoeParam deepseekDenseMoeParam;
        deepseekDenseMoeParam.transpose = param.transpose;
        deepseekDenseMoeParam.numOfExperts = param.numOfExperts;
        deepseekDenseMoeParam.num = param.numOfSelectedExperts;
        deepseekDenseMoeParam.expertParallelDegree = param.expertParallelDegree;
        deepseekDenseMoeParam.maskStartIdx = param.maskStartIdx;
        deepseekDense::CreateDeepseekDenseMoeOperation(deepseekDenseMoeParam, &moeNode.operation);
        if (moeNode.operation == nullptr) {
            ATB_LOG(ERROR) << "DeepseekDenseMoe op is nullptr: ";
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
            IN_MLP_GATEUP_WEIGHT_EXPERT_EIGHT,
            IN_MLP_DOWN_WEIGHT_EXPERT_EIGHT,
            IN_MLP_GATEUP_WEIGHT_EXPERT_NINE,
            IN_MLP_DOWN_WEIGHT_EXPERT_NINE,
            IN_MLP_GATEUP_WEIGHT_EXPERT_TEN,
            IN_MLP_DOWN_WEIGHT_EXPERT_TEN,
            IN_MLP_GATEUP_WEIGHT_EXPERT_ELEVEN,
            IN_MLP_DOWN_WEIGHT_EXPERT_ELEVEN,
            IN_MLP_GATEUP_WEIGHT_EXPERT_TWELVE,
            IN_MLP_DOWN_WEIGHT_EXPERT_TWELVE,
            IN_MLP_GATEUP_WEIGHT_EXPERT_THIRTEEN,
            IN_MLP_DOWN_WEIGHT_EXPERT_THIRTEEN,
            IN_MLP_GATEUP_WEIGHT_EXPERT_FOURTEEN,
            IN_MLP_DOWN_WEIGHT_EXPERT_FOURTEEN,
            IN_MLP_GATEUP_WEIGHT_EXPERT_FIFTEENN,
            IN_MLP_DOWN_WEIGHT_EXPERT_FIFTEEN,
            IN_MLP_GATEUP_WEIGHT_EXPERT_SIXTEEN,
            IN_MLP_DOWN_WEIGHT_EXPERT_SIXTEEN,
            IN_MLP_GATEUP_WEIGHT_EXPERT_SEVENTEEN,
            IN_MLP_DOWN_WEIGHT_EXPERT_SEVENTEEN,
            IN_MLP_GATEUP_WEIGHT_EXPERT_EIGHTEEN,
            IN_MLP_DOWN_WEIGHT_EXPERT_EIGHTEEN,
            IN_MLP_GATEUP_WEIGHT_EXPERT_NINETEEN,
            IN_MLP_DOWN_WEIGHT_EXPERT_NINETEEN,
            IN_MLP_GATEUP_WEIGHT_EXPERT_TWENTY,
            IN_MLP_DOWN_WEIGHT_EXPERT_TWENTY,
            IN_MLP_GATEUP_WEIGHT_EXPERT_TWENTY_ONE,
            IN_MLP_DOWN_WEIGHT_EXPERT_TWENTY_ONE,
            IN_MLP_GATEUP_WEIGHT_EXPERT_TWENTY_TWO,
            IN_MLP_DOWN_WEIGHT_EXPERT_TWENTY_TWO,
            IN_MLP_GATEUP_WEIGHT_EXPERT_TWENTY_THREE,
            IN_MLP_DOWN_WEIGHT_EXPERT_TWENTY_THREE,
            IN_MLP_GATEUP_WEIGHT_EXPERT_TWENTY_FOUR,
            IN_MLP_DOWN_WEIGHT_EXPERT_TWENTY_FOUR,
            IN_MLP_GATEUP_WEIGHT_EXPERT_TWENTY_FIVE,
            IN_MLP_DOWN_WEIGHT_EXPERT_TWENTY_FIVE,
            IN_MLP_GATEUP_WEIGHT_EXPERT_TWENTY_SIX,
            IN_MLP_DOWN_WEIGHT_EXPERT_TWENTY_SIX,
            IN_MLP_GATEUP_WEIGHT_EXPERT_TWENTY_SEVEN,
            IN_MLP_DOWN_WEIGHT_EXPERT_TWENTY_SEVEN,
            IN_MLP_GATEUP_WEIGHT_EXPERT_TWENTY_EIGHT,
            IN_MLP_DOWN_WEIGHT_EXPERT_TWENTY_EIGHT,
            IN_MLP_GATEUP_WEIGHT_EXPERT_TWENTY_NINE,
            IN_MLP_DOWN_WEIGHT_EXPERT_TWENTY_NINE,
            IN_MLP_GATEUP_WEIGHT_EXPERT_THIRTY,
            IN_MLP_DOWN_WEIGHT_EXPERT_THIRTY,
            IN_MLP_GATEUP_WEIGHT_EXPERT_THIRTY_ONE,
            IN_MLP_DOWN_WEIGHT_EXPERT_THIRTY_ONE,
            IN_MLP_GATEUP_WEIGHT_EXPERT_THIRTY_TWO,
            IN_MLP_DOWN_WEIGHT_EXPERT_THIRTY_TWO,
            IN_MLP_GATEUP_WEIGHT_EXPERT_THIRTY_THREE,
            IN_MLP_DOWN_WEIGHT_EXPERT_THIRTY_THREE,
            IN_MLP_GATEUP_WEIGHT_EXPERT_THIRTY_FOUR,
            IN_MLP_DOWN_WEIGHT_EXPERT_THIRTY_FOUR,
            IN_MLP_GATEUP_WEIGHT_EXPERT_THIRTY_FIVE,
            IN_MLP_DOWN_WEIGHT_EXPERT_THIRTY_FIVE,
            IN_MLP_GATEUP_WEIGHT_EXPERT_THIRTY_SIX,
            IN_MLP_DOWN_WEIGHT_EXPERT_THIRTY_SIX,
            IN_MLP_GATEUP_WEIGHT_EXPERT_THIRTY_SEVEN,
            IN_MLP_DOWN_WEIGHT_EXPERT_THIRTY_SEVEN,
            IN_MLP_GATEUP_WEIGHT_EXPERT_THIRTY_EIGHT,
            IN_MLP_DOWN_WEIGHT_EXPERT_THIRTY_EIGHT,
            IN_MLP_GATEUP_WEIGHT_EXPERT_THIRTY_NINE,
            IN_MLP_DOWN_WEIGHT_EXPERT_THIRTY_NINE,
            IN_MLP_GATEUP_WEIGHT_EXPERT_FOURTY,
            IN_MLP_DOWN_WEIGHT_EXPERT_FOURTY,
            IN_MLP_GATEUP_WEIGHT_EXPERT_FOURTY_ONE,
            IN_MLP_DOWN_WEIGHT_EXPERT_FOURTY_ONE,
            IN_MLP_GATEUP_WEIGHT_EXPERT_FOURTY_TWO,
            IN_MLP_DOWN_WEIGHT_EXPERT_FOURTY_TWO,
            IN_MLP_GATEUP_WEIGHT_EXPERT_FOURTY_THREE,
            IN_MLP_DOWN_WEIGHT_EXPERT_FOURTY_THREE,
            IN_MLP_GATEUP_WEIGHT_EXPERT_FOURTY_FOUR,
            IN_MLP_DOWN_WEIGHT_EXPERT_FOURTY_FOUR,
            IN_MLP_GATEUP_WEIGHT_EXPERT_FOURTY_FIVE,
            IN_MLP_DOWN_WEIGHT_EXPERT_FOURTY_FIVE,
            IN_MLP_GATEUP_WEIGHT_EXPERT_FOURTY_SIX,
            IN_MLP_DOWN_WEIGHT_EXPERT_FOURTY_SIX,
            IN_MLP_GATEUP_WEIGHT_EXPERT_FOURTY_SEVEN,
            IN_MLP_DOWN_WEIGHT_EXPERT_FOURTY_SEVEN,
            IN_MLP_GATEUP_WEIGHT_EXPERT_FOURTY_EIGHT,
            IN_MLP_DOWN_WEIGHT_EXPERT_FOURTY_EIGHT,
            IN_MLP_GATEUP_WEIGHT_EXPERT_FOURTY_NINE,
            IN_MLP_DOWN_WEIGHT_EXPERT_FOURTY_NINE,
            IN_MLP_GATEUP_WEIGHT_EXPERT_FIFTY,
            IN_MLP_DOWN_WEIGHT_EXPERT_FIFTY,
            IN_MLP_GATEUP_WEIGHT_EXPERT_FIFTY_ONE,
            IN_MLP_DOWN_WEIGHT_EXPERT_FIFTY_ONE,
            IN_MLP_GATEUP_WEIGHT_EXPERT_FIFTY_TWO,
            IN_MLP_DOWN_WEIGHT_EXPERT_FIFTY_TWO,
            IN_MLP_GATEUP_WEIGHT_EXPERT_FIFTY_THREE,
            IN_MLP_DOWN_WEIGHT_EXPERT_FIFTY_THREE,
            IN_MLP_GATEUP_WEIGHT_EXPERT_FIFTY_FOUR,
            IN_MLP_DOWN_WEIGHT_EXPERT_FIFTY_FOUR,
            IN_MLP_GATEUP_WEIGHT_EXPERT_FIFTY_FIVE,
            IN_MLP_DOWN_WEIGHT_EXPERT_FIFTY_FIVE,
            IN_MLP_GATEUP_WEIGHT_EXPERT_FIFTY_SIX,
            IN_MLP_DOWN_WEIGHT_EXPERT_FIFTY_SIX,
            IN_MLP_GATEUP_WEIGHT_EXPERT_FIFTY_SEVEN,
            IN_MLP_DOWN_WEIGHT_EXPERT_FIFTY_SEVEN,
            IN_MLP_GATEUP_WEIGHT_EXPERT_FIFTY_EIGHT,
            IN_MLP_DOWN_WEIGHT_EXPERT_FIFTY_EIGHT,
            IN_MLP_GATEUP_WEIGHT_EXPERT_FIFTY_NINEE,
            IN_MLP_DOWN_WEIGHT_EXPERT_FIFTY_NINE,
            IN_MLP_GATEUP_WEIGHT_EXPERT_SIXTY,
            IN_MLP_DOWN_WEIGHT_EXPERT_SIXTY,
            IN_MLP_GATEUP_WEIGHT_EXPERT_SIXTY_ONE,
            IN_MLP_DOWN_WEIGHT_EXPERT_SIXTY_ONE,
            IN_MLP_GATEUP_WEIGHT_EXPERT_SIXTY_TWO,
            IN_MLP_DOWN_WEIGHT_EXPERT_SIXTY_TWO,
            IN_MLP_GATEUP_WEIGHT_EXPERT_SIXTY_THREE,
            IN_MLP_DOWN_WEIGHT_EXPERT_SIXTY_THREE,
            IN_ONE_HOT_ONE,
            IN_ONE_HOT_ZERO,
            IN_FINAL_HIDDEN_STATE};
        moeNode.outTensorIds = {INTERMIDATE_MOE_OUT};
        ATB_LOG(INFO) << "Moe Dense calculation success";

        atb::Node &sharedMlpExpertNode = opGraph.nodes.at(nodeId++);
        atb_speed::deepseekDense::DeepseekDenseMlpWithoutExpertParam sharedMlpExpertParam;
        sharedMlpExpertParam.transpose = param.transpose;
        deepseekDense::CreateDeepseekDenseMlpWithoutExpertOperation(
            sharedMlpExpertParam, &sharedMlpExpertNode.operation);
        sharedMlpExpertNode.inTensorIds = {INTERMIDATE_SELFATTENTION_NORM_OUT,
                                    IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT,
                                    IN_MLP_DOWN_WEIGHT_EXPERT_SHARED_EXPERT};
        sharedMlpExpertNode.outTensorIds = {INTERMIDATE_HIDDEN_STATE_SHARED_EXPERTS};
        ATB_LOG(INFO) << "shared expert calculation success";

        atb::Node &mlpAddNode = opGraph.nodes.at(nodeId++);
        atb::infer::ElewiseParam sharedMlpAddParam;
        sharedMlpAddParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
        CreateOperation(sharedMlpAddParam, &mlpAddNode.operation);
        mlpAddNode.inTensorIds = {INTERMIDATE_HIDDEN_STATE_SHARED_EXPERTS, INTERMIDATE_MOE_OUT};
        mlpAddNode.outTensorIds = {INTERMIDATE_MOE_OUT_ALL};
    };

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
    moeAllReduceNode.inTensorIds = {INTERMIDATE_MOE_OUT_ALL};
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

} // namespace deepseekDense
} // namespace atb_speed