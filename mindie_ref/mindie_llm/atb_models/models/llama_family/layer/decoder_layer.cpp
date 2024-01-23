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

#include "models/llama_family/operation/rms_norm.h"
#include "models/llama_family/operation/linear.h"
#include "models/llama_family/operation/linear_parallel.h"
#include "models/llama_family/operation/attention.h"
#include "models/llama_family/operation/mlp.h"
#include "models/llama_family/layer/decoder_layer.h"

namespace atb_speed {
namespace llama_family {

static const uint64_t IN_TENSOR_COUNT = 43;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 5;
static const uint64_t NODE_COUNT = 6;

atb::Status DecoderLayer(const DecoderLayerParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);
    opGraph.name = param.isPrefill ? "Prefill_layer" : "Decoder_layer";

    size_t nodeId = 0;
    atb::Node &inputNormNode = opGraph.nodes.at(nodeId++);
    atb::Node &attentionNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfResidualAddNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfNormNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpParallelNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpResidualAddNode = opGraph.nodes.at(nodeId++);

    atb_speed::llama_family::FusionRmsNormParam fusionRmsNormParam;
    fusionRmsNormParam.quantType = param.quantType;
    fusionRmsNormParam.rmsNormEps = param.rmsNormEps;
    FusionRmsNorm(fusionRmsNormParam, &inputNormNode.operation);
    inputNormNode.inTensorIds = {IN_HIDDEN_STATES, IN_INPUT_NORM_WEIGHT, IN_BETA};
    inputNormNode.outTensorIds = {INTERMEDIATE_INPUT_NORM_OUT};

    atb_speed::llama_family::FusionAttentionParam fusionAttentionParam;
    // QKV linear param
    fusionAttentionParam.isPack = param.isPack;
    fusionAttentionParam.isGroupedQueryAttention = param.numAttentionHeadsPerRank != param.numKeyValueHeadsPerRank;
    fusionAttentionParam.qkvLinearParam.quantType = param.quantType;
    // rope param
    fusionAttentionParam.rotaryCoeff = 2;
    // self attention param
    fusionAttentionParam.isFA = param.isFA;
    fusionAttentionParam.isPrefill = param.isPrefill;
    fusionAttentionParam.isBF16 = param.isBF16;
    fusionAttentionParam.selfAttentionParam.headNum = param.numAttentionHeadsPerRank;
    fusionAttentionParam.selfAttentionParam.kvHeadNum = param.numKeyValueHeadsPerRank;
    fusionAttentionParam.selfAttentionParam.headDim = param.hiddenSizePerAttentionHead;
    fusionAttentionParam.selfAttentionParam.qkScale = 1.0 / sqrt(param.hiddenSizePerAttentionHead);
    if (param.isFA) {
        fusionAttentionParam.selfAttentionParam.isTriuMask = param.isPrefill ? 1 : 0;
        fusionAttentionParam.selfAttentionParam.coderType = param.isPrefill ? atb::infer::SelfAttentionParam::CoderType::ENCODER : atb::infer::SelfAttentionParam::CoderType::DECODER;
    } else {
        fusionAttentionParam.selfAttentionParam.isEncoder = param.isPrefill;
    }
    fusionAttentionParam.pageAttentionParam.headNum = param.numAttentionHeadsPerRank;
    fusionAttentionParam.pageAttentionParam.kvHeadNum = param.numKeyValueHeadsPerRank;
    fusionAttentionParam.pageAttentionParam.qkScale = 1.0 / sqrt(param.hiddenSizePerAttentionHead);
    fusionAttentionParam.pageAttentionParam.isSupportAlibi = param.isBF16;
    // self out linear param
    fusionAttentionParam.selfOutLinearParallelParam.parallelType = atb_speed::llama_family::ROW_PARALLEL;
    fusionAttentionParam.selfOutLinearParallelParam.fusionLinearParam.quantType = param.quantType;
    fusionAttentionParam.selfOutLinearParallelParam.rank = param.rank;
    fusionAttentionParam.selfOutLinearParallelParam.worldSize = param.worldSize;
    fusionAttentionParam.selfOutLinearParallelParam.backend = param.backend;
    atb_speed::llama_family::FusionAttention fusionAttentionObj;
    fusionAttentionObj.Attention(fusionAttentionParam, &attentionNode.operation);
    attentionNode.inTensorIds = {
        INTERMEDIATE_INPUT_NORM_OUT,
        IN_QKV_WEIGHT_0,
        IN_QKV_SCALE_0,
        IN_QKV_OFFSET_0,
        IN_QKV_DESCALE_0,
        IN_QKV_WEIGHT_1,
        IN_QKV_SCALE_1,
        IN_QKV_OFFSET_1,
        IN_QKV_DESCALE_1,
        IN_QKV_WEIGHT_2,
        IN_QKV_SCALE_2,
        IN_QKV_OFFSET_2,
        IN_QKV_DESCALE_2,
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
    };
    attentionNode.outTensorIds = {INTERMEDIATE_ATTENTION_OUT};

    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CreateOperation(addParam, &selfResidualAddNode.operation);
    selfResidualAddNode.inTensorIds = {
        IN_HIDDEN_STATES,
        INTERMEDIATE_ATTENTION_OUT
    };
    selfResidualAddNode.outTensorIds = {INTERMEDIATE_RESIDUAL_ADD_OUT};

    FusionRmsNorm(fusionRmsNormParam, &selfNormNode.operation);
    selfNormNode.inTensorIds = {INTERMEDIATE_RESIDUAL_ADD_OUT, IN_ATTENTION_NORM_WEIGHT, IN_BETA};
    selfNormNode.outTensorIds = {INTERMEDIATE_ATTENTION_NORM_OUT};

    atb_speed::llama_family::MlpParam mlpParam;
    mlpParam.isPack = param.isPack;
    mlpParam.gateUpLinearParam.quantType = param.quantType;
    mlpParam.downLinearParallelParam.fusionLinearParam.quantType = param.quantType;
    mlpParam.downLinearParallelParam.parallelType = atb_speed::llama_family::ROW_PARALLEL;
    mlpParam.downLinearParallelParam.rank = param.rank;
    mlpParam.downLinearParallelParam.worldSize = param.worldSize;
    mlpParam.downLinearParallelParam.backend = param.backend;
    Mlp(mlpParam, &mlpParallelNode.operation);
    mlpParallelNode.inTensorIds = {
        INTERMEDIATE_ATTENTION_NORM_OUT,
        IN_MLP_WEIGHT_0,
        IN_MLP_SCALE_0,
        IN_MLP_OFFSET_0,
        IN_MLP_DESCALE_0,
        IN_MLP_WEIGHT_1,
        IN_MLP_SCALE_1,
        IN_MLP_OFFSET_1,
        IN_MLP_DESCALE_1,
        IN_MLP_DOWN_WEIGHT,
        IN_MLP_DOWN_SCALE,
        IN_MLP_DOWN_OFFSET,
        IN_MLP_DOWN_DESCALE,
    };
    mlpParallelNode.outTensorIds = {INTERMEDIATE_MLP_OUT};

    CreateOperation(addParam, &mlpResidualAddNode.operation);
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

    atb::CreateOperation(opGraph, operation);
    return atb::NO_ERROR;
}

DecoderLayerBinder::DecoderLayerBinder() {}

DecoderLayerBinder::~DecoderLayerBinder() {}

void DecoderLayerBinder::ParseParam(const nlohmann::json &paramJson)
{
    ATB_LOG(INFO) << "enter DecoderLayerBinder ParseParam tokenOffset";
    seqLen_.clear();
    for (auto item : paramJson["seqLen"]) {
        seqLen_.push_back(item.get<int>());
    }

    tokenOffset_.clear();
    for (auto item : paramJson["tokenOffset"]) {
        tokenOffset_.push_back(item.get<int>());
    }

    layerId_ = paramJson["layerId"].get<int>();
}

void DecoderLayerBinder::BindTensor(atb::VariantPack &variantPack)
{
    ATB_LOG(INFO) << "enter DecoderLayerOperation BindTensor";
    variantPack.inTensors.at(IN_SEQ_LEN).hostData = seqLen_.data();
    variantPack.inTensors.at(IN_TOKEN_OFFSET).hostData = tokenOffset_.data();
    variantPack.inTensors.at(IN_LAYER_ID).hostData = &layerId_;
}

} // namespace llama_family
} // namespace atb_speed