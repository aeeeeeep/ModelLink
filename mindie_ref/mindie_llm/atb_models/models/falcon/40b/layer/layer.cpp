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
#include "layer.h"
#include "layers/parallel_layer_v2.h"
#include "models/falcon/40b/operation/falcon_rotary_position_embedding_operation.h"
#include "models/falcon/40b/operation/mlp.h" // support configure backend (hccl or lccl)

namespace atb_speed {
namespace falcon_40b {
const int ATTENTION_DIM_NUM = 4;
const int ATTENTION_DIM_2 = 2;
const int ATTENTION_DIM_3 = 3;

enum LayerParallelFlashAttentionTensorId : int {
    IN_NORM_ATTN_WEIGHT = 0,    // 0  ln_attn.weight
    IN_NORM_ATTN_BIAS,          // 1  ln_attn.bias
    IN_NORM_MLP_WEIGHT,         // 2  ln_mlp.weight
    IN_NORM_MLP_BIAS,           // 3  ln_mlp.bias
    IN_QKV_FUSED_WEIGHT,        // 4  self_attention.query_key_value.weight
    IN_ATTN_DENSE_WEIGHT,       // 5  self_attention.dense.weight
    IN_MLP_DENSEWEIGHT_UP,      // 6  mlp.dense_h_to_4h.weight
    IN_MLP_DENSEWEIGHT_DOWN,    // 7  mlp.dense_4h_to_h.weight
    IN_HIDDEN_STATES,
    IN_COS_TABLE,
    IN_SIN_TABLE,
    IN_ATTENTIONMASK,
    IN_CACHE_K,
    IN_CACHE_V,
    IN_TOKENOFFSET,
    IN_SEQLEN,
    IN_LAYERID,
    IN_HOLDER,
    OUT_LAYER_OUT,
    INTERMIDATE_INPUTNORM_OUT_ATTN,
    INTERMIDATE_INPUTNORM_OUT_MLP,
    INTERMIDATE_FUSED_QKV,
    INTERMIDATE_Q_POSITIONEMBED,
    INTERMIDATE_K_POSITIONEMBED,
    INTERMIDATE_VALUE,
    INTERMEDIATE_ATTN_OUTPUT,
    INTERMEDIATE_ATTN_DENSE_OUT,
    INTERMIDATE_MLP_OUT,
    INTERMEDIATE_RES_OUT,
};

static const uint64_t IN_TENSOR_COUNT = 18;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 10;
static const uint64_t NODE_COUNT = 9;

atb::Status LayerParallelFlashAttentionOperation(const LayerParallelFlashAttentionParam &param,
                                                 atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    atb::Node &inputNormAttnNode = opGraph.nodes.at(nodeId++);
    atb::Node &inputNormMlpNode = opGraph.nodes.at(nodeId++);
    atb::Node &fusedQkvLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &rotaryPositionEmbeddingNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfAttentionFusionNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfOutLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpResidualAddNode = opGraph.nodes.at(nodeId++);
    atb::Node &finalResidualAddNode = opGraph.nodes.at(nodeId++);

    atb::infer::LayerNormParam layerNormParam;
    layerNormParam.layerType = atb::infer::LayerNormParam::LAYER_NORM_NORM;
    layerNormParam.normParam.epsilon = param.layerNormEps;
    layerNormParam.normParam.beginNormAxis = 2;
    layerNormParam.normParam.beginParamsAxis = 1;
    CreateOperation(layerNormParam, &inputNormAttnNode.operation);
    inputNormAttnNode.inTensorIds = {IN_HIDDEN_STATES, IN_NORM_ATTN_WEIGHT, IN_NORM_ATTN_BIAS};
    inputNormAttnNode.outTensorIds = {INTERMIDATE_INPUTNORM_OUT_ATTN};

    CreateOperation(layerNormParam, &inputNormMlpNode.operation);
    inputNormMlpNode.inTensorIds = {IN_HIDDEN_STATES, IN_NORM_MLP_WEIGHT, IN_NORM_MLP_BIAS};
    inputNormMlpNode.outTensorIds = {INTERMIDATE_INPUTNORM_OUT_MLP};

    atb::infer::LinearParam fusedQkvLinearParam = {false, false, false};
    CreateOperation(fusedQkvLinearParam, &fusedQkvLinearNode.operation);
    fusedQkvLinearNode.inTensorIds  = {INTERMIDATE_INPUTNORM_OUT_ATTN, IN_QKV_FUSED_WEIGHT};
    fusedQkvLinearNode.outTensorIds = {INTERMIDATE_FUSED_QKV};

    // Rotary Position Embedding 旋转位置编码
    atb_speed::falcon_40b::RotaryPositionEmbeddingParam rotaryPositionEmbeddingParam;
    rotaryPositionEmbeddingParam.hiddenSize = param.hiddenSize;
    rotaryPositionEmbeddingParam.headNum = param.headNum;
    rotaryPositionEmbeddingParam.kvHeadNum = param.kvHeadNum;
    rotaryPositionEmbeddingParam.headDim = param.headDim;
    atb_speed::falcon_40b::RotaryPositionEmbedding(rotaryPositionEmbeddingParam,
                                                   &rotaryPositionEmbeddingNode.operation);
    rotaryPositionEmbeddingNode.inTensorIds = {INTERMIDATE_FUSED_QKV, IN_COS_TABLE, IN_SIN_TABLE, IN_SEQLEN};
    rotaryPositionEmbeddingNode.outTensorIds = {INTERMIDATE_Q_POSITIONEMBED,
                                                INTERMIDATE_K_POSITIONEMBED,
                                                INTERMIDATE_VALUE};
    // RoPE 这边输出得到的 shape 是 batch_size, query_length, num_heads, head_dim

    atb::infer::SelfAttentionParam selfAttentionParam;
    selfAttentionParam.headDim = param.headDim;
    selfAttentionParam.headNum = param.headNum;
    selfAttentionParam.qScale = param.qScale;
    selfAttentionParam.qkScale = param.qkScale;
    selfAttentionParam.isFusion = true;
    selfAttentionParam.withCache = true;
    CreateOperation(selfAttentionParam, &selfAttentionFusionNode.operation);
    selfAttentionFusionNode.inTensorIds = {INTERMIDATE_Q_POSITIONEMBED,
                                           INTERMIDATE_K_POSITIONEMBED,
                                           INTERMIDATE_VALUE,
                                           IN_CACHE_K,
                                           IN_CACHE_V,
                                           IN_ATTENTIONMASK,
                                           IN_TOKENOFFSET,
                                           IN_SEQLEN,
                                           IN_LAYERID};
    selfAttentionFusionNode.outTensorIds = {INTERMEDIATE_ATTN_OUTPUT};
    // attn_output = attn_output.reshape(batch_size, query_length, self.num_heads * self.head_dim)

    // output_tensor = self.dense(attn_output)
    // 这里 dense 之后需要 all_reduce(SUM) 应该使用 parallel
    atb_speed::common::ParallelParamV2 selfOutLinearParam;
    selfOutLinearParam.commParam.rank = param.rank;
    selfOutLinearParam.commParam.rankSize = param.rankSize;
    selfOutLinearParam.commParam.backend = "lccl";
    selfOutLinearParam.isBias = false;
    atb_speed::common::RowParallelLinearV2(selfOutLinearParam, &selfOutLinearNode.operation);
    selfOutLinearNode.inTensorIds = {INTERMEDIATE_ATTN_OUTPUT, IN_ATTN_DENSE_WEIGHT,
                                     IN_HOLDER, IN_HOLDER, IN_HOLDER, IN_HOLDER, IN_HOLDER};
    selfOutLinearNode.outTensorIds = {INTERMEDIATE_ATTN_DENSE_OUT};

    // 最后会自动完成 all_reduce(SUM) 的 Parallel MLP
    // 由于 new_decoder_architecture 和 parallel_attn 都为 True, 所以还是使用 INTERMIDATE_INPUTNORM_OUT_ATTN
    atb_speed::falcon_40b::MlpParam mlpParam;
    mlpParam.rank = param.rank;
    mlpParam.rankSize = param.rankSize;
    mlpParam.activationType = atb::infer::ActivationType::ACTIVATION_GELU;
    mlpParam.transpose = false;
    mlpParam.isBias = false;
    mlpParam.backend = "lccl";
    atb_speed::falcon_40b::MlpLayer(mlpParam, &mlpNode.operation);
    mlpNode.inTensorIds = {INTERMIDATE_INPUTNORM_OUT_MLP, IN_MLP_DENSEWEIGHT_UP, IN_MLP_DENSEWEIGHT_DOWN};
    mlpNode.outTensorIds = {INTERMIDATE_MLP_OUT};

    atb::infer::ElewiseParam mlpResidualAddParam;
    mlpResidualAddParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CreateOperation(mlpResidualAddParam, &mlpResidualAddNode.operation);
    mlpResidualAddNode.inTensorIds = {INTERMEDIATE_ATTN_DENSE_OUT, INTERMIDATE_MLP_OUT};
    mlpResidualAddNode.outTensorIds = {INTERMEDIATE_RES_OUT};

    atb::infer::ElewiseParam finalResidualAddParam;
    finalResidualAddParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CreateOperation(finalResidualAddParam, &finalResidualAddNode.operation);
    finalResidualAddNode.inTensorIds = {INTERMEDIATE_RES_OUT, IN_HIDDEN_STATES};
    finalResidualAddNode.outTensorIds = {OUT_LAYER_OUT};

    opGraph.nodes.resize(nodeId);

    opGraph.inferShapeFunc = [&](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(IN_HIDDEN_STATES);
        return atb::NO_ERROR;
    };
    atb::CreateOperation(opGraph, operation);
    return atb::NO_ERROR;
}

LayerPrallelFlashAttentionBinder::LayerPrallelFlashAttentionBinder() {}

LayerPrallelFlashAttentionBinder::~LayerPrallelFlashAttentionBinder() {}

void LayerPrallelFlashAttentionBinder::ParseParam(const nlohmann::json &paramJson)
{
    tokenOffset_.clear();
    for (auto item : paramJson["tokenOffset"]) {
        tokenOffset_.push_back(item.get<int>());
    }

    seqLen_.clear();
    for (auto item : paramJson["seqLen"]) {
        seqLen_.push_back(item.get<int>());
    }
    layerId_ = paramJson["layerId"].get<int>();
}

void LayerPrallelFlashAttentionBinder::BindTensor(atb::VariantPack &variantPack)
{
    const uint32_t tokenOffsetTensorId = IN_TOKENOFFSET;
    const uint32_t seqLenTensorId = IN_SEQLEN;
    const uint32_t layerIdTensorId = IN_LAYERID;
    variantPack.inTensors.at(tokenOffsetTensorId).hostData = tokenOffset_.data();
    variantPack.inTensors.at(seqLenTensorId).hostData = seqLen_.data();
    variantPack.inTensors.at(layerIdTensorId).hostData = &layerId_;
}

} // namespace falcon_40b
} // namespace atb_speed