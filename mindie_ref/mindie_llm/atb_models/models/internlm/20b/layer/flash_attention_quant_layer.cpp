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
#include "flash_attention_quant_layer.h"

#include "layers/mlp_gate_v2.h"
#include "layers/parallel_layer_v2.h"
#include "models/internlm/7b/operation/rope.h"

namespace atb_speed {
namespace internlm_20b {
enum FlashAttentionQuantLayerTensorId : int {
    IN_HIDDEN_STATES = 0,

    IN_NORM_WEIGHT,
    IN_NORM_BETA,

    // qProj
    IN_Q_LINEAR_WEIGHT,
    IN_Q_LINEAR_DEQSCALE,
    IN_Q_LINEAR_BIAS,

    // kProj
    IN_K_LINEAR_WEIGHT,
    IN_K_LINEAR_DEQSCALE,
    IN_K_LINEAR_BIAS,

    // vProj
    IN_V_LINEAR_WEIGHT,
    IN_V_LINEAR_DEQSCALE,
    IN_V_LINEAR_BIAS,

    // oProj
    IN_SELF_OUT_LINEAR_WEIGHT,
    IN_SELF_OUT_LINEAR_DEQSCALE,
    IN_SELF_OUT_LINEAR_BIAS,

    IN_SELF_OUT_NORM_WEIGHT,
    IN_SELF_OUT_BETA,

    IN_MLP_UP_WEIGHT,
    IN_MLP_UP_DEQSCALE,
    IN_MLP_UP_BIAS,

    IN_MLP_GATE_WEIGHT,
    IN_MLP_GATE_DEQSCALE,
    IN_MLP_GATE_BIAS,

    IN_MLP_DOWN_WEIGHT,
    IN_MLP_DOWN_DEQSCALE,
    IN_MLP_DOWN_BIAS,

    IN_COS_EMBED, // 目前只支持FP16
    IN_SIN_EMBED,

    IN_ATTENTION_MASK,
    IN_PAST_KEY,
    IN_PAST_VALUE,
    IN_TOKEN_OFFSET,
    IN_SEQ_LEN,
    IN_BETA,
    IN_HOLDER,
    IN_LAYER_ID,

    OUT_LAYER_OUT,

    INTERNAL_INPUT_NORM_OUT,
    INTERMIDATE_Q_MIXEDLINEAROUT,
    INTERMIDATE_K_MIXEDLINEAROUT,
    INTERNAL_MIXED_Q,
    INTERNAL_MIXED_K,
    INTERNAL_MIXED_V,
    INTERNAL_SELF_OUT,
    INTERNAL_SELF_LINEAR_OUT,
    INTERNAL_SELF_RESIDUAL_ADD_OUT,
    INTERNAL_SELF_NORM_OUT,
    INTERNAL_MLP_OUT,
};

static const uint64_t IN_TENSOR_COUNT = 36;             // layer层输入tensor数量
static const uint64_t OUT_TENSOR_COUNT = 1;             // layer层输出tensor数量
static const uint64_t INTERNAL_TENSOR_COUNT = 11;       // layer层中间tensor数量
static const uint64_t NODE_COUNT = 11;                  // layer层图节点数量
static const uint64_t ROTARY_COEFF = 2;                 // 书生模型旋转编码参数
static const uint64_t SELF_ATTENTION_V_INPUT_INDEX = 2; // self_attn 传入v的位置
static const uint64_t SELF_ATTENTION_V_INPUT_SIZE = 4;  // self_attn 传入v的shape的size

// 量化layer层
atb::Status FlashAttentionQuantLayer(const FlashAttentionQuantLayerParam &param, atb::Operation **operation)
{
    ATB_LOG(INFO) << __func__ << " called, headNum: " << param.headNum;
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERNAL_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);
    opGraph.name = GetFuncNameAndNameSpace(__PRETTY_FUNCTION__);

    // 初始化模型拓扑图计算节点
    size_t nodeId = 0;
    atb::Node &inputNormNode = opGraph.nodes.at(nodeId++); // input_layernorm
    atb::Node &qLinearNode = opGraph.nodes.at(nodeId++);   // q_proj
    atb::Node &kLinearNode = opGraph.nodes.at(nodeId++);   // k_proj
    atb::Node &vLinearNode = opGraph.nodes.at(nodeId++);   // v_proj
    atb::Node &ropeNode = opGraph.nodes.at(nodeId++);      // apply_rotary_pos_emb
    atb::Node &selfAttentionKvCacheNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfOutLinearNode = opGraph.nodes.at(nodeId++); // o_proj
    atb::Node &selfResidualAddNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfNormNode = opGraph.nodes.at(nodeId++); // post_attention_layernorm
    atb::Node &mlpNode = opGraph.nodes.at(nodeId++);      // mlp
    atb::Node &mlpResidualAddNode = opGraph.nodes.at(nodeId++);

    // 量化input_layernorm，因为书生模型qkv一致，因此量化参数scale和offset从qkv任选其一即可
    atb::infer::RmsNormParam inputNormParam;
    inputNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    inputNormParam.normParam.epsilon = param.rmsNormEps;
    inputNormParam.normParam.quantInputScale = param.qProjInputScale;
    inputNormParam.normParam.quantInputOffset = param.qProjInputOffset;
    inputNormParam.normParam.quantType = atb::infer::QUANT_INT8;
    CREATE_OPERATION(inputNormParam, &inputNormNode.operation);
    inputNormNode.inTensorIds = { IN_HIDDEN_STATES, IN_NORM_WEIGHT, IN_NORM_BETA };
    inputNormNode.outTensorIds = { INTERNAL_INPUT_NORM_OUT }; // int8

    // q_proj
    atb::infer::LinearParam mixedQkvLinearParam;
    mixedQkvLinearParam.linearType = atb::infer::LinearType::LINEAR_INT8INT8_INT32_FP16;
    CREATE_OPERATION(mixedQkvLinearParam, &qLinearNode.operation);
    qLinearNode.inTensorIds = { INTERNAL_INPUT_NORM_OUT, IN_Q_LINEAR_WEIGHT, IN_Q_LINEAR_BIAS, IN_Q_LINEAR_DEQSCALE };
    qLinearNode.outTensorIds = { INTERNAL_MIXED_Q };

    // k_proj
    CREATE_OPERATION(mixedQkvLinearParam, &kLinearNode.operation);
    kLinearNode.inTensorIds = { INTERNAL_INPUT_NORM_OUT, IN_K_LINEAR_WEIGHT, IN_K_LINEAR_BIAS, IN_K_LINEAR_DEQSCALE };
    kLinearNode.outTensorIds = { INTERNAL_MIXED_K };

    // v_proj
    CREATE_OPERATION(mixedQkvLinearParam, &vLinearNode.operation);
    vLinearNode.inTensorIds = { INTERNAL_INPUT_NORM_OUT, IN_V_LINEAR_WEIGHT, IN_V_LINEAR_BIAS, IN_V_LINEAR_DEQSCALE };
    vLinearNode.outTensorIds = { INTERNAL_MIXED_V };

    // 旋转计算
    atb_speed::internlm_7b::RopeParam ropeParam;
    ropeParam.rotaryCoeff = ROTARY_COEFF; // 旋转系数
    ropeParam.headNum = param.headNum;
    atb_speed::internlm_7b::Rope(ropeParam, &ropeNode.operation);
    ropeNode.inTensorIds = { INTERNAL_MIXED_Q, INTERNAL_MIXED_K, IN_COS_EMBED, IN_SIN_EMBED, IN_SEQ_LEN };
    ropeNode.outTensorIds = { INTERMIDATE_Q_MIXEDLINEAROUT, INTERMIDATE_K_MIXEDLINEAROUT };

    // self attention
    atb::infer::SelfAttentionParam selfAttentionParam;
    selfAttentionParam.headNum = param.headNum;
    selfAttentionParam.qScale = 1.0 / sqrt(param.dk);
    CREATE_OPERATION(selfAttentionParam, &selfAttentionKvCacheNode.operation);
    selfAttentionKvCacheNode.inTensorIds = { INTERMIDATE_Q_MIXEDLINEAROUT,
        INTERMIDATE_K_MIXEDLINEAROUT,
        INTERNAL_MIXED_V,
        IN_PAST_KEY,
        IN_PAST_VALUE,
        IN_ATTENTION_MASK,
        IN_TOKEN_OFFSET,
        IN_SEQ_LEN,
        IN_LAYER_ID };
    selfAttentionKvCacheNode.outTensorIds = { INTERNAL_SELF_OUT };
    selfAttentionKvCacheNode.inTensorReshapeFuncs.resize(selfAttentionKvCacheNode.operation->GetInputNum());
    selfAttentionKvCacheNode.inTensorReshapeFuncs[SELF_ATTENTION_V_INPUT_INDEX] = [=](const atb::Dims &oldShape,
        atb::Dims &newShape) {
        newShape.dimNum = SELF_ATTENTION_V_INPUT_SIZE;
        size_t newShapeDimIndex = 0;
        size_t oldShapeDimIndex = 0;
        newShape.dims[newShapeDimIndex++] = oldShape.dims[oldShapeDimIndex++];
        newShape.dims[newShapeDimIndex++] = oldShape.dims[oldShapeDimIndex++];
        newShape.dims[newShapeDimIndex++] = param.headNum;
        newShape.dims[newShapeDimIndex++] = oldShape.dims[oldShapeDimIndex++] / param.headNum;
    };

    // o_proj量化
    atb_speed::common::ParallelParamV2 selfOutLinearParam;
    selfOutLinearParam.commParam.rank = param.rank;
    selfOutLinearParam.commParam.rankSize = param.rankSize;
    selfOutLinearParam.commParam.backend = param.backend;
    selfOutLinearParam.isBias = true;
    selfOutLinearParam.isQuant = true; // true表示使用量化
    selfOutLinearParam.transposeB = true;
    selfOutLinearParam.quantParam.quantType = atb::infer::QUANT_INT8; // 量化int8
    selfOutLinearParam.quantParam.isQuantOp = true;                   // add quant op
    selfOutLinearParam.quantParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_QUANT;
    selfOutLinearParam.quantParam.inputScale = param.oProjInputScale; // 使用o_proj的量化参数
    selfOutLinearParam.quantParam.inputOffset = param.oProjInputOffset;
    atb_speed::common::RowParallelLinearV2(selfOutLinearParam, &selfOutLinearNode.operation);
    selfOutLinearNode.inTensorIds = { INTERNAL_SELF_OUT,
        IN_SELF_OUT_LINEAR_WEIGHT,
        IN_SELF_OUT_LINEAR_BIAS,
        IN_SELF_OUT_LINEAR_DEQSCALE,
        IN_HOLDER,
        IN_HOLDER,
        IN_HOLDER };
    selfOutLinearNode.outTensorIds = { INTERNAL_SELF_LINEAR_OUT }; // float

    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CREATE_OPERATION(addParam, &selfResidualAddNode.operation);
    selfResidualAddNode.inTensorIds = { IN_HIDDEN_STATES, INTERNAL_SELF_LINEAR_OUT };
    selfResidualAddNode.outTensorIds = { INTERNAL_SELF_RESIDUAL_ADD_OUT };

    atb::infer::RmsNormParam selfNormParam;
    selfNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    selfNormParam.normParam.quantInputScale = param.gateProjInputScale;   // gate
    selfNormParam.normParam.quantInputOffset = param.gateProjInputOffset; // gate
    selfNormParam.normParam.quantType = atb::infer::QUANT_INT8;           // 量化int8
    CREATE_OPERATION(selfNormParam, &selfNormNode.operation);
    selfNormNode.inTensorIds = { INTERNAL_SELF_RESIDUAL_ADD_OUT, IN_SELF_OUT_NORM_WEIGHT, IN_SELF_OUT_BETA }; // quant
    selfNormNode.outTensorIds = { INTERNAL_SELF_NORM_OUT };

    // 量化mlp
    atb_speed::common::MlpGateParamV2 mlpParam;
    mlpParam.isBias = true;
    mlpParam.isPack = false;
    mlpParam.isQuant = true;
    mlpParam.transposeB = true;
    mlpParam.commDownParam.rank = param.rank;
    mlpParam.commDownParam.rankSize = param.rankSize;
    mlpParam.commDownParam.backend = param.backend;
    mlpParam.activationType = atb::infer::ActivationType::ACTIVATION_SWISH;
    mlpParam.quantUpParam.quantType = atb::infer::QUANT_INT8;
    mlpParam.quantUpParam.isQuantOp = false;
    mlpParam.quantGateParam.quantType = atb::infer::QUANT_INT8;
    mlpParam.quantGateParam.isQuantOp = false;
    mlpParam.quantDownParam.quantType = atb::infer::QUANT_INT8;
    mlpParam.quantDownParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_QUANT;
    mlpParam.quantDownParam.inputScale = param.downProjInputScale;
    mlpParam.quantDownParam.inputOffset = param.downProjInputOffset;

    atb_speed::common::MlpGateLayerV2(mlpParam, &mlpNode.operation);
    mlpNode.inTensorIds = { INTERNAL_SELF_NORM_OUT,
        IN_MLP_UP_WEIGHT,
        IN_MLP_GATE_WEIGHT,
        IN_MLP_DOWN_WEIGHT,
        IN_MLP_UP_DEQSCALE,
        IN_MLP_GATE_DEQSCALE,
        IN_MLP_DOWN_DEQSCALE,
        IN_MLP_UP_BIAS,
        IN_MLP_GATE_BIAS,
        IN_MLP_DOWN_BIAS,
        IN_HOLDER,
        IN_HOLDER,
        IN_HOLDER,
        IN_HOLDER,
        IN_HOLDER,
        IN_HOLDER,
        IN_HOLDER,
        IN_HOLDER,
        IN_HOLDER }; // float
    mlpNode.outTensorIds = { INTERNAL_MLP_OUT };

    // add
    CREATE_OPERATION(addParam, &mlpResidualAddNode.operation);
    mlpResidualAddNode.inTensorIds = { INTERNAL_SELF_RESIDUAL_ADD_OUT, INTERNAL_MLP_OUT };
    mlpResidualAddNode.outTensorIds = { OUT_LAYER_OUT };

    // 全局tensor reshape
    opGraph.inferShapeFunc = [](const atb::SVector<atb::TensorDesc> &inTensorDescs,
        atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
    };
    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}

FlashAttentionQuantLayerBinder::FlashAttentionQuantLayerBinder() = default;

FlashAttentionQuantLayerBinder::~FlashAttentionQuantLayerBinder() = default;

void FlashAttentionQuantLayerBinder::ParseParam(const nlohmann::json &paramJson)
{
    tokenOffset_.clear();
    for (const auto &item : paramJson["tokenOffset"]) {
        tokenOffset_.push_back(item.get<int>());
    }
    seqLen_.clear();
    for (const auto &item : paramJson["seqLen"]) {
        seqLen_.push_back(item.get<int>());
    }
}

void FlashAttentionQuantLayerBinder::BindTensor(atb::VariantPack &variantPack)
{
    variantPack.inTensors.at(IN_TOKEN_OFFSET).hostData = tokenOffset_.data();
    variantPack.inTensors.at(IN_SEQ_LEN).hostData = seqLen_.data();
}

void from_json(const nlohmann::json &paramJson, FlashAttentionQuantLayerParam &param)
{
    paramJson.at("rmsNormEps").get_to(param.rmsNormEps);
    paramJson.at("headNum").get_to(param.headNum);
    paramJson.at("dk").get_to(param.dk);
    paramJson.at("qProjInputScale").get_to(param.qProjInputScale);
    paramJson.at("qProjInputOffset").get_to(param.qProjInputOffset);
    paramJson.at("kProjInputScale").get_to(param.kProjInputScale);
    paramJson.at("kProjInputOffset").get_to(param.kProjInputOffset);
    paramJson.at("vProjInputScale").get_to(param.vProjInputScale);
    paramJson.at("vProjInputOffset").get_to(param.vProjInputOffset);
    paramJson.at("oProjInputScale").get_to(param.oProjInputScale);
    paramJson.at("oProjInputOffset").get_to(param.oProjInputOffset);
    paramJson.at("gateProjInputScale").get_to(param.gateProjInputScale);
    paramJson.at("gateProjInputOffset").get_to(param.gateProjInputOffset);
    paramJson.at("downProjInputScale").get_to(param.downProjInputScale);
    paramJson.at("downProjInputOffset").get_to(param.downProjInputOffset);
    paramJson.at("upProjInputScale").get_to(param.upProjInputScale);
    paramJson.at("upProjInputOffset").get_to(param.upProjInputOffset);
    if (paramJson.contains("rank")) {
        paramJson.at("rank").get_to(param.rank);
    }
    if (paramJson.contains("rankSize")) {
        paramJson.at("rankSize").get_to(param.rankSize);
    }
    if (paramJson.contains("backend")) {
        paramJson.at("backend").get_to(param.backend);
    }
}

atb::Operation *CreateFlashAttentionQuantLayer(const nlohmann::json &paramJson)
{
    ATB_LOG(INFO) << GetFuncNameAndNameSpace(__PRETTY_FUNCTION__);
    atb::Operation *op;
    atb_speed::internlm_20b::FlashAttentionQuantLayer(paramJson.get<FlashAttentionQuantLayerParam>(), &op);
    return op;
}
} // namespace internlm_20b
} // namespace atb_speed