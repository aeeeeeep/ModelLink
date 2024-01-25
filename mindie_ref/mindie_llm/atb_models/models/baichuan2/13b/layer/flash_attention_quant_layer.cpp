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

namespace atb_speed {
namespace baichuan2_13b {
enum FlashAttentionQuantLayerTensorId : int {
    IN_HIDDEN_STATES = 0,

    IN_NORM_WEIGHT,
    // w_pack
    IN_QKV_MIXED_LINEAR_WEIGHT,
    IN_QKV_MIXED_DEQSCALE,
    IN_QKV_MIXED_BIAS,
    // o_proj
    IN_SELF_OUT_LINEAR_WEIGHT,
    IN_SELF_OUT_LINEAR_DEQSCALE,
    IN_SELF_OUT_LINEAR_BIAS,

    IN_MLP_UP_WEIGHT,
    IN_MLP_UP_DEQSCALE,
    IN_MLP_UP_BIAS,

    IN_MLP_GATE_WEIGHT,
    IN_MLP_GATE_DEQSCALE,
    IN_MLP_GATE_BIAS,

    IN_MLP_DOWN_WEIGHT,
    IN_MLP_DOWN_DEQSCALE,
    IN_MLP_DOWN_BIAS,

    IN_SELF_OUT_NORM_WEIGHT,

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
    INTERNAL_QKV_MIXED_LINEAR_OUT,
    INTERNAL_MIXED_Q,
    INTERNAL_MIXED_K,
    INTERNAL_MIXED_V,
    INTERNAL_SELF_OUT,
    INTERNAL_SELF_LINEAR_OUT,
    INTERNAL_SELF_RESIDUAL_ADD_OUT,
    INTERNAL_SELF_NORM_OUT,
    INTERNAL_MLP_OUT,
};

static const uint64_t IN_TENSOR_COUNT = 26;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERNAL_TENSOR_COUNT = 10;
static const uint64_t NODE_COUNT = 9;

void from_json(const nlohmann::json &paramJson, FlashAttentionQuantLayerParam &param)
{
    paramJson.at("rmsNormEps").get_to(param.rmsNormEps);
    paramJson.at("headNum").get_to(param.headNum);
    paramJson.at("dk").get_to(param.dk);

    paramJson.at("w_packInputScale").get_to(param.wPackInputScale);
    paramJson.at("w_packInputOffset").get_to(param.wPackInputOffset);
    paramJson.at("o_projInputScale").get_to(param.oProjInputScale);
    paramJson.at("o_projInputOffset").get_to(param.oProjInputOffset);
    paramJson.at("gate_projInputScale").get_to(param.gateProjInputScale);
    paramJson.at("gate_projInputOffset").get_to(param.gateProjInputOffset);
    paramJson.at("down_projInputScale").get_to(param.downProjInputScale);
    paramJson.at("down_projInputOffset").get_to(param.downProjInputOffset);
    paramJson.at("up_projInputScale").get_to(param.upProjInputScale);
    paramJson.at("up_projInputOffset").get_to(param.upProjInputOffset);
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
    atb_speed::baichuan2_13b::FlashAttentionQuantLayer(paramJson.get<FlashAttentionQuantLayerParam>(), &op);
    return op;
}

atb::Status FlashAttentionQuantLayer(const FlashAttentionQuantLayerParam &param, atb::Operation **operation)
{
    ATB_LOG(INFO) << __func__ << " called, headNum: " << param.headNum;
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERNAL_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);
    opGraph.name = GetFuncNameAndNameSpace(__PRETTY_FUNCTION__);

    size_t nodeId = 0;
    atb::Node &inputNormNode = opGraph.nodes.at(nodeId++);
    atb::Node &qkvLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &splitQKVNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfAttentionKvCacheNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfOutLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfResidualAddNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfNormNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpResidualAddNode = opGraph.nodes.at(nodeId++);

    atb::infer::RmsNormParam inputNormParam;
    inputNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    inputNormParam.normParam.epsilon = param.rmsNormEps;
    inputNormParam.normParam.quantInputScale = param.wPackInputScale;
    inputNormParam.normParam.quantInputOffset = param.wPackInputOffset;
    inputNormParam.normParam.quantType = atb::infer::QUANT_INT8;
    CREATE_OPERATION(inputNormParam, &inputNormNode.operation);
    inputNormNode.inTensorIds = {IN_HIDDEN_STATES, IN_NORM_WEIGHT, IN_BETA};
    inputNormNode.outTensorIds = {INTERNAL_INPUT_NORM_OUT}; // int8

    atb::infer::LinearQuantParam mixedQkvLinearParam;
    mixedQkvLinearParam.transposeB = true;
    CREATE_OPERATION(mixedQkvLinearParam, &qkvLinearNode.operation);
    qkvLinearNode.inTensorIds = {INTERNAL_INPUT_NORM_OUT, IN_QKV_MIXED_LINEAR_WEIGHT, IN_QKV_MIXED_BIAS,
                                 IN_QKV_MIXED_DEQSCALE};
    qkvLinearNode.outTensorIds = {INTERNAL_QKV_MIXED_LINEAR_OUT}; // float

    atb::infer::SplitParam splitParam = {2, 3};
    CREATE_OPERATION(splitParam, &splitQKVNode.operation);
    splitQKVNode.inTensorIds = {INTERNAL_QKV_MIXED_LINEAR_OUT};
    splitQKVNode.outTensorIds = {INTERNAL_MIXED_Q, INTERNAL_MIXED_K, INTERNAL_MIXED_V};

    atb::infer::SelfAttentionParam selfAttentionParam;
    selfAttentionParam.headDim = param.dk;
    selfAttentionParam.headNum = param.headNum;
    selfAttentionParam.qScale = 1.0 / sqrt(param.dk);
    selfAttentionParam.isSupportAlibi = true;
    CREATE_OPERATION(selfAttentionParam, &selfAttentionKvCacheNode.operation);
    selfAttentionKvCacheNode.inTensorIds = {INTERNAL_MIXED_Q, INTERNAL_MIXED_K, INTERNAL_MIXED_V,
                                            IN_PAST_KEY,      IN_PAST_VALUE,    IN_ATTENTION_MASK,
                                            IN_TOKEN_OFFSET,  IN_SEQ_LEN,       IN_LAYER_ID};
    selfAttentionKvCacheNode.outTensorIds = {INTERNAL_SELF_OUT};

    selfAttentionKvCacheNode.inTensorReshapeFuncs.resize(selfAttentionKvCacheNode.operation->GetInputNum());
    selfAttentionKvCacheNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 4;
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = oldShape.dims[1];
        newShape.dims[2] = param.headNum;
        newShape.dims[3] = oldShape.dims[2] / param.headNum;
    };
    selfAttentionKvCacheNode.inTensorReshapeFuncs[1] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 4;
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = oldShape.dims[1];
        newShape.dims[2] = param.headNum;
        newShape.dims[3] = oldShape.dims[2] / param.headNum;
    };
    selfAttentionKvCacheNode.inTensorReshapeFuncs[2] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 4;
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = oldShape.dims[1];
        newShape.dims[2] = param.headNum;
        newShape.dims[3] = oldShape.dims[2] / param.headNum;
    };

    atb_speed::common::ParallelParamV2 selfOutLinearParam;
    selfOutLinearParam.commParam.rank = param.rank;
    selfOutLinearParam.commParam.rankSize = param.rankSize;
    selfOutLinearParam.commParam.backend = param.backend;
    selfOutLinearParam.isBias = true;
    selfOutLinearParam.isQuant = true;
    selfOutLinearParam.transposeB = true;
    selfOutLinearParam.quantParam.quantType = atb::infer::QUANT_INT8;
    selfOutLinearParam.quantParam.isQuantOp = true; // add quant op
    selfOutLinearParam.quantParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_QUANT;
    selfOutLinearParam.quantParam.inputScale = param.oProjInputScale;
    selfOutLinearParam.quantParam.inputOffset = param.oProjInputOffset;
    atb_speed::common::RowParallelLinearV2(selfOutLinearParam, &selfOutLinearNode.operation);
    selfOutLinearNode.inTensorIds = {INTERNAL_SELF_OUT,
                                     IN_SELF_OUT_LINEAR_WEIGHT,
                                     IN_SELF_OUT_LINEAR_BIAS,
                                     IN_SELF_OUT_LINEAR_DEQSCALE,
                                     IN_HOLDER,
                                     IN_HOLDER,
                                     IN_HOLDER};
    selfOutLinearNode.outTensorIds = {INTERNAL_SELF_LINEAR_OUT}; // float

    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CREATE_OPERATION(addParam, &selfResidualAddNode.operation);
    selfResidualAddNode.inTensorIds = {IN_HIDDEN_STATES, INTERNAL_SELF_LINEAR_OUT};
    selfResidualAddNode.outTensorIds = {INTERNAL_SELF_RESIDUAL_ADD_OUT};

    atb::infer::RmsNormParam selfNormParam;
    selfNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    selfNormParam.normParam.quantInputScale = param.gateProjInputScale;   // gate up
    selfNormParam.normParam.quantInputOffset = param.gateProjInputOffset; // gate up
    selfNormParam.normParam.quantType = atb::infer::QUANT_INT8;
    CREATE_OPERATION(selfNormParam, &selfNormNode.operation);
    selfNormNode.inTensorIds = {INTERNAL_SELF_RESIDUAL_ADD_OUT, IN_SELF_OUT_NORM_WEIGHT, IN_BETA}; // quant
    selfNormNode.outTensorIds = {INTERNAL_SELF_NORM_OUT};

    atb_speed::common::MlpGateParamV2 mlpParam;
    mlpParam.isBias = true;
    mlpParam.isPack = false;
    mlpParam.isQuant = true;
    mlpParam.transposeB = true;
    mlpParam.commDownParam.rank = param.rank;
    mlpParam.commDownParam.rankSize = param.rankSize;
    mlpParam.commDownParam.backend = param.backend;
    mlpParam.activationType = atb::infer::ActivationType::ACTIVATION_SWISH;
    // add quant op
    mlpParam.quantDownParam.quantType = atb::infer::QUANT_INT8;
    mlpParam.quantDownParam.isQuantOp = true;
    mlpParam.quantDownParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_QUANT;
    mlpParam.quantDownParam.inputScale = param.downProjInputScale;
    mlpParam.quantDownParam.inputOffset = param.downProjInputOffset;

    atb_speed::common::MlpGateLayerV2(mlpParam, &mlpNode.operation);
    mlpNode.inTensorIds = {INTERNAL_SELF_NORM_OUT,
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
                           IN_HOLDER}; // float
    mlpNode.outTensorIds = {INTERNAL_MLP_OUT};

    CREATE_OPERATION(addParam, &mlpResidualAddNode.operation);
    mlpResidualAddNode.inTensorIds = {INTERNAL_SELF_RESIDUAL_ADD_OUT, INTERNAL_MLP_OUT};
    mlpResidualAddNode.outTensorIds = {OUT_LAYER_OUT};

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
} // namespace baichuan2_13b
} // namespace atb_speed