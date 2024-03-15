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
#include "flash_attention_layer.h"

#include "layers/mlp_gate_v2.h"
#include "layers/parallel_layer_v2.h"
#include "atb_speed/utils/operation_factory.h"

namespace atb_speed {
namespace baichuan2_13b {

REGISTER_OPERATION(baichuan2_13b, CreateFlashAttentionLayer);

enum FlashAttentionLayerTensorId : int {
    IN_HIDDEN_STATES = 0,
    IN_NORM_WEIGHT,
    IN_QKV_MIXED_LINEAR_WEIGHT,
    IN_SELF_OUT_LINEAR_WEIGHT,
    IN_SELF_OUT_NORM_WEIGHT,
    IN_MLP_GATE_WEIGHT,
    IN_MLP_DOWN_WEIGHT,
    IN_MLP_UP_WEIGHT,
    IN_ATTENTION_MASK,
    IN_PAST_KEY,
    IN_PAST_VALUE,
    IN_TOKEN_OFFSET,
    IN_SEQ_LEN,
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

static const uint64_t IN_TENSOR_COUNT = 15;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERNAL_TENSOR_COUNT = 10;
static const uint64_t NODE_COUNT = 9;

void from_json(const nlohmann::json &paramJson, FlashAttentionLayerParam &param)
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
    if (paramJson.contains("backend")) {
        paramJson.at("backend").get_to(param.backend);
    }
}

atb::Operation *CreateFlashAttentionLayer(const nlohmann::json &paramJson)
{
    ATB_LOG(INFO) << GetFuncNameAndNameSpace(__PRETTY_FUNCTION__);
    atb::Operation *op;
    atb_speed::baichuan2_13b::FlashAttentionLayer(paramJson.get<FlashAttentionLayerParam>(), &op);
    return op;
}
atb::Status FlashAttentionLayer(const FlashAttentionLayerParam &param, atb::Operation **operation)
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

    atb::infer::RmsNormParam rmsNormParam;
    rmsNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    rmsNormParam.normParam.epsilon = param.rmsNormEps;
    CREATE_OPERATION(rmsNormParam, &inputNormNode.operation);
    inputNormNode.inTensorIds = {IN_HIDDEN_STATES, IN_NORM_WEIGHT};
    inputNormNode.outTensorIds = {INTERNAL_INPUT_NORM_OUT};

    atb::infer::LinearParam linearParam;
    linearParam.hasBias = false;
    CREATE_OPERATION(linearParam, &qkvLinearNode.operation);
    qkvLinearNode.inTensorIds = {INTERNAL_INPUT_NORM_OUT, IN_QKV_MIXED_LINEAR_WEIGHT};
    qkvLinearNode.outTensorIds = {INTERNAL_QKV_MIXED_LINEAR_OUT};

    atb::infer::SplitParam splitParam = {2, 3};
    CREATE_OPERATION(splitParam, &splitQKVNode.operation);
    splitQKVNode.inTensorIds = {INTERNAL_QKV_MIXED_LINEAR_OUT};
    splitQKVNode.outTensorIds = {INTERNAL_MIXED_Q, INTERNAL_MIXED_K, INTERNAL_MIXED_V};

    atb::infer::SelfAttentionParam selfAttentionParam;
    selfAttentionParam.headNum = param.headNum;
    selfAttentionParam.qScale = 1.0 / sqrt(param.dk);
    selfAttentionParam.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_ALIBI;
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
    selfOutLinearParam.isBias = false;
    atb_speed::common::RowParallelLinearV2(selfOutLinearParam, &selfOutLinearNode.operation);
    selfOutLinearNode.inTensorIds = {
        INTERNAL_SELF_OUT, IN_SELF_OUT_LINEAR_WEIGHT, IN_HOLDER, IN_HOLDER, IN_HOLDER, IN_HOLDER, IN_HOLDER,
    };
    selfOutLinearNode.outTensorIds = {INTERNAL_SELF_LINEAR_OUT};

    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CREATE_OPERATION(addParam, &selfResidualAddNode.operation);
    selfResidualAddNode.inTensorIds = {IN_HIDDEN_STATES, INTERNAL_SELF_LINEAR_OUT};
    selfResidualAddNode.outTensorIds = {INTERNAL_SELF_RESIDUAL_ADD_OUT};

    CREATE_OPERATION(rmsNormParam, &selfNormNode.operation);
    selfNormNode.inTensorIds = {INTERNAL_SELF_RESIDUAL_ADD_OUT, IN_SELF_OUT_NORM_WEIGHT};
    selfNormNode.outTensorIds = {INTERNAL_SELF_NORM_OUT};

    atb_speed::common::MlpGateParamV2 mlpParam;
    mlpParam.commDownParam.rank = param.rank;
    mlpParam.commDownParam.rankSize = param.rankSize;
    mlpParam.commDownParam.backend = param.backend;
    mlpParam.activationType = atb::infer::ActivationType::ACTIVATION_SWISH;
    mlpParam.transposeB = true;
    mlpParam.isBias = false;
    mlpParam.isPack = false;
    atb_speed::common::MlpGateLayerV2(mlpParam, &mlpNode.operation);
    mlpNode.inTensorIds = {INTERNAL_SELF_NORM_OUT,
                           IN_MLP_UP_WEIGHT,
                           IN_MLP_GATE_WEIGHT,
                           IN_MLP_DOWN_WEIGHT,
                           IN_HOLDER,
                           IN_HOLDER,
                           IN_HOLDER,
                           IN_HOLDER,
                           IN_HOLDER,
                           IN_HOLDER,
                           IN_HOLDER,
                           IN_HOLDER,
                           IN_HOLDER,
                           IN_HOLDER,
                           IN_HOLDER,
                           IN_HOLDER,
                           IN_HOLDER,
                           IN_HOLDER,
                           IN_HOLDER};
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

FlashAttentionLayerBinder::FlashAttentionLayerBinder() = default;

FlashAttentionLayerBinder::~FlashAttentionLayerBinder() = default;

void FlashAttentionLayerBinder::ParseParam(const nlohmann::json &paramJson)
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

void FlashAttentionLayerBinder::BindTensor(atb::VariantPack &variantPack)
{
    variantPack.inTensors.at(IN_TOKEN_OFFSET).hostData = tokenOffset_.data();
    variantPack.inTensors.at(IN_SEQ_LEN).hostData = seqLen_.data();
}
} // namespace baichuan2_13b
} // namespace atb_speed