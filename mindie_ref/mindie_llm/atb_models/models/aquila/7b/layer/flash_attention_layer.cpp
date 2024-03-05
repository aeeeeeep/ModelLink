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
#include "flash_attention_layer.h"

#include "layers/mlp_gate_v2.h"
#include "layers/parallel_layer_v2.h"
#include "models/aquila/7b/operation/rope.h"

namespace atb_speed {
namespace aquila_7b {
enum FlashAttentionRopeLayerTensorId : int {
    IN_HIDDENSTATES = 0,  // [batchSize, seqLen, hiddenSize]
    IN_NORMWEIGHT,
    IN_QLINEARWEIGHT,
    IN_KLINEARWEIGHT,
    IN_VLINEARWEIGHT,
    IN_SELFOUTLINEARWEIGHT,
    IN_SELFOUTNORMWEIGHT,
    IN_MLPGATEWEIGHT,
    IN_MLPDOWNWEIGHT,
    IN_MLPUPWEIGHT,
    IN_COS_EMBED,  // 目前只支持FP16
    IN_SIN_EMBED,
    IN_ATTENTIONMASK,
    IN_PASTKEY,
    IN_PASTVALUE,
    IN_TOKENOFFSET,
    IN_SEQLEN,
    IN_HOLDER,
    IN_LAYERID,
    OUT_LAYEROUT,
    INTERNAL_INPUTNORMOUT,
    INTERNAL_QLINEAROUT,
    INTERNAL_KLINEAROUT,
    INTERNAL_VLINEAROUT,
    INTERNAL_QEMBED,
    INTERNAL_KEMBED,
    INTERNAL_SELFOUT,
    INTERNAL_SELFLINEAROUT,
    INTERNAL_ATTENTIONRESIDUALADDOUT,
    INTERNAL_SELFNORMOUT,
    INTERNAL_MLPOUT,
};

static const uint64_t IN_TENSOR_COUNT = 19;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 11;
static const uint64_t NODE_COUNT = 11;

void from_json(const nlohmann::json &paramJson, FlashAttentionRopeLayerParam &param)
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

atb::Operation *CreateFlashAttentionRopeLayer(const nlohmann::json &paramJson)
{
    ATB_LOG(INFO) << GetFuncNameAndNameSpace(__PRETTY_FUNCTION__);
    atb::Operation *op;
    atb_speed::aquila_7b::FlashAttentionRopeLayer(paramJson.get<FlashAttentionRopeLayerParam>(), &op);
    return op;
}

atb::Status FlashAttentionRopeLayer(const FlashAttentionRopeLayerParam &param, atb::Operation **operation)
{
    ATB_LOG(INFO) << __func__ << " called, headNum: " << param.headNum;
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);
    opGraph.name = GetFuncNameAndNameSpace(__PRETTY_FUNCTION__);

    size_t nodeId = 0;
    atb::Node &inputNormNode = opGraph.nodes.at(nodeId++);
    atb::Node &qLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &kLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &vLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &ropeNode = opGraph.nodes.at(nodeId++);
    atb::Node &flashAttentionNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfOutLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &attentionResidualAddNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfNormNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpResidualAddNode = opGraph.nodes.at(nodeId++);

    // input_layernorm
    atb::infer::RmsNormParam rmsNormParam;
    rmsNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    rmsNormParam.normParam.epsilon = param.rmsNormEps;
    CREATE_OPERATION(rmsNormParam, &inputNormNode.operation);
    inputNormNode.inTensorIds = {IN_HIDDENSTATES, IN_NORMWEIGHT};
    inputNormNode.outTensorIds = {INTERNAL_INPUTNORMOUT};

    // q_proj
    atb::infer::LinearParam linearParam;
    linearParam.hasBias = false;
    CREATE_OPERATION(linearParam, &qLinearNode.operation);
    qLinearNode.inTensorIds = {INTERNAL_INPUTNORMOUT, IN_QLINEARWEIGHT};
    qLinearNode.outTensorIds = {INTERNAL_QLINEAROUT};
    
    // k_proj
    CREATE_OPERATION(linearParam, &kLinearNode.operation);
    kLinearNode.inTensorIds = {INTERNAL_INPUTNORMOUT, IN_KLINEARWEIGHT};
    kLinearNode.outTensorIds = {INTERNAL_KLINEAROUT};

    // v_proj
    CREATE_OPERATION(linearParam, &vLinearNode.operation);
    vLinearNode.inTensorIds = {INTERNAL_INPUTNORMOUT, IN_VLINEARWEIGHT};
    vLinearNode.outTensorIds = {INTERNAL_VLINEAROUT};

    // rope (q_embedding + k_embedding)
    atb_speed::aquila_7b::RopeParam ropeParam;
    ropeParam.rotaryCoeff = 2;
    ropeParam.headNum = param.headNum;
    atb_speed::aquila_7b::Rope(ropeParam, &ropeNode.operation);
    ropeNode.inTensorIds = {INTERNAL_QLINEAROUT, INTERNAL_KLINEAROUT, IN_COS_EMBED, IN_SIN_EMBED, IN_SEQLEN};
    ropeNode.outTensorIds = {INTERNAL_QEMBED, INTERNAL_KEMBED};

    // flash attention
    atb::infer::SelfAttentionParam selfAttentionParam;
    selfAttentionParam.headNum = param.headNum;
    selfAttentionParam.qScale = 1.0 / sqrt(param.dk);
    selfAttentionParam.clampMin = -1024.0;
    selfAttentionParam.clampMax = 1024.0;
    selfAttentionParam.clampType = atb::infer::SelfAttentionParam::CLAMP_TYPE_MIN_MAX;
    CREATE_OPERATION(selfAttentionParam, &flashAttentionNode.operation);
    flashAttentionNode.inTensorIds = {INTERNAL_QEMBED,
                                      INTERNAL_KEMBED,
                                      INTERNAL_VLINEAROUT,
                                      IN_PASTKEY,
                                      IN_PASTVALUE,
                                      IN_ATTENTIONMASK,
                                      IN_TOKENOFFSET,
                                      IN_SEQLEN,
                                      IN_LAYERID};
    flashAttentionNode.outTensorIds = {INTERNAL_SELFOUT};
    flashAttentionNode.inTensorReshapeFuncs.resize(flashAttentionNode.inTensorIds.size());
    flashAttentionNode.inTensorReshapeFuncs.at(2) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 4;
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = oldShape.dims[1];
        newShape.dims[2] = param.headNum;
        newShape.dims[3] = oldShape.dims[2] / param.headNum;
    };

    // o_proj
    atb_speed::common::ParallelParamV2 selfOutLinearParam;
    selfOutLinearParam.commParam.rank = param.rank;
    selfOutLinearParam.commParam.rankSize = param.rankSize;
    selfOutLinearParam.commParam.backend = param.backend;
    selfOutLinearParam.isBias = false;
    atb_speed::common::RowParallelLinearV2(selfOutLinearParam, &selfOutLinearNode.operation);
    selfOutLinearNode.inTensorIds = {
        INTERNAL_SELFOUT, IN_SELFOUTLINEARWEIGHT, IN_HOLDER, IN_HOLDER, IN_HOLDER, IN_HOLDER, IN_HOLDER};
    selfOutLinearNode.outTensorIds = {INTERNAL_SELFLINEAROUT};

    // residual
    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CREATE_OPERATION(addParam, &attentionResidualAddNode.operation);
    attentionResidualAddNode.inTensorIds = {IN_HIDDENSTATES, INTERNAL_SELFLINEAROUT};
    attentionResidualAddNode.outTensorIds = {INTERNAL_ATTENTIONRESIDUALADDOUT};

    // post_attention_layernorm
    CREATE_OPERATION(rmsNormParam, &selfNormNode.operation);
    selfNormNode.inTensorIds = {INTERNAL_ATTENTIONRESIDUALADDOUT, IN_SELFOUTNORMWEIGHT};
    selfNormNode.outTensorIds = {INTERNAL_SELFNORMOUT};

    // mlp
    atb_speed::common::MlpGateParamV2 mlpParam;
    mlpParam.commDownParam.rank = param.rank;
    mlpParam.commDownParam.rankSize = param.rankSize;
    mlpParam.commDownParam.backend = param.backend;
    mlpParam.activationType = atb::infer::ActivationType::ACTIVATION_SWISH;
    mlpParam.transposeB = true;
    mlpParam.isBias = false;
    mlpParam.isPack = false;
    atb_speed::common::MlpGateLayerV2(mlpParam, &mlpNode.operation);
    mlpNode.inTensorIds = {
        INTERNAL_SELFNORMOUT,
        IN_MLPUPWEIGHT,
        IN_MLPGATEWEIGHT,
        IN_MLPDOWNWEIGHT,
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
        IN_HOLDER,
    };
    mlpNode.outTensorIds = {INTERNAL_MLPOUT};

    // residual
    CREATE_OPERATION(addParam, &mlpResidualAddNode.operation);
    mlpResidualAddNode.inTensorIds = {INTERNAL_ATTENTIONRESIDUALADDOUT, INTERNAL_MLPOUT};
    mlpResidualAddNode.outTensorIds = {OUT_LAYEROUT};

    opGraph.inferShapeFunc = [](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
    };
    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}

FlashAttentionRopeLayerBinder::FlashAttentionRopeLayerBinder() = default;

FlashAttentionRopeLayerBinder::~FlashAttentionRopeLayerBinder() = default;

void FlashAttentionRopeLayerBinder::ParseParam(const nlohmann::json &paramJson)
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

void FlashAttentionRopeLayerBinder::BindTensor(atb::VariantPack &variantPack)
{
    variantPack.inTensors.at(IN_TOKENOFFSET).hostData = tokenOffset_.data();
    variantPack.inTensors.at(IN_SEQLEN).hostData = seqLen_.data();
}
} // namespace aquila_7b
} // namespace atb_speed