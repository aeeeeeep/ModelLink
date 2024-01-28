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
#include "flash_attention_rope_layer.h"

#include "layers/mlp_gate.h"
#include "layers/parallel_layer.h"
#include "models/internlm/7b/operation/rope.h"

namespace atb_speed {
namespace internlm_7b {
enum FlashAttentionRopeLayerTensorId : int {
    IN_HIDDENSTATES = 0,
    IN_NORMWEIGHT,
    IN_Q_LINEARWEIGHT,
    IN_Q_LINEARBIAS,
    IN_K_LINEARWEIGHT,
    IN_K_LINEARBIAS,
    IN_V_LINEARWEIGHT,
    IN_V_LINEARBIAS,
    IN_SELFOUTLINEARWEIGHT,
    IN_SELFOUT_LINEARBIAS,
    IN_SELFOUTNORMWEIGHT,
    IN_MLPGATEWEIGHT,
    IN_MLPDOWNWEIGHT,
    IN_MLPUPWEIGHT,
    IN_COS_EMBED, // 目前只支持FP16
    IN_SIN_EMBED,
    IN_ATTENTIONMASK,
    IN_PASTKEY,
    IN_PASTVALUE,
    IN_TOKENOFFSET,
    IN_SEQLEN,
    IN_LAYERID,
    OUT_LAYEROUT,
    INTERMIDATE_INPUTNORMOUT,
    INTERMIDATE_Q_MIXEDLINEAROUT,
    INTERMIDATE_K_MIXEDLINEAROUT,
    INTERMIDATE_V_MIXEDLINEAROUT,
    INTERMIDATE_Q_POSITIONEMBED,
    INTERMIDATE_K_POSITIONEMBED,
    INTERMIDATE_SELFOUT,
    INTERMIDATE_SELFLINEAROUT,
    INTERMIDATE_SELFRESIDUALADDOUT,
    INTERMIDATE_SELFNORMOUT,
    INTERMIDATE_MLPOUT,
};

static const uint64_t IN_TENSOR_COUNT = 22;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 11;
static const uint64_t NODE_COUNT = 11;
static const uint64_t SELF_ATTENTION_V_INPUT_INDEX = 2;
static const uint64_t SELF_ATTENTION_V_INPUT_SIZE = 4;
static const uint64_t ROTARY_COEFF = 2;

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
    // (bsz,seq_len,hidden_size) - > (bsz,seq_len,hidden_size)
    inputNormNode.inTensorIds = {IN_HIDDENSTATES, IN_NORMWEIGHT};
    inputNormNode.outTensorIds = {INTERMIDATE_INPUTNORMOUT};

    atb::infer::LinearParam linearParam = {false, false, true};
    CREATE_OPERATION(linearParam, &qLinearNode.operation);
    qLinearNode.inTensorIds = {INTERMIDATE_INPUTNORMOUT, IN_Q_LINEARWEIGHT, IN_Q_LINEARBIAS};
    qLinearNode.outTensorIds = {INTERMIDATE_Q_MIXEDLINEAROUT};

    CREATE_OPERATION(linearParam, &kLinearNode.operation);
    kLinearNode.inTensorIds = {INTERMIDATE_INPUTNORMOUT, IN_K_LINEARWEIGHT, IN_K_LINEARBIAS};
    kLinearNode.outTensorIds = {INTERMIDATE_K_MIXEDLINEAROUT};

    CREATE_OPERATION(linearParam, &vLinearNode.operation);
    vLinearNode.inTensorIds = {INTERMIDATE_INPUTNORMOUT, IN_V_LINEARWEIGHT, IN_V_LINEARBIAS};
    vLinearNode.outTensorIds = {INTERMIDATE_V_MIXEDLINEAROUT};

    atb_speed::internlm_7b::RopeParam ropeParam;
    ropeParam.rotaryCoeff = ROTARY_COEFF; // 旋转系数
    ropeParam.headNum = param.headNum;
    atb_speed::internlm_7b::Rope(ropeParam, &ropeNode.operation);
    ropeNode.inTensorIds = {INTERMIDATE_Q_MIXEDLINEAROUT, INTERMIDATE_K_MIXEDLINEAROUT, IN_COS_EMBED, IN_SIN_EMBED,
                            IN_SEQLEN};
    ropeNode.outTensorIds = {INTERMIDATE_Q_POSITIONEMBED, INTERMIDATE_K_POSITIONEMBED};

    atb::infer::SelfAttentionParam selfAttentionParam;
    selfAttentionParam.isFusion = true;
    selfAttentionParam.headDim = param.dk;
    selfAttentionParam.headNum = param.headNum;
    selfAttentionParam.qScale = 1.0 / sqrt(param.dk);
    CREATE_OPERATION(selfAttentionParam, &selfAttentionKvCacheNode.operation);
    selfAttentionKvCacheNode.inTensorIds = {INTERMIDATE_Q_POSITIONEMBED,
                                            INTERMIDATE_K_POSITIONEMBED,
                                            INTERMIDATE_V_MIXEDLINEAROUT,
                                            IN_PASTKEY,
                                            IN_PASTVALUE,
                                            IN_ATTENTIONMASK,
                                            IN_TOKENOFFSET,
                                            IN_SEQLEN,
                                            IN_LAYERID};
    selfAttentionKvCacheNode.outTensorIds = {INTERMIDATE_SELFOUT};
    selfAttentionKvCacheNode.inTensorReshapeFuncs.resize(selfAttentionKvCacheNode.inTensorIds.size());
    selfAttentionKvCacheNode.inTensorReshapeFuncs.at(SELF_ATTENTION_V_INPUT_INDEX) = [=](const atb::Dims &oldShape,
                                                                                         atb::Dims &newShape) {
        newShape.dimNum = SELF_ATTENTION_V_INPUT_SIZE;
        size_t newShapeDimIndex = 0;
        size_t oldShapeDimIndex = 0;
        newShape.dims[newShapeDimIndex++] = oldShape.dims[oldShapeDimIndex++];
        newShape.dims[newShapeDimIndex++] = oldShape.dims[oldShapeDimIndex++];
        newShape.dims[newShapeDimIndex++] = param.headNum;
        newShape.dims[newShapeDimIndex++] = oldShape.dims[oldShapeDimIndex++] / param.headNum;
    };

    atb_speed::common::ParallelParam selfOutLinearParam;
    selfOutLinearParam.rank = param.rank;
    selfOutLinearParam.rankSize = param.rankSize;
    selfOutLinearParam.isBias = true;
    atb_speed::common::RowParallelLinear(selfOutLinearParam, &selfOutLinearNode.operation);
    selfOutLinearNode.inTensorIds = {INTERMIDATE_SELFOUT, IN_SELFOUTLINEARWEIGHT, IN_SELFOUT_LINEARBIAS};
    selfOutLinearNode.outTensorIds = {INTERMIDATE_SELFLINEAROUT};

    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CREATE_OPERATION(addParam, &selfResidualAddNode.operation);
    selfResidualAddNode.inTensorIds = {IN_HIDDENSTATES, INTERMIDATE_SELFLINEAROUT};
    selfResidualAddNode.outTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT};

    CREATE_OPERATION(rmsNormParam, &selfNormNode.operation);
    selfNormNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT, IN_SELFOUTNORMWEIGHT};
    selfNormNode.outTensorIds = {INTERMIDATE_SELFNORMOUT};

    atb_speed::common::MlpGateParam mlpParam;
    mlpParam.rank = param.rank;
    mlpParam.rankSize = param.rankSize;
    mlpParam.activationType = atb::infer::ActivationType::ACTIVATION_SWISH;
    mlpParam.transposeB = false;
    mlpParam.isBias = false;
    mlpParam.isPack = false;
    atb_speed::common::MlpGateLayer(mlpParam, &mlpNode.operation);
    mlpNode.inTensorIds = {INTERMIDATE_SELFNORMOUT, IN_MLPUPWEIGHT, IN_MLPGATEWEIGHT, IN_MLPDOWNWEIGHT};
    mlpNode.outTensorIds = {INTERMIDATE_MLPOUT};

    CREATE_OPERATION(addParam, &mlpResidualAddNode.operation);
    mlpResidualAddNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT, INTERMIDATE_MLPOUT};
    mlpResidualAddNode.outTensorIds = {OUT_LAYEROUT};

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
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
}

atb::Operation *CreateFlashAttentionRopeLayer(const nlohmann::json &paramJson)
{
    ATB_LOG(INFO) << GetFuncNameAndNameSpace(__PRETTY_FUNCTION__);
    atb::Operation *op;
    atb_speed::internlm_7b::FlashAttentionRopeLayer(paramJson.get<FlashAttentionRopeLayerParam>(), &op);
    return op;
}

} // namespace internlm_7b
} // namespace atb_speed