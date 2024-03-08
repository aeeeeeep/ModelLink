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
#include "encoder_vl_layer.h"

#include "layers/mlp_gate_v2.h"
#include "layers/parallel_layer_v2.h"

namespace atb_speed {
namespace vlmo {
enum EncoderLayerTensorId : int {
    IN_HIDDENSTATES = 0,
    IN_ATTENTIONMASK,
    IN_PASTKEY,
    IN_PASTVALUE,
    IN_TOKENOFFSET,
    IN_SEQLEN,
    IN_HOLDER,
    IN_LAYERID,
    IN_GAMMA1,
    IN_GAMMA2,
    IN_NORMWEIGHT,
    IN_NORMBIASID,
    IN_QBAISID,
    IN_KBAISID,
    IN_VBIASID,
    IN_QKVMIXEDLINEARWEIGHT,
    IN_SELFOUTLINEARWEIGHT,
    IN_SELFOUTLINEBIASID,
    IN_NORM2VLWEIGHT,
    IN_NORM2VLBIAS,
    IN_MLPVLUPWEIGHT,
    IN_MLPVLDOWNWEIGHT,
    IN_MPLVLBIASUP,
    IN_MPLVLBIASDOWN,
    OUT_LAYEROUT,
    INTERMIDATE_INPUTNORMOUT,
    INTERMIDATE_QKVBIAS_OUT,
    INTERMIDATE_QKVMIXEDLINEAROUT,
    INTERMIDATE_SELFOUT,
    INTERMIDATE_QKBIAS_OUT,
    INTERMIDATE_QKVTRANSROUT,
    INTERMIDATE_MIXEDQ,
    INTERMIDATE_MIXEDK,
    INTERMIDATE_MIXEDV,
    INTERMIDATE_SELFLINEAROUT,
    INTERMIDATE_GAMMA1_OUT,
    INTERMIDATE_SELFRESIDUALADDOUT,
    INTERMIDATE_NORM2VL_OUT,
    INTERMIDATE_MLPVL_OUT,
    INTERMIDATE_GAMMA2_VL_OUT,
};

static const uint64_t IN_TENSOR_COUNT = 24;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 15;
static const uint64_t NODE_COUNT = 14;

atb::Status EncoderVlLayer(const EncoderVllayerParam &param, atb::Operation **operation)
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
    atb::Node &catQKNode = opGraph.nodes.at(nodeId++);
    atb::Node &catKVNode = opGraph.nodes.at(nodeId++);
    atb::Node &qkvLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &transposeNode = opGraph.nodes.at(nodeId++);
    atb::Node &splitQKVNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfAttentionNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfOutLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &gama1MultNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfResidualAddNode = opGraph.nodes.at(nodeId++);
    atb::Node &normalVlNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpVlNode = opGraph.nodes.at(nodeId++);
    atb::Node &gama2MultVlNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfResidualVlAddNode = opGraph.nodes.at(nodeId++);

    atb::infer::LayerNormParam layerNormParam;
    layerNormParam.layerType = atb::infer::LayerNormParam::LayerNormType::LAYER_NORM_NORM;
    layerNormParam.normParam.epsilon = param.layerNormEps;
    layerNormParam.normParam.beginNormAxis = 2;
    layerNormParam.normParam.beginParamsAxis = 0;
    CREATE_OPERATION(layerNormParam, &inputNormNode.operation);
    inputNormNode.inTensorIds = {IN_HIDDENSTATES, IN_NORMWEIGHT, IN_NORMBIASID};
    inputNormNode.outTensorIds = {INTERMIDATE_INPUTNORMOUT};

    atb::infer::ConcatParam catQKVParam;
    catQKVParam.concatDim = 0;
    CreateOperation(catQKVParam, &catQKNode.operation);
    catQKNode.inTensorIds = {IN_QBAISID, IN_KBAISID};
    catQKNode.outTensorIds = {INTERMIDATE_QKBIAS_OUT};

    CreateOperation(catQKVParam, &catKVNode.operation);
    catKVNode.inTensorIds = {INTERMIDATE_QKBIAS_OUT, IN_VBIASID};
    catKVNode.outTensorIds = {INTERMIDATE_QKVBIAS_OUT};

    atb::infer::LinearParam linearParam = {false, true, true};
    CREATE_OPERATION(linearParam, &qkvLinearNode.operation);
    qkvLinearNode.inTensorIds = {INTERMIDATE_INPUTNORMOUT, IN_QKVMIXEDLINEARWEIGHT, INTERMIDATE_QKVBIAS_OUT};
    qkvLinearNode.outTensorIds = {INTERMIDATE_QKVMIXEDLINEAROUT};

    atb::infer::TransposeParam transParam;
    transParam.perm = {2, 0, 1, 3, 4};
    CREATE_OPERATION(transParam, &transposeNode.operation);
    transposeNode.inTensorIds = {INTERMIDATE_QKVMIXEDLINEAROUT};
    transposeNode.outTensorIds = {INTERMIDATE_QKVTRANSROUT};
    transposeNode.inTensorReshapeFuncs.resize(transposeNode.inTensorIds.size());
    transposeNode.inTensorReshapeFuncs.at(0) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 5;
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = oldShape.dims[1];
        newShape.dims[2] = 3;
        newShape.dims[3] = param.headNum;
        newShape.dims[4] = oldShape.dims[2] / 3 / param.headNum;
    };

    atb::infer::SplitParam splitParam = {0, 3};
    CREATE_OPERATION(splitParam, &splitQKVNode.operation);
    splitQKVNode.inTensorIds = {INTERMIDATE_QKVTRANSROUT};
    splitQKVNode.outTensorIds = {INTERMIDATE_MIXEDQ, INTERMIDATE_MIXEDK, INTERMIDATE_MIXEDV};

    atb::infer::SelfAttentionParam selfAttentionParam;
    selfAttentionParam.headNum = param.headNum;
    selfAttentionParam.qScale = 1.0f / std::sqrt(param.dk);
    selfAttentionParam.qkScale = 1.0f;
    selfAttentionParam.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_ALIBI;
    CREATE_OPERATION(selfAttentionParam, &selfAttentionNode.operation);
    selfAttentionNode.inTensorIds = {INTERMIDATE_MIXEDQ, INTERMIDATE_MIXEDK, INTERMIDATE_MIXEDV,
                                     IN_PASTKEY,         IN_PASTVALUE,       IN_ATTENTIONMASK,
                                     IN_TOKENOFFSET,     IN_SEQLEN,          IN_LAYERID};
    selfAttentionNode.outTensorIds = {INTERMIDATE_SELFOUT};
    selfAttentionNode.inTensorReshapeFuncs.resize(selfAttentionNode.inTensorIds.size());
    selfAttentionNode.inTensorReshapeFuncs.at(0) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2;
        newShape.dims[0] = oldShape.dims[1] * oldShape.dims[2];
        newShape.dims[1] = oldShape.dims[3] * oldShape.dims[4];
    };
    selfAttentionNode.inTensorReshapeFuncs.at(1) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2;
        newShape.dims[0] = oldShape.dims[1] * oldShape.dims[2];
        newShape.dims[1] = oldShape.dims[3] * oldShape.dims[4];
    };
    selfAttentionNode.inTensorReshapeFuncs.at(2) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2;
        newShape.dims[0] = oldShape.dims[1] * oldShape.dims[2];
        newShape.dims[1] = oldShape.dims[3] * oldShape.dims[4];
    };

    atb::infer::LinearParam layerParam = {false, false, true};
    CREATE_OPERATION(layerParam, &selfOutLinearNode.operation);
    selfOutLinearNode.inTensorIds = {INTERMIDATE_SELFOUT, IN_SELFOUTLINEARWEIGHT, IN_SELFOUTLINEBIASID};
    selfOutLinearNode.outTensorIds = {INTERMIDATE_SELFLINEAROUT};

    atb::infer::ElewiseParam gamma1MutmalParam;
    gamma1MutmalParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    CREATE_OPERATION(gamma1MutmalParam, &gama1MultNode.operation);
    gama1MultNode.inTensorIds = {IN_GAMMA1, INTERMIDATE_SELFLINEAROUT};
    gama1MultNode.outTensorIds = {INTERMIDATE_GAMMA1_OUT};

    atb::infer::ElewiseParam addGamma1Param;
    addGamma1Param.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CREATE_OPERATION(addGamma1Param, &selfResidualAddNode.operation);
    selfResidualAddNode.inTensorIds = {IN_HIDDENSTATES, INTERMIDATE_GAMMA1_OUT};
    selfResidualAddNode.outTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT};

    atb::infer::LayerNormParam layerTextNormParam;

    layerTextNormParam.layerType = atb::infer::LayerNormParam::LayerNormType::LAYER_NORM_NORM;
    layerTextNormParam.normParam.beginNormAxis = 2;
    layerTextNormParam.normParam.beginParamsAxis = 0;
    layerTextNormParam.normParam.epsilon = param.layerNormEps;
    CREATE_OPERATION(layerTextNormParam, &normalVlNode.operation);
    normalVlNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT, IN_NORM2VLWEIGHT, IN_NORM2VLBIAS};
    normalVlNode.outTensorIds = {INTERMIDATE_NORM2VL_OUT};

    atb_speed::common::MlpGateParamV2 mlpTextParam;
    mlpTextParam.commDownParam.rank = param.rank;
    mlpTextParam.commDownParam.rankSize = param.rankSize;
    mlpTextParam.activationType = atb::infer::ActivationType::ACTIVATION_GELU;
    mlpTextParam.transposeB = true;
    mlpTextParam.isBias = true;
    mlpTextParam.noGate = true;
    mlpTextParam.isPack = false;
    atb_speed::common::MlpGateLayerV2(mlpTextParam, &mlpVlNode.operation);
    mlpVlNode.inTensorIds = {INTERMIDATE_NORM2VL_OUT,
                             IN_MLPVLUPWEIGHT,
                             IN_HOLDER,
                             IN_MLPVLDOWNWEIGHT,
                             IN_HOLDER,
                             IN_HOLDER,
                             IN_HOLDER,
                             IN_MPLVLBIASUP,
                             IN_HOLDER,
                             IN_MPLVLBIASDOWN,
                             IN_HOLDER,
                             IN_HOLDER,
                             IN_HOLDER,
                             IN_HOLDER,
                             IN_HOLDER,
                             IN_HOLDER,
                             IN_HOLDER,
                             IN_HOLDER,
                             IN_HOLDER};
    mlpVlNode.outTensorIds = {INTERMIDATE_MLPVL_OUT};

    atb::infer::ElewiseParam gamma2TextMutmalParam;
    gamma2TextMutmalParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    CREATE_OPERATION(gamma2TextMutmalParam, &gama2MultVlNode.operation);
    gama2MultVlNode.inTensorIds = {IN_GAMMA2, INTERMIDATE_MLPVL_OUT};
    gama2MultVlNode.outTensorIds = {INTERMIDATE_GAMMA2_VL_OUT};

    atb::infer::ElewiseParam addGamma2TextParam;
    addGamma2TextParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CREATE_OPERATION(addGamma2TextParam, &selfResidualVlAddNode.operation);
    selfResidualVlAddNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT, INTERMIDATE_GAMMA2_VL_OUT};
    selfResidualVlAddNode.outTensorIds = {OUT_LAYEROUT};

    opGraph.inferShapeFunc = [](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
    };
    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}

EncoderVlLayerBinder::EncoderVlLayerBinder() = default;

EncoderVlLayerBinder::~EncoderVlLayerBinder() = default;

void EncoderVlLayerBinder::ParseParam(const nlohmann::json &paramJson)
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

void EncoderVlLayerBinder::BindTensor(atb::VariantPack &variantPack)
{
    variantPack.inTensors.at(IN_TOKENOFFSET).hostData = tokenOffset_.data();
    variantPack.inTensors.at(IN_SEQLEN).hostData = seqLen_.data();
}

void from_json(const nlohmann::json &paramJson, EncoderVllayerParam &param)
{
    paramJson.at("layerNormEps").get_to(param.layerNormEps);
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
    if (paramJson.contains("maxTextLen")) {
        paramJson.at("maxTextLen").get_to(param.maxTextLen);
    }
}

atb::Operation *CreateEncoderVlLayer(const nlohmann::json &paramJson)
{
    ATB_LOG(INFO) << GetFuncNameAndNameSpace(__PRETTY_FUNCTION__);
    atb::Operation *op;
    atb_speed::vlmo::EncoderVlLayer(paramJson.get<EncoderVllayerParam>(), &op);
    return op;
}

} // namespace vlmo
} // namespace atb_speed
