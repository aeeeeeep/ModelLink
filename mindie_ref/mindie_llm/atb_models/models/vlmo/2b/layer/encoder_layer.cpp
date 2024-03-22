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
#include "encoder_layer.h"

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
    IN_NORM2TEXTWEIGHT,
    IN_NORM2TEXTBIAS,
    IN_NORM2IMAGEWEIGHT,
    IN_NORM2IMAGEBIAS,
    IN_MLPTEXTUPWEIGHT,
    IN_MLPTEXTDOWNWEIGHT,
    IN_MPLTEXTBIASUP,
    IN_MPLTEXTBIASDOWN,
    IN_MLPIMAGEUPWIGHT,
    IN_MLPIMAGEDOWNWIGHT,
    IN_MPLIMAGEBIASUP,
    IN_MPLIMAGEBIASDOWN,
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
    INTERMIDATE_SLICE_TEXT_OUT,
    INTERMIDATE_NORM2TEXT_OUT,
    INTERMIDATE_MLPTEXT_OUT,
    INTERMIDATE_GAMMA2_TEXT_OUT,
    INTERMIDATE_SELFRESIDUALADDTEXTOUT,
    INTERMIDATE_SLICE_IMAGE_OUT,
    INTERMIDATE_NORM2IMAGE_OUT,
    INTERMIDATE_MLPIMAGE_OUT,
    INTERMIDATE_GAMMA2_IMAGE_OUT,
    INTERMIDATE_SELFRESIDUALADDIMAGEOUT,
};

static const uint64_t IN_TENSOR_COUNT = 30;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 22;
static const uint64_t NODE_COUNT = 21;

atb::Status EncoderLayer(const EncoderLayerParam &param, atb::Operation **operation)
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
    atb::Node &sliceTextNode = opGraph.nodes.at(nodeId++);
    atb::Node &normalTextNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpTextNode = opGraph.nodes.at(nodeId++);
    atb::Node &gama2MultTextNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfResidualTextAddNode = opGraph.nodes.at(nodeId++);
    atb::Node &sliceImageNode = opGraph.nodes.at(nodeId++);
    atb::Node &normalImageNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpImageNode = opGraph.nodes.at(nodeId++);
    atb::Node &gama2MultImageNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfResidualImageAddNode = opGraph.nodes.at(nodeId++);
    atb::Node &catNode = opGraph.nodes.at(nodeId++);

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

    atb_speed::common::ParallelParamV2 selfOutLinearParam;
    selfOutLinearParam.commParam.rank = param.rank;
    selfOutLinearParam.commParam.rankSize = param.rankSize;
    selfOutLinearParam.commParam.backend = param.backend;
    selfOutLinearParam.isBias = true;
    atb_speed::common::RowParallelLinearV2(selfOutLinearParam, &selfOutLinearNode.operation);
    selfOutLinearNode.inTensorIds = {
        INTERMIDATE_SELFOUT, IN_SELFOUTLINEARWEIGHT, IN_SELFOUTLINEBIASID, IN_HOLDER, IN_HOLDER, IN_HOLDER, IN_HOLDER
    };
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

    atb::infer::SliceParam sliceTextParam;
    sliceTextParam.offsets = {0, 0, 0};
    sliceTextParam.size = {-1, param.maxTextLen, -1};
    CREATE_OPERATION(sliceTextParam, &sliceTextNode.operation);
    sliceTextNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT};
    sliceTextNode.outTensorIds = {INTERMIDATE_SLICE_TEXT_OUT};

    atb::infer::LayerNormParam layerTextNormParam;

    layerTextNormParam.layerType = atb::infer::LayerNormParam::LayerNormType::LAYER_NORM_NORM;
    layerTextNormParam.normParam.beginNormAxis = 2;
    layerTextNormParam.normParam.beginParamsAxis = 0;
    layerTextNormParam.normParam.epsilon = param.layerNormEps;
    CREATE_OPERATION(layerTextNormParam, &normalTextNode.operation);
    normalTextNode.inTensorIds = {INTERMIDATE_SLICE_TEXT_OUT, IN_NORM2TEXTWEIGHT, IN_NORM2TEXTBIAS};
    normalTextNode.outTensorIds = {INTERMIDATE_NORM2TEXT_OUT};

    atb_speed::common::MlpGateParamV2 mlpTextParam;
    mlpTextParam.commDownParam.rank = param.rank;
    mlpTextParam.commDownParam.rankSize = param.rankSize;
    mlpTextParam.activationType = atb::infer::ActivationType::ACTIVATION_GELU;
    mlpTextParam.transposeB = true;
    mlpTextParam.isBias = true;
    mlpTextParam.noGate = true;
    mlpTextParam.isPack = false;
    atb_speed::common::MlpGateLayerV2(mlpTextParam, &mlpTextNode.operation);
    mlpTextNode.inTensorIds = {INTERMIDATE_NORM2TEXT_OUT,
                               IN_MLPTEXTUPWEIGHT,
                               IN_HOLDER,
                               IN_MLPTEXTDOWNWEIGHT,
                               IN_HOLDER,
                               IN_HOLDER,
                               IN_HOLDER,
                               IN_MPLTEXTBIASUP,
                               IN_HOLDER,
                               IN_MPLTEXTBIASDOWN,
                               IN_HOLDER,
                               IN_HOLDER,
                               IN_HOLDER,
                               IN_HOLDER,
                               IN_HOLDER,
                               IN_HOLDER,
                               IN_HOLDER,
                               IN_HOLDER,
                               IN_HOLDER};
    mlpTextNode.outTensorIds = {INTERMIDATE_MLPTEXT_OUT};

    atb::infer::ElewiseParam gamma2TextMutmalParam;
    gamma2TextMutmalParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    CREATE_OPERATION(gamma2TextMutmalParam, &gama2MultTextNode.operation);
    gama2MultTextNode.inTensorIds = {IN_GAMMA2, INTERMIDATE_MLPTEXT_OUT};
    gama2MultTextNode.outTensorIds = {INTERMIDATE_GAMMA2_TEXT_OUT};

    atb::infer::ElewiseParam addGamma2TextParam;
    addGamma2TextParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CREATE_OPERATION(addGamma2TextParam, &selfResidualTextAddNode.operation);
    selfResidualTextAddNode.inTensorIds = {INTERMIDATE_SLICE_TEXT_OUT, INTERMIDATE_GAMMA2_TEXT_OUT};
    selfResidualTextAddNode.outTensorIds = {INTERMIDATE_SELFRESIDUALADDTEXTOUT};

    atb::infer::SliceParam sliceImageParam;
    sliceImageParam.offsets = {0, param.maxTextLen, 0};
    sliceImageParam.size = {-1, -1, -1};
    CREATE_OPERATION(sliceImageParam, &sliceImageNode.operation);
    sliceImageNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT};
    sliceImageNode.outTensorIds = {INTERMIDATE_SLICE_IMAGE_OUT};

    atb::infer::LayerNormParam layerIMAGENormParam;
    layerIMAGENormParam.layerType = atb::infer::LayerNormParam::LayerNormType::LAYER_NORM_NORM;
    layerIMAGENormParam.normParam.beginNormAxis = 2;
    layerIMAGENormParam.normParam.beginParamsAxis = 0;
    layerIMAGENormParam.normParam.epsilon = param.layerNormEps;
    CREATE_OPERATION(layerIMAGENormParam, &normalImageNode.operation);
    normalImageNode.inTensorIds = {INTERMIDATE_SLICE_IMAGE_OUT, IN_NORM2IMAGEWEIGHT, IN_NORM2IMAGEBIAS};
    normalImageNode.outTensorIds = {INTERMIDATE_NORM2IMAGE_OUT};

    atb_speed::common::MlpGateParamV2 mlpIMAGEParam;
    mlpIMAGEParam.commDownParam.rank = param.rank;
    mlpIMAGEParam.commDownParam.rankSize = param.rankSize;
    mlpIMAGEParam.activationType = atb::infer::ActivationType::ACTIVATION_GELU;
    mlpIMAGEParam.transposeB = true;
    mlpIMAGEParam.isBias = true;
    mlpIMAGEParam.noGate = true;
    mlpIMAGEParam.isPack = false;
    atb_speed::common::MlpGateLayerV2(mlpIMAGEParam, &mlpImageNode.operation);
    mlpImageNode.inTensorIds = {INTERMIDATE_NORM2IMAGE_OUT,
                                IN_MLPIMAGEUPWIGHT,
                                IN_HOLDER,
                                IN_MLPIMAGEDOWNWIGHT,
                                IN_HOLDER,
                                IN_HOLDER,
                                IN_HOLDER,
                                IN_MPLIMAGEBIASUP,
                                IN_HOLDER,
                                IN_MPLIMAGEBIASDOWN,
                                IN_HOLDER,
                                IN_HOLDER,
                                IN_HOLDER,
                                IN_HOLDER,
                                IN_HOLDER,
                                IN_HOLDER,
                                IN_HOLDER,
                                IN_HOLDER,
                                IN_HOLDER};
    mlpImageNode.outTensorIds = {INTERMIDATE_MLPIMAGE_OUT};

    atb::infer::ElewiseParam gamma2IMAGEMutmalParam;
    gamma2IMAGEMutmalParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    CREATE_OPERATION(gamma2IMAGEMutmalParam, &gama2MultImageNode.operation);
    gama2MultImageNode.inTensorIds = {IN_GAMMA2, INTERMIDATE_MLPIMAGE_OUT};
    gama2MultImageNode.outTensorIds = {INTERMIDATE_GAMMA2_IMAGE_OUT};

    atb::infer::ElewiseParam addGamma2IMAGEParam;
    addGamma2IMAGEParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CREATE_OPERATION(addGamma2IMAGEParam, &selfResidualImageAddNode.operation);
    selfResidualImageAddNode.inTensorIds = {INTERMIDATE_SLICE_IMAGE_OUT, INTERMIDATE_GAMMA2_IMAGE_OUT};
    selfResidualImageAddNode.outTensorIds = {INTERMIDATE_SELFRESIDUALADDIMAGEOUT};

    atb::infer::ConcatParam catParam;
    catParam.concatDim = 1;
    CREATE_OPERATION(catParam, &catNode.operation);
    catNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDTEXTOUT, INTERMIDATE_SELFRESIDUALADDIMAGEOUT};
    catNode.outTensorIds = {OUT_LAYEROUT};
    opGraph.inferShapeFunc = [](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
    };
    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}

EncoderLayerBinder::EncoderLayerBinder() = default;

EncoderLayerBinder::~EncoderLayerBinder() = default;

void EncoderLayerBinder::ParseParam(const nlohmann::json &paramJson)
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

void EncoderLayerBinder::BindTensor(atb::VariantPack &variantPack)
{
    variantPack.inTensors.at(IN_TOKENOFFSET).hostData = tokenOffset_.data();
    variantPack.inTensors.at(IN_SEQLEN).hostData = seqLen_.data();
}

void from_json(const nlohmann::json &paramJson, EncoderLayerParam &param)
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

atb::Operation *CreateEncoderLayer(const nlohmann::json &paramJson)
{
    ATB_LOG(INFO) << GetFuncNameAndNameSpace(__PRETTY_FUNCTION__);
    atb::Operation *op;
    atb_speed::vlmo::EncoderLayer(paramJson.get<EncoderLayerParam>(), &op);
    return op;
}

} // namespace vlmo
} // namespace atb_speed
