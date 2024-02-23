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
    IN_RELATIVE_POSITION_BIAS,


    OUT_LAYEROUT,

    INTERMIDATE_INPUTNORMOUT,
    INTERMIDATE_QKVBIAS_OUT,
    INTERMIDATE_QKVMIXEDLINEAROUT,

    INTERMIDATE_SELFOUT,
    

    INTERMIDATE_MASKEDBIAS,
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

static const uint64_t IN_TENSOR_COUNT = 31;//30;
static const uint64_t OUT_TENSOR_COUNT = 1;//1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 23;
static const uint64_t NODE_COUNT = 22;//21;

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
    atb::Node &maskFillBiasNode = opGraph.nodes.at(nodeId++); //ok
    atb::Node &inputNormNode = opGraph.nodes.at(nodeId++); //ok
    atb::Node &catQKNode = opGraph.nodes.at(nodeId++); //new
    atb::Node &catKVNode = opGraph.nodes.at(nodeId++); //new
    atb::Node &qkvLinearNode = opGraph.nodes.at(nodeId++); //ok
    atb::Node &transposeNode = opGraph.nodes.at(nodeId++); //new
    atb::Node &splitQKVNode = opGraph.nodes.at(nodeId++); //ok

    atb::Node &selfAttentionNode = opGraph.nodes.at(nodeId++); //ok

    // atb::Node &outSelfNode= opGraph.nodes.at(nodeId++); //new

    atb::Node &selfOutLinearNode = opGraph.nodes.at(nodeId++); //ok
    atb::Node &gama1MultNode = opGraph.nodes.at(nodeId++); //new
    atb::Node &selfResidualAddNode = opGraph.nodes.at(nodeId++);//ok
    atb::Node &sliceTextNode = opGraph.nodes.at(nodeId++);//new
    atb::Node &normalTextNode = opGraph.nodes.at(nodeId++);//new
    atb::Node &mlpTextNode = opGraph.nodes.at(nodeId++);//new
    atb::Node &gama2MultTextNode = opGraph.nodes.at(nodeId++); //new
    atb::Node &selfResidualTextAddNode = opGraph.nodes.at(nodeId++);//new
    atb::Node &sliceImageNode = opGraph.nodes.at(nodeId++);//new
    atb::Node &normalImageNode = opGraph.nodes.at(nodeId++);//new
    atb::Node &mlpImageNode = opGraph.nodes.at(nodeId++);//new
    atb::Node &gama2MultImageNode = opGraph.nodes.at(nodeId++); //new
    atb::Node &selfResidualImageAddNode = opGraph.nodes.at(nodeId++);//new
    atb::Node &catNode = opGraph.nodes.at(nodeId++);//new

    atb::infer::FillParam maskfillParam;
    maskfillParam.value = { -10000};
    maskfillParam.withMask = true;
    CREATE_OPERATION(maskfillParam, &maskFillBiasNode.operation);
    maskFillBiasNode.inTensorIds = {IN_RELATIVE_POSITION_BIAS, IN_ATTENTIONMASK};
    maskFillBiasNode.outTensorIds = {INTERMIDATE_MASKEDBIAS};
    maskFillBiasNode.inTensorReshapeFuncs.resize(maskFillBiasNode.inTensorIds.size());
    maskFillBiasNode.inTensorReshapeFuncs.at(0) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        if(oldShape.dimNum==3){
            newShape.dimNum = 4;
            newShape.dims[0] = 1;
            newShape.dims[1] = oldShape.dims[0];
            newShape.dims[2] = oldShape.dims[1];
            newShape.dims[3] = oldShape.dims[2];
        }else{
            newShape = oldShape;
        }
        
    };
    maskFillBiasNode.inTensorReshapeFuncs.at(1) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        if(oldShape.dimNum==2){
            newShape.dimNum = 4;
            newShape.dims[0] = 1;
            newShape.dims[1] = 1;
            newShape.dims[2] = oldShape.dims[0];
            newShape.dims[3] = oldShape.dims[1];
        }else{
            newShape = oldShape;
        }
        
    };

    // self.norm1(x)  x 1 941 768
    atb::infer::LayerNormParam layerNormParam;
    layerNormParam.layerType = atb::infer::LayerNormParam::LayerNormType::LAYER_NORM_NORM;
    layerNormParam.normParam.epsilon = param.layerNormEps;
    layerNormParam.normParam.beginNormAxis = 2;
    layerNormParam.normParam.beginParamsAxis = 0;
    CREATE_OPERATION(layerNormParam, &inputNormNode.operation);
    // (bsz,seq_len,hidden_size) - > (bsz,seq_len,hidden_size)
    inputNormNode.inTensorIds = {IN_HIDDENSTATES, IN_NORMWEIGHT, IN_NORMBIASID};
    inputNormNode.outTensorIds = {INTERMIDATE_INPUTNORMOUT};
    // inputNormNode.inTensorReshapeFuncs.resize(inputNormNode.inTensorIds.size());
    // inputNormNode.inTensorReshapeFuncs.at(1) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {// TODO
    //     newShape.dimNum = 3;
    //     newShape.dims[0] = 1;
    //     newShape.dims[1] = 1;
    //     newShape.dims[2] = oldShape.dims[0];
    // };
    // inputNormNode.inTensorReshapeFuncs.at(2) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {// TODO
    //     newShape.dimNum = 3;
    //     newShape.dims[0] = 1;
    //     newShape.dims[1] = 1;
    //     newShape.dims[2] = oldShape.dims[0];
    // };


    // cat q k bias 
    atb::infer::ConcatParam catQKVParam;
    catQKVParam.concatDim = 0;
    CreateOperation(catQKVParam, &catQKNode.operation);
    catQKNode.inTensorIds = {IN_QBAISID, IN_KBAISID};
    catQKNode.outTensorIds = {INTERMIDATE_QKBIAS_OUT};

    // cat qkv bias [dim*3]
    CreateOperation(catQKVParam, &catKVNode.operation);
    catKVNode.inTensorIds = {INTERMIDATE_QKBIAS_OUT, IN_VBIASID};
    catKVNode.outTensorIds = {INTERMIDATE_QKVBIAS_OUT};


    
    // (bsz,seq_len,hidden_size) - > (bsz,seq_len,hidden_size)
    atb::infer::LinearParam linearParam = {false, true, true};
    CREATE_OPERATION(linearParam, &qkvLinearNode.operation);
    qkvLinearNode.inTensorIds = {INTERMIDATE_INPUTNORMOUT, IN_QKVMIXEDLINEARWEIGHT,INTERMIDATE_QKVBIAS_OUT};
    qkvLinearNode.outTensorIds = {INTERMIDATE_QKVMIXEDLINEAROUT};
    
    // reshape 
    atb::infer::TransposeParam transParam;
    //transParam.perm = { 2, 0, 3, 1, 4 };
    transParam.perm = { 2, 0, 1, 3, 4 };
    CREATE_OPERATION(transParam, &transposeNode.operation);
    transposeNode.inTensorIds = {INTERMIDATE_QKVMIXEDLINEAROUT};
    transposeNode.outTensorIds = {INTERMIDATE_QKVTRANSROUT};
    transposeNode.inTensorReshapeFuncs.resize(transposeNode.inTensorIds.size());
    transposeNode.inTensorReshapeFuncs.at(0) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {// TODO
        newShape.dimNum = 5;
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = oldShape.dims[1];
        newShape.dims[2] = 3;
        newShape.dims[3] = param.headNum;
        newShape.dims[4] = oldShape.dims[2] / 3 / param.headNum;
        // 0  1   2  3  4 

        // 3 1 941 12 64
        //
        //原始  1 941 768
    };


    // 1 941 2304 -> 
    // atb::infer::SplitParam splitParam = {0, 3};
    atb::infer::SplitParam splitParam = {0, 3}; // 1 941 2304 -> 3 * 1 941 [12 64]
    CREATE_OPERATION(splitParam, &splitQKVNode.operation);
    splitQKVNode.inTensorIds = {INTERMIDATE_QKVTRANSROUT};
    splitQKVNode.outTensorIds = {INTERMIDATE_MIXEDQ, INTERMIDATE_MIXEDK, INTERMIDATE_MIXEDV};


    atb::infer::SelfAttentionParam selfAttentionParam;
    selfAttentionParam.headDim = param.dk;
    selfAttentionParam.headNum = param.headNum;
    selfAttentionParam.qScale = 1.0f / std::sqrt(param.dk);
    selfAttentionParam.qkScale = 1.0f;
    selfAttentionParam.isSupportAlibi = true;
    CREATE_OPERATION(selfAttentionParam, &selfAttentionNode.operation);
    selfAttentionNode.inTensorIds = {INTERMIDATE_MIXEDQ,
                                            INTERMIDATE_MIXEDK,
                                            INTERMIDATE_MIXEDV,
                                            IN_PASTKEY,
                                            IN_PASTVALUE,
                                            INTERMIDATE_MASKEDBIAS,
                                            IN_TOKENOFFSET,
                                            IN_SEQLEN,
                                            IN_LAYERID};
    selfAttentionNode.outTensorIds = {INTERMIDATE_SELFOUT};
    selfAttentionNode.inTensorReshapeFuncs.resize(selfAttentionNode.inTensorIds.size());
    selfAttentionNode.inTensorReshapeFuncs.at(0) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {// TODO
        newShape.dimNum = 2;
        newShape.dims[0] = oldShape.dims[1]*oldShape.dims[2];
        newShape.dims[1] = oldShape.dims[3]*oldShape.dims[4];
    };
    selfAttentionNode.inTensorReshapeFuncs.at(1) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {// TODO
        newShape.dimNum = 2;
        newShape.dims[0] = oldShape.dims[1]*oldShape.dims[2];
        newShape.dims[1] = oldShape.dims[3]*oldShape.dims[4];
    };
    selfAttentionNode.inTensorReshapeFuncs.at(2) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {// TODO
        newShape.dimNum = 2;
        newShape.dims[0] = oldShape.dims[1]*oldShape.dims[2];
        newShape.dims[1] = oldShape.dims[3]*oldShape.dims[4];
    };
    // selfAttentionNode.inTensorReshapeFuncs.at(5) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {// TODO
    //     newShape.dimNum = 4;
    //     newShape.dims[0] = 1;
    //     newShape.dims[1] = oldShape.dims[0];
    //     newShape.dims[2] = oldShape.dims[1];
    //     newShape.dims[3] = oldShape.dims[2];
    // };

    // atb::infer::TransposeParam outSelfTransParam;
    // outSelfTransParam.perm = { 0,1,2 };
    // CREATE_OPERATION(outSelfTransParam, &outSelfNode.operation);
    // outSelfNode.inTensorIds = {INTERMIDATE_SELFOUT};
    // outSelfNode.outTensorIds = {OUT_ATTEN_RES};
    


   
    atb::infer::LinearParam layerParam={false, true, true};
    CREATE_OPERATION(layerParam, &selfOutLinearNode.operation);
    selfOutLinearNode.inTensorIds = {
        INTERMIDATE_SELFOUT, IN_SELFOUTLINEARWEIGHT,IN_SELFOUTLINEBIASID};
    selfOutLinearNode.outTensorIds = {INTERMIDATE_SELFLINEAROUT};
    // Attention over


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

    ////  

    //  1 941 768 
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
    // (bsz,seq_len,hidden_size) - > (bsz,seq_len,hidden_size)
    normalTextNode.inTensorIds = {INTERMIDATE_SLICE_TEXT_OUT, IN_NORM2TEXTWEIGHT,IN_NORM2TEXTBIAS};
    normalTextNode.outTensorIds = {INTERMIDATE_NORM2TEXT_OUT};// 1 40 768


    atb_speed::common::MlpGateParamV2 mlpTextParam;
    mlpTextParam.commDownParam.rank = param.rank;
    mlpTextParam.commDownParam.rankSize = param.rankSize;
    mlpTextParam.activationType = atb::infer::ActivationType::ACTIVATION_GELU;
    mlpTextParam.transposeB = true;
    mlpTextParam.isBias = true;
    mlpTextParam.noGate = true;
    mlpTextParam.isPack = false;
    atb_speed::common::MlpGateLayerV2(mlpTextParam, &mlpTextNode.operation);
    mlpTextNode.inTensorIds = {INTERMIDATE_NORM2TEXT_OUT, IN_MLPTEXTUPWEIGHT, IN_HOLDER, IN_MLPTEXTDOWNWEIGHT,
                        IN_HOLDER, IN_HOLDER, IN_HOLDER, IN_MPLTEXTBIASUP, IN_HOLDER,
                        IN_MPLTEXTBIASDOWN, IN_HOLDER, IN_HOLDER, IN_HOLDER, IN_HOLDER,
                        IN_HOLDER, IN_HOLDER, IN_HOLDER, IN_HOLDER, IN_HOLDER};
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
    sliceImageParam.offsets = {0, param.maxTextLen,0};
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
    // (bsz,seq_len,hidden_size) - > (bsz,seq_len,hidden_size)
    normalImageNode.inTensorIds = {INTERMIDATE_SLICE_IMAGE_OUT, IN_NORM2IMAGEWEIGHT,IN_NORM2IMAGEBIAS};
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
    mlpImageNode.inTensorIds = {INTERMIDATE_NORM2IMAGE_OUT, IN_MLPIMAGEUPWIGHT, IN_HOLDER, IN_MLPIMAGEDOWNWIGHT,
                        IN_HOLDER, IN_HOLDER, IN_HOLDER, IN_MPLIMAGEBIASUP, IN_HOLDER,
                        IN_MPLIMAGEBIASDOWN, IN_HOLDER, IN_HOLDER, IN_HOLDER, IN_HOLDER,
                        IN_HOLDER, IN_HOLDER, IN_HOLDER, IN_HOLDER, IN_HOLDER};
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
} // namespace vlmo
} // namespace atb_speed
