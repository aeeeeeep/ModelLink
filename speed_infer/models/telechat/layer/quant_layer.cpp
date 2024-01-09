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
#include "models/telechat/operation/common_mlp_quant.h"
#include "models/telechat/operation/rope.h"

namespace atb_speed {
namespace telechat {
enum QuantFALayerTensorId {
    IN_HIDDENSTATES = 0,
    IN_QMIXEDWEIGHT,
    IN_QMIXEDDEQSCALE,
    IN_QMIXEDBIAS,
    IN_KVMIXEDWEIGHT,
    IN_KVMIXEDDEQSCALE,
    IN_KVMIXEDBIAS,
    IN_SELFOUTLINEARWEIGHT,
    IN_SELFOUTLINEARDEQSCALE,
    IN_SELFOUTLINEARBIAS,
    IN_MLPGATEWEIGHT,
    IN_MLPLINEARDEQSCALEGATE,
    IN_LINEARBIASGATE,
    IN_MLPUPWEIGHT,
    IN_MLPLINEARDEQSCALEUP,
    IN_LINEARBIASUP,
    IN_MLPDOWNWEIGHT,
    IN_MLPLINEARDEQSCALEDOWN,
    IN_LINEARBIASDOWN,
    IN_NORMWEIGHT,
    IN_NORMBIAS,
    IN_SELFOUTNORMWEIGHT,
    IN_SELFOUTNORMBIAS,
    IN_COSEMBED,
    IN_SINEMBED,
    IN_ATTENTIONMASK,
    IN_PASTKEY,
    IN_PASTVALUE,
    IN_TOKENOFFSET,
    IN_SEQLEN,
    IN_LAYERID,
    OUT_TELECHATLAYEROUT,
    INTERNAL_INPUTNORMOUT,
    INTERNAL_QMIXEDLINEAROUT,
    INTERNAL_SELFQUNTOUT,
    INTERNAL_KVMIXEDLINEAROUT,
    INTERNAL_KMIXEDLINEAROUT,
    INTERNAL_VMIXEDLINEAROUT,
    INTERNAL_POSITIONEMBEDQ,
    INTERNAL_POSITIONEMBEDK,
    INTERNAL_SELFOUT,
    INTERNAL_SELFLINEAROUT,
    INTERNAL_SELFRESIDUALADDOUT,
    INTERNAL_SELFNORMOUT,
    INTERNAL_MLPOUT,
};

static const uint64_t IN_TENSOR_COUNT = 31;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERNAL_TENSOR_COUNT = 13;
static const uint64_t NODE_COUNT = 12;

atb::Status QuantFALayer(const QuantFALayerParam &param, atb::Operation **operation)
{   ATB_LOG(INFO) << "enter quant layer";
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERNAL_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    atb::Node &inputNormNode = opGraph.nodes.at(nodeId++);
    atb::Node &mixedQLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &mixedKVLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &splitKVNode = opGraph.nodes.at(nodeId++);
    atb::Node &ropeNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfAttentionKvCacheFusedNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfOutQuantNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfOutLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfResidualAddNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfNormNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpQuantNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpResidualAddNode = opGraph.nodes.at(nodeId++);

	if (param.isFloatQueryLayer || param.isFloatKVLayer) {
        atb::infer::RmsNormParam rmsNormParam;
        rmsNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
        rmsNormParam.normParam.epsilon = param.rmsNormEps;
        CreateOperation(rmsNormParam, &inputNormNode.operation);
        inputNormNode.inTensorIds = {IN_HIDDENSTATES, IN_NORMWEIGHT};
        inputNormNode.outTensorIds = {INTERNAL_INPUTNORMOUT};
    } else {
        atb::infer::RmsNormParam rmsNormParam;
        rmsNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
        rmsNormParam.normParam.quantInputScale = param.inputScale_qkv;
        rmsNormParam.normParam.quantInputOffset = param.inputOffset_qkv;
        rmsNormParam.normParam.quantType = atb::infer::QUANT_INT8;

        rmsNormParam.normParam.epsilon = param.rmsNormEps;
        CreateOperation(rmsNormParam, &inputNormNode.operation);
        inputNormNode.inTensorIds = {IN_HIDDENSTATES, IN_NORMWEIGHT, IN_NORMBIAS};
        inputNormNode.outTensorIds = {INTERNAL_INPUTNORMOUT};
    }

    if (param.isFloatQueryLayer) {
        atb::infer::LinearParam linearQParam = {false, true, false};
        CreateOperation(linearQParam, &mixedQLinearNode.operation);
        mixedQLinearNode.inTensorIds = {INTERNAL_INPUTNORMOUT, IN_QMIXEDWEIGHT};
        mixedQLinearNode.outTensorIds = {INTERNAL_QMIXEDLINEAROUT};
    } else {
        atb::infer::LinearQuantParam linearQParam = {false, true, true};
        CreateOperation(linearQParam, &mixedQLinearNode.operation);
        mixedQLinearNode.inTensorIds = {INTERNAL_INPUTNORMOUT, IN_QMIXEDWEIGHT, IN_QMIXEDBIAS, IN_QMIXEDDEQSCALE};
        mixedQLinearNode.outTensorIds = {INTERNAL_QMIXEDLINEAROUT};
    }

    if (param.isFloatKVLayer) {
        atb::infer::LinearParam linearKVParam = {false, true, false};
        CreateOperation(linearKVParam, &mixedKVLinearNode.operation);
        mixedKVLinearNode.inTensorIds = {INTERNAL_INPUTNORMOUT, IN_KVMIXEDWEIGHT};
        mixedKVLinearNode.outTensorIds = {INTERNAL_KVMIXEDLINEAROUT};
    } else {
        atb::infer::LinearQuantParam linearKVParam = {false, true, true};
        CreateOperation(linearKVParam, &mixedKVLinearNode.operation);
        mixedKVLinearNode.inTensorIds = {INTERNAL_INPUTNORMOUT, IN_KVMIXEDWEIGHT, IN_KVMIXEDBIAS, IN_KVMIXEDDEQSCALE};
        mixedKVLinearNode.outTensorIds = {INTERNAL_KVMIXEDLINEAROUT};
    }

    atb::infer::SplitParam splitParam;
    splitParam.splitDim = 3;
    splitParam.splitNum = 2;
    CreateOperation(splitParam, &splitKVNode.operation);
    splitKVNode.inTensorIds = {INTERNAL_KVMIXEDLINEAROUT};
    splitKVNode.outTensorIds = {INTERNAL_KMIXEDLINEAROUT, INTERNAL_VMIXEDLINEAROUT};
    splitKVNode.inTensorReshapeFuncs.resize(splitKVNode.inTensorIds.size());
    splitKVNode.inTensorReshapeFuncs.at(0) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 4;
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = oldShape.dims[1];
        newShape.dims[2] = param.headNum;
        newShape.dims[3] = oldShape.dims[2] / param.headNum;
    };

    atb_speed::telechat::RopeParam ropeParam;
    ropeParam.rotaryCoeff = 2;
    ropeParam.headNum = param.headNum;
    atb_speed::telechat::Rope(ropeParam, &ropeNode.operation);
    ropeNode.inTensorIds = {INTERNAL_QMIXEDLINEAROUT, INTERNAL_KMIXEDLINEAROUT, IN_COSEMBED, IN_SINEMBED, IN_SEQLEN};
    ropeNode.outTensorIds = {INTERNAL_POSITIONEMBEDQ, INTERNAL_POSITIONEMBEDK};

    atb::infer::SelfAttentionParam selfAttentionKvCacheParam;
    selfAttentionKvCacheParam.headNum = param.headNum;
    selfAttentionKvCacheParam.headDim = param.dk;
    selfAttentionKvCacheParam.qScale = 1.0f;
    selfAttentionKvCacheParam.qkScale = 1.0f / std::sqrt(param.dk);
    CreateOperation(selfAttentionKvCacheParam, &selfAttentionKvCacheFusedNode.operation);
    selfAttentionKvCacheFusedNode.inTensorIds = {INTERNAL_POSITIONEMBEDQ,
                                                INTERNAL_POSITIONEMBEDK,
                                                INTERNAL_VMIXEDLINEAROUT,
                                                IN_PASTKEY,
                                                IN_PASTVALUE,
                                                IN_ATTENTIONMASK,
                                                IN_TOKENOFFSET,
                                                IN_SEQLEN,
                                                IN_LAYERID};
    selfAttentionKvCacheFusedNode.outTensorIds = {INTERNAL_SELFOUT};

    atb::infer::ElewiseParam selfOutQuantParam;
    selfOutQuantParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_QUANT;
    selfOutQuantParam.quantParam.inputScale = param.inputScale_dense;
    selfOutQuantParam.quantParam.inputOffset = param.inputOffset_dense;
    CreateOperation(selfOutQuantParam, &selfOutQuantNode.operation);
    selfOutQuantNode.inTensorIds = {INTERNAL_SELFOUT};
    selfOutQuantNode.outTensorIds = {INTERNAL_SELFQUNTOUT};

    atb::infer::LinearQuantParam linearBiasParam = {false, false, true};
    CreateOperation(linearBiasParam, &selfOutLinearNode.operation);
    selfOutLinearNode.inTensorIds = {INTERNAL_SELFQUNTOUT, IN_SELFOUTLINEARWEIGHT, IN_SELFOUTLINEARBIAS, IN_SELFOUTLINEARDEQSCALE};
    selfOutLinearNode.outTensorIds = {INTERNAL_SELFLINEAROUT};

    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CreateOperation(addParam, &selfResidualAddNode.operation);
    selfResidualAddNode.inTensorIds = {IN_HIDDENSTATES, INTERNAL_SELFLINEAROUT};
    selfResidualAddNode.outTensorIds = {INTERNAL_SELFRESIDUALADDOUT};

    atb::infer::RmsNormParam rmsMlpNormParam;
    rmsMlpNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    rmsMlpNormParam.normParam.quantInputScale = param.inputScale_gate_up;
    rmsMlpNormParam.normParam.quantInputOffset = param.inputOffset_gate_up;
    rmsMlpNormParam.normParam.quantType = atb::infer::QUANT_INT8;
    CreateOperation(rmsMlpNormParam, &selfNormNode.operation);
    selfNormNode.inTensorIds = {INTERNAL_SELFRESIDUALADDOUT, IN_SELFOUTNORMWEIGHT, IN_SELFOUTNORMBIAS};
    selfNormNode.outTensorIds = {INTERNAL_SELFNORMOUT};

    atb_speed::telechat::CommonMlpQuantParam mlpQuantParam;
    mlpQuantParam.inputScale_down_proj = param.inputScale_down_proj;
    mlpQuantParam.inputOffset_down_proj = param.inputOffset_down_proj;
    mlpQuantParam.isFloat = param.isFloatDownLayer;
    atb_speed::telechat::CommonMlpQuant(mlpQuantParam, &mlpQuantNode.operation);
    mlpQuantNode.inTensorIds = {INTERNAL_SELFNORMOUT,
                                IN_MLPGATEWEIGHT,
                                IN_MLPLINEARDEQSCALEGATE,
                                IN_LINEARBIASGATE,
                                IN_MLPUPWEIGHT,
                                IN_MLPLINEARDEQSCALEUP,
                                IN_LINEARBIASUP,
                                IN_MLPDOWNWEIGHT,
                                IN_MLPLINEARDEQSCALEDOWN,
                                IN_LINEARBIASDOWN};
    mlpQuantNode.outTensorIds = {INTERNAL_MLPOUT};

    CreateOperation(addParam, &mlpResidualAddNode.operation);
    mlpResidualAddNode.inTensorIds = {INTERNAL_SELFRESIDUALADDOUT, INTERNAL_MLPOUT};
    mlpResidualAddNode.outTensorIds = {OUT_TELECHATLAYEROUT};

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                atb::SVector<atb::TensorDesc> &outTensorDescs) {
        const atb::TensorDesc &keyTensorDesc = inTensorDescs.at(IN_PASTKEY);
        const atb::TensorDesc &valueTensorDesc = inTensorDescs.at(IN_PASTVALUE);
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
    };

    atb::CreateOperation(opGraph, operation);
    return atb::NO_ERROR;
}
}   // namespace telechat
}   // namespace atb_speed