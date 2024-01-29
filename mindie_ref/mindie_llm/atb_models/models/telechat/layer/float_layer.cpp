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

#include "float_layer.h"
#include "models/telechat/operation/rope.h"
#include "models/telechat/operation/mlp_gate_v2.h"
#include "models/telechat/operation/parallel_layer_v2.h"

namespace atb_speed {
namespace telechat {
enum FloatFALayerTensorId {
    IN_HIDDENSTATES = 0,
    IN_QMIXEDWEIGHT,
    IN_KVMIXEDWEIGHT,
    IN_SELFOUTLINEARWEIGHT,
    IN_SELFOUTLINEARBIAS,
    IN_MLPGATEWEIGHT,
    IN_MLPUPWEIGHT,
    IN_MLPDOWNWEIGHT,
    IN_LINEARBIASDOWN,
    IN_NORMWEIGHT,
    IN_SELFOUTNORMWEIGHT,
    IN_COSEMBED,
    IN_SINEMBED,
    IN_ATTENTIONMASK,
    IN_PASTKEY,
    IN_PASTVALUE,
    IN_TOKENOFFSET,
    IN_SEQLEN,
    IN_HOLDER,
    IN_LAYERID,

    OUT_TELECHATLAYEROUT,

    INTERNAL_INPUTNORMOUT,
    INTERNAL_QMIXEDLINEAROUT,
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

static const uint64_t IN_TENSOR_COUNT = 20;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERNAL_TENSOR_COUNT = 12;
static const uint64_t NODE_COUNT = 11;

atb::Status FloatFALayer(const FloatFALayerParam &param, atb::Operation **operation)
{
    ATB_LOG(INFO) << "enter float layer";
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
    atb::Node &selfOutLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfResidualAddNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfNormNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpQuantNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpResidualAddNode = opGraph.nodes.at(nodeId++);

    ATB_LOG(INFO) << "RmsNorm";
    atb::infer::RmsNormParam rmsNormParam;
    rmsNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    rmsNormParam.normParam.epsilon = param.rmsNormEps;
    CreateOperation(rmsNormParam, &inputNormNode.operation);
    inputNormNode.inTensorIds = { IN_HIDDENSTATES, IN_NORMWEIGHT };
    inputNormNode.outTensorIds = { INTERNAL_INPUTNORMOUT };

    ATB_LOG(INFO) << "Linear Q";
    atb::infer::LinearParam linearQParam = { false, true, false };
    CreateOperation(linearQParam, &mixedQLinearNode.operation);
    mixedQLinearNode.inTensorIds = { INTERNAL_INPUTNORMOUT, IN_QMIXEDWEIGHT };
    mixedQLinearNode.outTensorIds = { INTERNAL_QMIXEDLINEAROUT };

    ATB_LOG(INFO) << "Linear KV";
    atb::infer::LinearParam linearKVParam = { false, true, false };
    CreateOperation(linearKVParam, &mixedKVLinearNode.operation);
    mixedKVLinearNode.inTensorIds = { INTERNAL_INPUTNORMOUT, IN_KVMIXEDWEIGHT };
    mixedKVLinearNode.outTensorIds = { INTERNAL_KVMIXEDLINEAROUT };

    ATB_LOG(INFO) << "Split";
    atb::infer::SplitParam splitParam;
    splitParam.splitDim = 3;
    splitParam.splitNum = 2;
    CreateOperation(splitParam, &splitKVNode.operation);
    splitKVNode.inTensorIds = { INTERNAL_KVMIXEDLINEAROUT };
    splitKVNode.outTensorIds = { INTERNAL_KMIXEDLINEAROUT, INTERNAL_VMIXEDLINEAROUT };
    splitKVNode.inTensorReshapeFuncs.resize(splitKVNode.inTensorIds.size());
    splitKVNode.inTensorReshapeFuncs.at(0) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 4;
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = oldShape.dims[1];
        newShape.dims[2] = param.headNum;
        newShape.dims[3] = oldShape.dims[2] / param.headNum;
    };

    ATB_LOG(INFO) << "ROPE";
    atb_speed::telechat::RopeParam ropeParam;
    ropeParam.rotaryCoeff = 2;
    ropeParam.headNum = param.headNum;
    atb_speed::telechat::Rope(ropeParam, &ropeNode.operation);
    ropeNode.inTensorIds = { INTERNAL_QMIXEDLINEAROUT, INTERNAL_KMIXEDLINEAROUT, IN_COSEMBED, IN_SINEMBED, IN_SEQLEN };
    ropeNode.outTensorIds = { INTERNAL_POSITIONEMBEDQ, INTERNAL_POSITIONEMBEDK };

    ATB_LOG(INFO) << "KV Cache";
    atb::infer::SelfAttentionParam selfAttentionKvCacheParam;
    selfAttentionKvCacheParam.headNum = param.headNum;
    selfAttentionKvCacheParam.headDim = param.dk;
    selfAttentionKvCacheParam.qScale = 1.0f;
    selfAttentionKvCacheParam.qkScale = 1.0f / std::sqrt(param.dk);
    selfAttentionKvCacheParam.isFusion = true;
    CreateOperation(selfAttentionKvCacheParam, &selfAttentionKvCacheFusedNode.operation);
    selfAttentionKvCacheFusedNode.inTensorIds = { INTERNAL_POSITIONEMBEDQ,
                                                  INTERNAL_POSITIONEMBEDK,
                                                  INTERNAL_VMIXEDLINEAROUT,
                                                  IN_PASTKEY,
                                                  IN_PASTVALUE,
                                                  IN_ATTENTIONMASK,
                                                  IN_TOKENOFFSET,
                                                  IN_SEQLEN,
                                                  IN_LAYERID };
    selfAttentionKvCacheFusedNode.outTensorIds = { INTERNAL_SELFOUT };

    ATB_LOG(INFO) << "Parallel linear";
    atb_speed::telechat::ParallelParamV2 linearBiasParam;
    linearBiasParam.commParam.rank = param.rank;
    linearBiasParam.commParam.rankSize = param.rankSize;
    linearBiasParam.isBias = true;
    linearBiasParam.isQuant = false;
    linearBiasParam.transposeB = true;
    atb_speed::telechat::RowParallelLinearV2(linearBiasParam, &selfOutLinearNode.operation);
    selfOutLinearNode.inTensorIds = { INTERNAL_SELFOUT, IN_SELFOUTLINEARWEIGHT, IN_SELFOUTLINEARBIAS, IN_HOLDER,
                                      IN_HOLDER };
    selfOutLinearNode.outTensorIds = { INTERNAL_SELFLINEAROUT };

    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CreateOperation(addParam, &selfResidualAddNode.operation);
    selfResidualAddNode.inTensorIds = { IN_HIDDENSTATES, INTERNAL_SELFLINEAROUT };
    selfResidualAddNode.outTensorIds = { INTERNAL_SELFRESIDUALADDOUT };

    atb::infer::RmsNormParam rmsMlpNormParam;
    rmsMlpNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    rmsMlpNormParam.normParam.epsilon = param.rmsNormEps;
    CreateOperation(rmsMlpNormParam, &selfNormNode.operation);
    selfNormNode.inTensorIds = { INTERNAL_SELFRESIDUALADDOUT, IN_SELFOUTNORMWEIGHT };
    selfNormNode.outTensorIds = { INTERNAL_SELFNORMOUT };

    ATB_LOG(INFO) << "MLP";
    atb_speed::telechat::MlpGateParamV2 mlpQuantParam;
    mlpQuantParam.activationType = atb::infer::ActivationType::ACTIVATION_SWISH;
    mlpQuantParam.isBias = false;
    mlpQuantParam.transposeB = true;
    mlpQuantParam.isPack = false;
    mlpQuantParam.isUpQuant = false;
    mlpQuantParam.isGateQuant = false;
    mlpQuantParam.isDownQuant = false;
    mlpQuantParam.commDownParam.rankSize = param.rankSize;
    mlpQuantParam.commDownParam.rank = param.rank;

    atb_speed::telechat::MlpGateLayerV2(mlpQuantParam, &mlpQuantNode.operation);
    mlpQuantNode.inTensorIds = {
        INTERNAL_SELFNORMOUT,
        IN_MLPUPWEIGHT,
        IN_MLPGATEWEIGHT,
        IN_MLPDOWNWEIGHT,
        IN_HOLDER,
        IN_HOLDER,
        IN_HOLDER,
        IN_HOLDER,
        IN_HOLDER,
        IN_LINEARBIASDOWN,
        IN_HOLDER,
        IN_HOLDER,
        IN_HOLDER,
    };
    mlpQuantNode.outTensorIds = { INTERNAL_MLPOUT };

    ATB_LOG(INFO) << "residual add";
    CreateOperation(addParam, &mlpResidualAddNode.operation);
    mlpResidualAddNode.inTensorIds = { INTERNAL_SELFRESIDUALADDOUT, INTERNAL_MLPOUT };
    mlpResidualAddNode.outTensorIds = { OUT_TELECHATLAYEROUT };

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        const atb::TensorDesc &keyTensorDesc = inTensorDescs.at(IN_PASTKEY);
        const atb::TensorDesc &valueTensorDesc = inTensorDescs.at(IN_PASTVALUE);
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
    };

    CREATE_OPERATION(opGraph, operation);
    ATB_LOG(INFO) << "end float layer";
    return atb::NO_ERROR;
}
}  // namespace telechat
}  // namespace atb_speed