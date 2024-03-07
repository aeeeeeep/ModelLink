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

#include "paged_attention_layer.h"
#include "layers/mlp_gate_v2.h"
#include "layers/parallel_layer_v2.h"
#include "models/telechat/operation/rope.h"
#include "models/telechat/operation/mlp_gate_v2.h"
#include "models/telechat/operation/parallel_layer_v2.h"

namespace atb_speed {
namespace telechat {
enum PALayerTensorId {
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
    IN_HOLDER,
    IN_K_CACHE,
    IN_V_CACHE,
    IN_BLOCK_TABLES,
    IN_SLOTS,
    IN_INPUT_LENGTHS,   // 33

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
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 12;
static const uint64_t NODE_COUNT = 11;
static const uint64_t REASHAPE_DIMNUM = 3;
static const uint64_t ROTARY_COEFF = 2;

void ReshapeHeads(const atb::Dims &oldShape, atb::Dims &newShape, int headNum)
{
    newShape.dimNum = REASHAPE_DIMNUM;
    newShape.dims[0] = oldShape.dims[0];
    newShape.dims[1] = headNum;
    newShape.dims[2] = oldShape.dims[1] / headNum;
}

atb::Status PALayer(const PALayerParam &param, atb::Operation **operation)
{
    ATB_LOG(INFO) << __func__ << " called, headNum: " << param.headNum;
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = IN_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);
    if (param.isPrefill) {
        opGraph.name = "Prefill_Pa_Layer";
    } else {
        opGraph.name = "Decoder_Pa_Layer";
    }

    size_t nodeId = 0;
    atb::Node &inputNormNode = opGraph.nodes.at(nodeId++);
    atb::Node &mixedQLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &mixedKVLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &splitKVNode = opGraph.nodes.at(nodeId++);
    atb::Node &ropeNode = opGraph.nodes.at(nodeId++);
    atb::Node &attentionNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfResidualAddNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfNormNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpResidualAddNode = opGraph.nodes.at(nodeId++);

    // rmsNorm
    ATB_LOG(INFO) << "RmsNorm";
    atb::infer::RmsNormParam rmsNormParam;
    rmsNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    rmsNormParam.normParam.epsilon = param.rmsNormEps;
    CreateOperation(rmsNormParam, &inputNormNode.operation);
    inputNormNode.inTensorIds = { IN_HIDDENSTATES, IN_NORMWEIGHT };
    inputNormNode.outTensorIds = { INTERNAL_INPUTNORMOUT };

    // q
    ATB_LOG(INFO) << "Linear Q";
    atb::infer::LinearParam linearQParam;
    linearQParam.hasBias = false;
    linearQParam.transposeB = param.transposedWeight;
    CreateOperation(linearQParam, &mixedQLinearNode.operation);
    mixedQLinearNode.inTensorIds = { INTERNAL_INPUTNORMOUT, IN_QMIXEDWEIGHT };
    mixedQLinearNode.outTensorIds = { INTERNAL_QMIXEDLINEAROUT };

    // kv
    ATB_LOG(INFO) << "Linear KV";
    atb::infer::LinearParam linearKVParam;
    linearKVParam.hasBias = false;
    linearKVParam.transposeB = param.transposedWeight;
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

    atb::Node &reshapeAndCacheNode = opGraph.nodes.at(nodeId++);
    ATB_LOG(INFO) << "ROPE";
    atb_speed::telechat::RopeParam ropeParam;
    ropeParam.rotaryCoeff = ROTARY_COEFF;
    ropeParam.headNum = param.headNum;
    CreateOperation(ropeParam, &ropeNode.operation);
    ropeNode.inTensorIds = { INTERNAL_QMIXEDLINEAROUT, INTERNAL_KMIXEDLINEAROUT, IN_COSEMBED, IN_SINEMBED, IN_INPUT_LENGTHS };
    ropeNode.outTensorIds = { INTERNAL_POSITIONEMBEDQ, INTERNAL_POSITIONEMBEDK };

    atb::infer::ReshapeAndCacheParam reshapeCacheParm;
    CreateOperation(reshapeCacheParm, &reshapeAndCacheNode.operation);
    reshapeAndCacheNode.inTensorIds = { INTERNAL_POSITIONEMBEDK, INTERNAL_VMIXEDLINEAROUT, IN_K_CACHE, IN_V_CACHE,
                                       IN_SLOTS };
    reshapeAndCacheNode.outTensorIds = { IN_K_CACHE, IN_V_CACHE };
    reshapeAndCacheNode.inTensorReshapeFuncs.resize(reshapeAndCacheNode.inTensorIds.size());
    reshapeAndCacheNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        ReshapeHeads(oldShape, newShape, param.headNum);
    };
    reshapeAndCacheNode.inTensorReshapeFuncs[1] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        ReshapeHeads(oldShape, newShape, param.headNum);
    };

    if (param.isPrefill) {
        atb::infer::SelfAttentionParam faEnParam;
        faEnParam.headNum = param.headNum;
        faEnParam.qkScale = 1.0 / sqrt(param.dk);
        faEnParam.kvHeadNum = param.headNum;
        CreateOperation(faEnParam, &attentionNode.operation);
        attentionNode.inTensorIds = {INTERNAL_POSITIONEMBEDQ, INTERNAL_POSITIONEMBEDK, INTERNAL_VMIXEDLINEAROUT,
                                     IN_ATTENTIONMASK, IN_INPUT_LENGTHS};
        attentionNode.outTensorIds = {INTERNAL_SELFOUT};
        attentionNode.inTensorReshapeFuncs.resize(attentionNode.inTensorIds.size());
        attentionNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            ReshapeHeads(oldShape, newShape, param.headNum);
        };
        attentionNode.inTensorReshapeFuncs[1] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            ReshapeHeads(oldShape, newShape, param.headNum);
        };
        attentionNode.inTensorReshapeFuncs[2] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            ReshapeHeads(oldShape, newShape, param.headNum);
        };
    } else {
        atb::infer::PagedAttentionParam paDeParam;
        paDeParam.headNum = param.headNum;
        paDeParam.qkScale = 1.0 / sqrt(param.dk);
        paDeParam.kvHeadNum = param.headNum;
        CreateOperation(paDeParam, &attentionNode.operation);
        attentionNode.inTensorIds = {INTERNAL_POSITIONEMBEDQ, IN_K_CACHE, IN_V_CACHE, IN_BLOCK_TABLES,
                                     IN_INPUT_LENGTHS};
        attentionNode.outTensorIds = {INTERNAL_SELFOUT};
        attentionNode.inTensorReshapeFuncs.resize(attentionNode.inTensorIds.size());
        attentionNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            ReshapeHeads(oldShape, newShape, param.headNum);
        };
    }
    
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

    atb_speed::telechat::MlpGateLayerV2(mlpQuantParam, &mlpNode.operation);
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
        IN_LINEARBIASDOWN,
        IN_HOLDER,
        IN_HOLDER,
        IN_HOLDER,
    };

    mlpNode.outTensorIds = { INTERNAL_MLPOUT };

    ATB_LOG(INFO) << "residual add";
    CreateOperation(addParam, &mlpResidualAddNode.operation);
    mlpResidualAddNode.inTensorIds = { INTERNAL_SELFRESIDUALADDOUT, INTERNAL_MLPOUT };
    mlpResidualAddNode.outTensorIds = { OUT_TELECHATLAYEROUT };

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
    };

    CREATE_OPERATION(opGraph, operation);
    ATB_LOG(INFO) << "end float layer";
    return atb::NO_ERROR;
}

FlashAttentionHostBinder::FlashAttentionHostBinder() {}

FlashAttentionHostBinder::~FlashAttentionHostBinder() {}

void FlashAttentionHostBinder::ParseParam(const nlohmann::json &paramJson)
{
    seqLen_.clear();
    for (auto item : paramJson["seqLen"]) {
        seqLen_.push_back(item.get<int32_t>());
    }
}

void FlashAttentionHostBinder::BindTensor(atb::VariantPack &variantPack)
{
    variantPack.inTensors.at(IN_INPUT_LENGTHS).hostData = seqLen_.data();
}

}  // namespace telechat
}  // namespace atb_speed