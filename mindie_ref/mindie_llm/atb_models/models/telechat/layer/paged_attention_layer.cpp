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
    IN_K_CACHE,
    IN_V_CACHE,
    IN_BLOCK_TABLES,
    IN_SLOTS,
    IN_INPUT_LENGTHS,   // 18
    IN_HOLDER,

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
static const uint64_t NODE_COUNT = 12;
static const uint64_t REASHAPE_DIMNUM = 3;
static const uint64_t ROTARY_COEFF = 2;

void ReshapeHeads(const atb::Dims &oldShape, atb::Dims &newShape, int headNum)
{
    newShape.dimNum = REASHAPE_DIMNUM;
    newShape.dims[0] = oldShape.dims[0];
    newShape.dims[1] = headNum;
    newShape.dims[2] = oldShape.dims[1] / headNum;
}

void from_json(const nlohmann::json &paramJson, PALayerParam &param)
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
    if (paramJson.contains("transposedWeight")) {
        paramJson.at("transposedWeight").get_to(param.transposedWeight);
    }
    if (paramJson.contains("isPrefill")) {
        paramJson.at("isPrefill").get_to(param.isPrefill);
    }
    if (paramJson.contains("backend")) {
        paramJson.at("backend").get_to(param.backend);
    }
}

atb::Status PALayer(const PALayerParam &param, atb::Operation **operation)
{
    ATB_LOG(INFO) << __func__ << " called, headNum: " << param.headNum;
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
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
    atb::Node &reshapeAndCacheNode = opGraph.nodes.at(nodeId++);
    atb::Node &attentionNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfOutLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfResidualAddNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfNormNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpResidualAddNode = opGraph.nodes.at(nodeId++);

    // rmsNorm
    ATB_LOG(INFO) << "RmsNorm";
    atb::infer::RmsNormParam rmsNormParam;
    rmsNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    rmsNormParam.normParam.epsilon = param.rmsNormEps;
    CREATE_OPERATION(rmsNormParam, &inputNormNode.operation);
    inputNormNode.inTensorIds = { IN_HIDDENSTATES, IN_NORMWEIGHT };
    inputNormNode.outTensorIds = { INTERNAL_INPUTNORMOUT };

    // q
    ATB_LOG(INFO) << "Linear Q";
    atb::infer::LinearParam linearQParam;
    linearQParam.hasBias = false;
    linearQParam.transposeB = param.transposedWeight;
    CREATE_OPERATION(linearQParam, &mixedQLinearNode.operation);
    mixedQLinearNode.inTensorIds = { INTERNAL_INPUTNORMOUT, IN_QMIXEDWEIGHT };
    mixedQLinearNode.outTensorIds = { INTERNAL_QMIXEDLINEAROUT };

    // kv
    ATB_LOG(INFO) << "Linear KV";
    atb::infer::LinearParam linearKVParam;
    linearKVParam.hasBias = false;
    linearKVParam.transposeB = param.transposedWeight;
    CREATE_OPERATION(linearKVParam, &mixedKVLinearNode.operation);
    mixedKVLinearNode.inTensorIds = { INTERNAL_INPUTNORMOUT, IN_KVMIXEDWEIGHT };
    mixedKVLinearNode.outTensorIds = { INTERNAL_KVMIXEDLINEAROUT };

    ATB_LOG(INFO) << "Split";
    atb::infer::SplitParam splitParam;
    splitParam.splitDim = -1;
    splitParam.splitNum = 2;
    CREATE_OPERATION(splitParam, &splitKVNode.operation);
    splitKVNode.inTensorIds = { INTERNAL_KVMIXEDLINEAROUT };
    splitKVNode.outTensorIds = { INTERNAL_KMIXEDLINEAROUT, INTERNAL_VMIXEDLINEAROUT };
    splitKVNode.inTensorReshapeFuncs.resize(splitKVNode.inTensorIds.size());
    splitKVNode.inTensorReshapeFuncs.at(0) = [=](const atb::Dims &oldShape, atb::Dims &newShape) { 
        ReshapeHeads(oldShape, newShape, param.headNum);
    };

    ATB_LOG(INFO) << "ROPE";
    atb::infer::RopeParam ropeParam;
    ropeParam.rotaryCoeff = ROTARY_COEFF;
    atb_speed::telechat::Rope(ropeParam, &ropeNode.operation);
    ropeNode.inTensorIds = { INTERNAL_QMIXEDLINEAROUT, INTERNAL_KMIXEDLINEAROUT, IN_COSEMBED, IN_SINEMBED, IN_INPUT_LENGTHS };
    ropeNode.outTensorIds = { INTERNAL_POSITIONEMBEDQ, INTERNAL_POSITIONEMBEDK };
    ropeNode.inTensorReshapeFuncs.resize(ropeNode.inTensorIds.size());
    ropeNode.inTensorReshapeFuncs[1] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2;
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = oldShape.dims[1] * oldShape.dims[2];
    };

    atb::infer::ReshapeAndCacheParam reshapeCacheParm;
    CREATE_OPERATION(reshapeCacheParm, &reshapeAndCacheNode.operation);
    reshapeAndCacheNode.inTensorIds = { INTERNAL_POSITIONEMBEDK, INTERNAL_VMIXEDLINEAROUT, IN_K_CACHE, IN_V_CACHE,
                                       IN_SLOTS };
    reshapeAndCacheNode.outTensorIds = { IN_K_CACHE, IN_V_CACHE };
    reshapeAndCacheNode.inTensorReshapeFuncs.resize(reshapeAndCacheNode.inTensorIds.size());
    reshapeAndCacheNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        ReshapeHeads(oldShape, newShape, param.headNum);
    };

    if (param.isPrefill) {
        atb::infer::SelfAttentionParam faEnParam;
        faEnParam.headNum = param.headNum;
        faEnParam.qScale = 1.0f;
        faEnParam.qkScale = 1.0 / sqrt(param.dk);
        faEnParam.kvHeadNum = param.headNum;
        faEnParam.calcType = atb::infer::SelfAttentionParam::PA_ENCODER;
        CREATE_OPERATION(faEnParam, &attentionNode.operation);
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
    } else {
        atb::infer::PagedAttentionParam paDeParam;
        paDeParam.headNum = param.headNum;
        paDeParam.qkScale = 1.0 / sqrt(param.dk);
        paDeParam.kvHeadNum = param.headNum;
        CREATE_OPERATION(paDeParam, &attentionNode.operation);
        attentionNode.inTensorIds = {INTERNAL_POSITIONEMBEDQ, IN_K_CACHE, IN_V_CACHE, IN_BLOCK_TABLES,
                                     IN_INPUT_LENGTHS};
        attentionNode.outTensorIds = {INTERNAL_SELFOUT};
        attentionNode.inTensorReshapeFuncs.resize(attentionNode.inTensorIds.size());
        attentionNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            ReshapeHeads(oldShape, newShape, param.headNum);
        };
    }
    
    atb_speed::common::ParallelParamV2 selfOutLinearParam;
    selfOutLinearParam.commParam.rank = param.rank;
    selfOutLinearParam.commParam.rankSize = param.rankSize;
    selfOutLinearParam.isBias = true;
    selfOutLinearParam.commParam.backend = param.backend;
    selfOutLinearParam.transposeB = param.transposedWeight;
    atb_speed::common::RowParallelLinearV2(selfOutLinearParam, &selfOutLinearNode.operation);
    selfOutLinearNode.inTensorIds = {
        INTERNAL_SELFOUT, IN_SELFOUTLINEARWEIGHT, IN_SELFOUTLINEARBIAS, IN_HOLDER, IN_HOLDER, IN_HOLDER, IN_HOLDER};
    selfOutLinearNode.outTensorIds = { INTERNAL_SELFLINEAROUT };
    selfOutLinearNode.inTensorReshapeFuncs.resize(selfOutLinearNode.inTensorIds.size());
    selfOutLinearNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2;                                    // 2: dim num
        newShape.dims[0] = oldShape.dims[0];                    // 0: dim 0, n tokens
        newShape.dims[1] = oldShape.dims[1] * oldShape.dims[2]; // 1 hidden size: old 1, head num , old 2 head size
    };

    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CREATE_OPERATION(addParam, &selfResidualAddNode.operation);
    selfResidualAddNode.inTensorIds = { IN_HIDDENSTATES, INTERNAL_SELFLINEAROUT };
    selfResidualAddNode.outTensorIds = { INTERNAL_SELFRESIDUALADDOUT };
    
    atb::infer::RmsNormParam rmsMlpNormParam;
    rmsMlpNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    rmsMlpNormParam.normParam.epsilon = param.rmsNormEps;
    CREATE_OPERATION(rmsMlpNormParam, &selfNormNode.operation);
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
    mlpQuantParam.commDownParam.backend = param.backend;

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
    CREATE_OPERATION(addParam, &mlpResidualAddNode.operation);
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