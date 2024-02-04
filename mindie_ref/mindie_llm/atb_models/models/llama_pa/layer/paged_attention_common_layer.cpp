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
#include "paged_attention_common_layer.h"

#include "layers/mlp_gate_v2.h"
#include "layers/parallel_layer_v2.h"

namespace atb_speed {
namespace llama_pa {

static const uint64_t IN_TENSOR_COUNT = 35;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 12;
static const uint64_t QUANT_NODE_COUNT = 12;
static const uint64_t FLOAT_NODE_COUNT = 11;
static const uint64_t REASHAPE_DIMNUM = 3;
static const uint64_t ROTARY_COEFF = 2;

enum PaCommonLayerTensorId : int {
    IN_HIDDENSTATES = 0,
    // 浮点权重
    IN_NORMWEIGHT,
    IN_QMIXDWEIGHT,
    IN_KMIXDWEIGHT,
    IN_VMIXDWEIGHT,
    IN_SELFOUTLINEARWEIGHT,
    IN_SELFOUTNORMWEIGHT,
    IN_MLPUPWEIGHT,
    IN_MLPGATEWEIGHT,
    IN_MLPDOWNWEIGHT,
    // 量化权重
    IN_QMIXD_DEQSCALE,
    IN_QMIXD_BIAS,
    IN_KMIXD_DEQSCALE,
    IN_KMIXD_BIAS,
    IN_VMIXD_DEQSCALE,
    IN_VMIXD_BIAS,
    IN_SELFOUTLINEAR_DEQSCALE,
    IN_SELFOUTLINEAR_BIAS,
    IN_MLPGATE_DEQSCALE,
    IN_MLPGATE_BIAS,
    IN_MLPDOWN_DEQSCALE,
    IN_MLPDOWN_BIAS,
    IN_MLPUP_DEQSCALE,
    IN_MLPUP_BIAS,
    // anti所需
    IN_NORM_BIAS,
    IN_SELFOUTNORM_BIAS,
    // 入参
    IN_COSEMBED,
    IN_SINEMBED,
    IN_ATTENTIONMASK,
    IN_K_CACHE,
    IN_V_CACHE,
    IN_BLOCK_TABLES,
    IN_SLOTS,
    IN_INPUT_LENGTHS,   // 33
    IN_HOLDER,

    OUT_LAYEROUT,

    INTERMIDATE_INPUTNORMOUT,
    INTERMIDATE_MIXEDQ,
    INTERMIDATE_MIXEDK,
    INTERMIDATE_MIXEDV,
    INTERMIDATE_POSITIONEMBEDQ,
    INTERMIDATE_POSITIONEMBEDK,
    INTERMIDATE_ATTENTIONOUT,
    INTERMIDATE_SELFLINEAROUT,
    INTERMIDATE_SELFRESIDUALADDOUT,
    INTERMIDATE_SELFNORMOUT,
    INTERMIDATE_MLPOUT,
    INTERMIDATE_QKVMIXEDLINEAROUT, // float use
};

void from_json(const nlohmann::json &paramJson, PaCommonLayerParam &param)
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
    if (paramJson.contains("isBF16")) {
        paramJson.at("isBF16").get_to(param.isBF16);
    }
    if (paramJson.contains("isQuant")) {
        paramJson.at("isQuant").get_to(param.isQuant);
    }
    // 量化参数
    if (paramJson.contains("qkvInputScale")) {
        paramJson.at("qkvInputScale").get_to(param.qkvInputScale);
    }
    if (paramJson.contains("qkvInputOffset")) {
        paramJson.at("qkvInputOffset").get_to(param.qkvInputOffset);
    }
    if (paramJson.contains("denseInputScale")) {
        paramJson.at("denseInputScale").get_to(param.denseInputScale);
    }
    if (paramJson.contains("denseInputOffset")) {
        paramJson.at("denseInputOffset").get_to(param.denseInputOffset);
    }
    if (paramJson.contains("selfLnInputScale")) {
        paramJson.at("selfLnInputScale").get_to(param.selfLnInputScale);
    }
    if (paramJson.contains("selfLnInputOffset")) {
        paramJson.at("selfLnInputOffset").get_to(param.selfLnInputOffset);
    }
    if (paramJson.contains("ffnOutInputScale")) {
        paramJson.at("ffnOutInputScale").get_to(param.ffnOutInputScale);
    }
    if (paramJson.contains("ffnOutInputOffset")) {
        paramJson.at("ffnOutInputOffset").get_to(param.ffnOutInputOffset);
    }
}

void CommonReshapeHeads(const atb::Dims &oldShape, atb::Dims &newShape, int headNum)
{
    newShape.dimNum = REASHAPE_DIMNUM;
    newShape.dims[0] = oldShape.dims[0];
    newShape.dims[1] = headNum;
    newShape.dims[2] = oldShape.dims[1] / headNum;
}

atb::Status PaCommonLayer(const PaCommonLayerParam &param, atb::Operation **operation)
{
    ATB_LOG(INFO) << __func__ << " called, headNum: " << param.headNum << ", is Quant:" << param.isQuant;
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    if (param.isQuant) {
        // 量化少一个中间变量
        opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT - 1;
        opGraph.nodes.resize(QUANT_NODE_COUNT);
    } else {
        opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
        opGraph.nodes.resize(FLOAT_NODE_COUNT);
    }
    if (param.isPrefill) {
        opGraph.name = "Prefill_Common_Pa_Layer";
    } else {
        opGraph.name = "Decoder_Common_Pa_Layer";
    }

    size_t nodeId = 0;
    atb::Node &inputNormNode = opGraph.nodes.at(nodeId++);

    if (param.isQuant) {
        atb::infer::RmsNormParam rmsNormParam;
        rmsNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
        rmsNormParam.normParam.epsilon = param.rmsNormEps;
        rmsNormParam.normParam.quantInputScale = param.qkvInputScale;
        rmsNormParam.normParam.quantInputOffset = param.qkvInputOffset;
        rmsNormParam.normParam.quantType = atb::infer::QUANT_INT8;
        CREATE_OPERATION(rmsNormParam, &inputNormNode.operation);
        inputNormNode.inTensorIds = {IN_HIDDENSTATES, IN_NORMWEIGHT, IN_NORM_BIAS};
        inputNormNode.outTensorIds = {INTERMIDATE_INPUTNORMOUT};

        atb::Node &mixdQLinearNode = opGraph.nodes.at(nodeId++);
        atb::infer::LinearQuantParam quantQkvLinearParam;
        quantQkvLinearParam.transposeB = true;
        CREATE_OPERATION(quantQkvLinearParam, &mixdQLinearNode.operation);
        mixdQLinearNode.inTensorIds = {INTERMIDATE_INPUTNORMOUT, IN_QMIXDWEIGHT, IN_QMIXD_BIAS, IN_QMIXD_DEQSCALE};
        mixdQLinearNode.outTensorIds = {INTERMIDATE_MIXEDQ};

        atb::Node &mixdKLinearNode = opGraph.nodes.at(nodeId++);
        CREATE_OPERATION(quantQkvLinearParam, &mixdKLinearNode.operation);
        mixdKLinearNode.inTensorIds = {INTERMIDATE_INPUTNORMOUT, IN_KMIXDWEIGHT, IN_KMIXD_BIAS, IN_KMIXD_DEQSCALE};
        mixdKLinearNode.outTensorIds = {INTERMIDATE_MIXEDK};

        atb::Node &mixdVLinearNode = opGraph.nodes.at(nodeId++);
        CREATE_OPERATION(quantQkvLinearParam, &mixdVLinearNode.operation);
        mixdVLinearNode.inTensorIds = {INTERMIDATE_INPUTNORMOUT, IN_VMIXDWEIGHT, IN_VMIXD_BIAS, IN_VMIXD_DEQSCALE};
        mixdVLinearNode.outTensorIds = {INTERMIDATE_MIXEDV};
    } else {
        // float
        atb::infer::RmsNormParam rmsNormParam;
        rmsNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
        rmsNormParam.normParam.epsilon = param.rmsNormEps;
        CREATE_OPERATION(rmsNormParam, &inputNormNode.operation);
        inputNormNode.inTensorIds = {IN_HIDDENSTATES, IN_NORMWEIGHT};
        inputNormNode.outTensorIds = {INTERMIDATE_INPUTNORMOUT};

        // qkv
        atb::Node &qkvLinearNode = opGraph.nodes.at(nodeId++);
        atb::infer::LinearParam linearParam = {false, false, false};
        CREATE_OPERATION(linearParam, &qkvLinearNode.operation);
        qkvLinearNode.inTensorIds = {INTERMIDATE_INPUTNORMOUT, IN_QMIXDWEIGHT};
        qkvLinearNode.outTensorIds = {INTERMIDATE_QKVMIXEDLINEAROUT};

        atb::Node &splitQkvNode = opGraph.nodes.at(nodeId++);
        atb::infer::SplitParam splitParam = {-1, 3};
        CREATE_OPERATION(splitParam, &splitQkvNode.operation);
        splitQkvNode.inTensorIds = {INTERMIDATE_QKVMIXEDLINEAROUT};
        splitQkvNode.outTensorIds = {INTERMIDATE_MIXEDQ, INTERMIDATE_MIXEDK, INTERMIDATE_MIXEDV};
        splitQkvNode.inTensorReshapeFuncs.resize(splitQkvNode.inTensorIds.size());
    }

    atb::Node &ropeNode = opGraph.nodes.at(nodeId++);
    atb::Node &reshapeAndCacheNode = opGraph.nodes.at(nodeId++);
    atb::Node &attentionNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfOutLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfResidualAddNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfNormNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpResidualAddNode = opGraph.nodes.at(nodeId++);
    
    atb::infer::RopeParam ropeParam;
    ropeParam.rotaryCoeff = ROTARY_COEFF;
    CREATE_OPERATION(ropeParam, &ropeNode.operation);
    ropeNode.inTensorIds = {INTERMIDATE_MIXEDQ, INTERMIDATE_MIXEDK, IN_COSEMBED, IN_SINEMBED, IN_INPUT_LENGTHS};
    ropeNode.outTensorIds = {INTERMIDATE_POSITIONEMBEDQ, INTERMIDATE_POSITIONEMBEDK};

    atb::infer::ReshapeAndCacheParam reshapeCacheParm;
    CREATE_OPERATION(reshapeCacheParm, &reshapeAndCacheNode.operation);
    reshapeAndCacheNode.inTensorIds = {INTERMIDATE_POSITIONEMBEDK, INTERMIDATE_MIXEDV, IN_K_CACHE, IN_V_CACHE,
                                       IN_SLOTS};
    reshapeAndCacheNode.outTensorIds = {};
    reshapeAndCacheNode.inTensorReshapeFuncs.resize(reshapeAndCacheNode.inTensorIds.size());
    reshapeAndCacheNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        CommonReshapeHeads(oldShape, newShape, param.headNum);
    };
    reshapeAndCacheNode.inTensorReshapeFuncs[1] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        CommonReshapeHeads(oldShape, newShape, param.headNum);
    };

    if (param.isPrefill) {
        atb::infer::SelfAttentionParam faEnParam;
        faEnParam.headNum = param.headNum;
        faEnParam.qkScale = 1.0 / sqrt(param.dk);
        faEnParam.kvHeadNum = param.headNum;
        faEnParam.isEncoder = true;
        CREATE_OPERATION(faEnParam, &attentionNode.operation);
        attentionNode.inTensorIds = {INTERMIDATE_POSITIONEMBEDQ, INTERMIDATE_POSITIONEMBEDK, INTERMIDATE_MIXEDV,
                                     IN_ATTENTIONMASK, IN_INPUT_LENGTHS};
        attentionNode.outTensorIds = {INTERMIDATE_ATTENTIONOUT};
        attentionNode.inTensorReshapeFuncs.resize(attentionNode.inTensorIds.size());
        attentionNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            CommonReshapeHeads(oldShape, newShape, param.headNum);
        };
        attentionNode.inTensorReshapeFuncs[1] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            CommonReshapeHeads(oldShape, newShape, param.headNum);
        };
        attentionNode.inTensorReshapeFuncs[2] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            CommonReshapeHeads(oldShape, newShape, param.headNum);
        };
    } else {
        atb::infer::PagedAttentionParam paDeParam;
        paDeParam.headNum = param.headNum;
        paDeParam.qkScale = 1.0 / sqrt(param.dk);
        paDeParam.kvHeadNum = param.headNum;
        paDeParam.isSupportAlibi = param.isBF16;
        CREATE_OPERATION(paDeParam, &attentionNode.operation);
        attentionNode.inTensorIds = {INTERMIDATE_POSITIONEMBEDQ, IN_K_CACHE, IN_V_CACHE, IN_BLOCK_TABLES,
                                     IN_INPUT_LENGTHS};
        attentionNode.outTensorIds = {INTERMIDATE_ATTENTIONOUT};
        attentionNode.inTensorReshapeFuncs.resize(attentionNode.inTensorIds.size());
        attentionNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            CommonReshapeHeads(oldShape, newShape, param.headNum);
        };
    }

    if (param.isQuant) {
        atb_speed::common::ParallelParamV2 selfOutLinearParam;
        selfOutLinearParam.commParam.rank = param.rank;
        selfOutLinearParam.commParam.rankSize = param.rankSize;
        selfOutLinearParam.commParam.backend = param.backend;
        selfOutLinearParam.isBias = true;
        selfOutLinearParam.isQuant = true;
        selfOutLinearParam.transposeB = true;
        selfOutLinearParam.quantParam.quantType = atb::infer::QUANT_INT8;
        selfOutLinearParam.quantParam.isQuantOp = true;
        selfOutLinearParam.quantParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_QUANT;
        selfOutLinearParam.quantParam.inputScale = param.denseInputScale;
        selfOutLinearParam.quantParam.inputOffset = param.denseInputOffset;
        atb_speed::common::RowParallelLinearV2(selfOutLinearParam, &selfOutLinearNode.operation);
        selfOutLinearNode.inTensorIds = {INTERMIDATE_ATTENTIONOUT, IN_SELFOUTLINEARWEIGHT,
                                        IN_SELFOUTLINEAR_BIAS, IN_SELFOUTLINEAR_DEQSCALE, IN_HOLDER,
                                        IN_HOLDER, IN_HOLDER};
        selfOutLinearNode.outTensorIds = {INTERMIDATE_SELFLINEAROUT};
        selfOutLinearNode.inTensorReshapeFuncs.resize(selfOutLinearNode.inTensorIds.size());
        selfOutLinearNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            newShape.dimNum = 2;
            newShape.dims[0] = oldShape.dims[0];
            newShape.dims[1] = oldShape.dims[1] * oldShape.dims[2];
        };
    } else {
        // float
        atb_speed::common::ParallelParamV2 selfOutLinearParam;
        selfOutLinearParam.commParam.rank = param.rank;
        selfOutLinearParam.commParam.rankSize = param.rankSize;
        selfOutLinearParam.commParam.backend = param.backend;
        selfOutLinearParam.isBias = false;
        selfOutLinearParam.transposeB = param.transposedWeight;
        atb_speed::common::RowParallelLinearV2(selfOutLinearParam, &selfOutLinearNode.operation);
        selfOutLinearNode.inTensorIds = {INTERMIDATE_ATTENTIONOUT, IN_SELFOUTLINEARWEIGHT,
                                         IN_HOLDER, IN_HOLDER, IN_HOLDER, IN_HOLDER, IN_HOLDER};
        selfOutLinearNode.outTensorIds = {INTERMIDATE_SELFLINEAROUT};
        selfOutLinearNode.inTensorReshapeFuncs.resize(selfOutLinearNode.inTensorIds.size());
        selfOutLinearNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            newShape.dimNum = 2;  // dimNum is 2
            newShape.dims[0] = oldShape.dims[0];
            newShape.dims[1] = oldShape.dims[1] * oldShape.dims[2];
        };
    }

    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CREATE_OPERATION(addParam, &selfResidualAddNode.operation);
    selfResidualAddNode.inTensorIds = {IN_HIDDENSTATES, INTERMIDATE_SELFLINEAROUT};
    selfResidualAddNode.outTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT};

    if (param.isQuant) {
        atb::infer::RmsNormParam selfNormParam;
        selfNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
        selfNormParam.normParam.epsilon = param.rmsNormEps;
        selfNormParam.normParam.quantInputScale = param.selfLnInputScale;
        selfNormParam.normParam.quantInputOffset = param.selfLnInputOffset;
        selfNormParam.normParam.quantType = atb::infer::QUANT_INT8;
        CREATE_OPERATION(selfNormParam, &selfNormNode.operation);
        selfNormNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT, IN_SELFOUTNORMWEIGHT, IN_SELFOUTNORM_BIAS};
        selfNormNode.outTensorIds = {INTERMIDATE_SELFNORMOUT};

        atb_speed::common::MlpGateParamV2 mlpParam;
        mlpParam.isBias=true;
        mlpParam.isPack=false;
        mlpParam.isQuant=true;
        mlpParam.transposeB=true;
        mlpParam.commDownParam.rank = param.rank;
        mlpParam.commDownParam.rankSize = param.rankSize;
        mlpParam.commDownParam.backend = param.backend;
        mlpParam.quantUpParam.quantType = atb::infer::QUANT_INT8;
        mlpParam.quantUpParam.isQuantOp = false;
        mlpParam.quantGateParam.quantType = atb::infer::QUANT_INT8;
        mlpParam.quantGateParam.isQuantOp = false;
        mlpParam.activationType = atb::infer::ActivationType::ACTIVATION_SWISH;
        mlpParam.quantDownParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_QUANT;
        mlpParam.quantDownParam.inputScale = param.ffnOutInputScale;
        mlpParam.quantDownParam.inputOffset = param.ffnOutInputOffset;
        atb_speed::common::MlpGateLayerV2(mlpParam, &mlpNode.operation);
        mlpNode.inTensorIds = {INTERMIDATE_SELFNORMOUT, IN_MLPUPWEIGHT, IN_MLPGATEWEIGHT, IN_MLPDOWNWEIGHT,
                            IN_MLPUP_DEQSCALE, IN_MLPGATE_DEQSCALE, IN_MLPDOWN_DEQSCALE,
                            IN_MLPUP_BIAS, IN_MLPGATE_BIAS, IN_MLPDOWN_BIAS,
                            IN_HOLDER, IN_HOLDER, IN_HOLDER, IN_HOLDER, IN_HOLDER, IN_HOLDER,
                            IN_HOLDER, IN_HOLDER, IN_HOLDER};
        mlpNode.outTensorIds = {INTERMIDATE_MLPOUT};
    } else {
        // float
        atb::infer::RmsNormParam selfNormParam;
        selfNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
        selfNormParam.normParam.epsilon = param.rmsNormEps;
        CREATE_OPERATION(selfNormParam, &selfNormNode.operation);
        selfNormNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT, IN_SELFOUTNORMWEIGHT};
        selfNormNode.outTensorIds = {INTERMIDATE_SELFNORMOUT};

        atb_speed::common::MlpGateParamV2 mlpParam;
        mlpParam.commDownParam.rank = param.rank;
        mlpParam.commDownParam.rankSize = param.rankSize;
        mlpParam.commDownParam.backend = param.backend;
        mlpParam.activationType = atb::infer::ActivationType::ACTIVATION_SWISH;
        mlpParam.transposeB = false;
        mlpParam.isBias = false;
        mlpParam.isPack = true;
        atb_speed::common::MlpGateLayerV2(mlpParam, &mlpNode.operation);
        mlpNode.inTensorIds = {INTERMIDATE_SELFNORMOUT, IN_MLPUPWEIGHT, IN_HOLDER, IN_MLPDOWNWEIGHT,
                               IN_HOLDER, IN_HOLDER, IN_HOLDER,
                               IN_HOLDER, IN_HOLDER, IN_HOLDER,
                               IN_HOLDER, IN_HOLDER, IN_HOLDER,
                               IN_HOLDER, IN_HOLDER, IN_HOLDER,
                               IN_HOLDER, IN_HOLDER, IN_HOLDER};
        mlpNode.outTensorIds = {INTERMIDATE_MLPOUT};
    }

    CREATE_OPERATION(addParam, &mlpResidualAddNode.operation);
    mlpResidualAddNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT, INTERMIDATE_MLPOUT};
    mlpResidualAddNode.outTensorIds = {OUT_LAYEROUT};

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
    };

    atb::CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}

CommonFlashAttentionHostBinder::CommonFlashAttentionHostBinder() {}

CommonFlashAttentionHostBinder::~CommonFlashAttentionHostBinder() {}

void CommonFlashAttentionHostBinder::ParseParam(const nlohmann::json &paramJson)
{
    seqLen_.clear();
    for (auto item : paramJson["seqLen"]) {
        seqLen_.push_back(item.get<int32_t>());
    }
}

void CommonFlashAttentionHostBinder::BindTensor(atb::VariantPack &variantPack)
{
    variantPack.inTensors.at(IN_INPUT_LENGTHS).hostData = seqLen_.data();
}
} // namespace llama_pa
} // namespace atb_speed