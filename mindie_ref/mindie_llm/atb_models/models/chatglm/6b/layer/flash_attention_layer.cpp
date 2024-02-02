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

#include "flash_attention_layer.h"
#include "layers/parallel_layer_v2.h"
#include "layers/mlp_gate_v2.h"
#include "layers/attention.h"

namespace atb_speed {
namespace chatglm_6b {
enum ChatglmCommonTensorId: int {
    IN_HIDDENSTATES = 0,
    IN_NORMWEIGHT,
    IN_NORMBIAS,
    IN_QKVMIXDWEIGHT,
    IN_QKVMIXDBIAS,
    IN_SELFOUTLINEARWEIGHT,
    IN_SELFOUTLINEARBIAS,
    IN_SELFOUTNORMWEIGHT,
    IN_SELFOUTNORMBIAS,
    IN_MLPLINEARWEIGHTUP,
    IN_MLPLINEARBIASUP,
    IN_MLPLINEARWEIGHTDOWN,
    IN_MLPLINEARBIASDOWN,

    IN_QKVMIXDDEQSCALE, // 量化独有权重
    IN_SELFOUTLINEARDEQSCALE,
    IN_MLPLINEARDEQSCALETUP,
    IN_MLPLINEARDEQSCALEDOWN,

    IN_QKVMIXDWEIGHT_INDEX, // 稀疏独有权重
    IN_QKVMIXDOFFSETX,
    IN_QKVMIXDWEIGHT_COMPRESSINFO,
    IN_SELFOUTLINEARWEIGHT_INDEX,
    IN_SELFOUTLINEAROFFSETX,
    IN_SELFOUTLINEARWEIGHT_COMPRESSINFO,
    IN_MLPLINEARWEIGHT_INDEX,
    IN_MLPLINEAROFFSETX,
    IN_MLPLINEARWEIGHT_COMPRESSINFO,
    IN_MLPLINEARWEIGHTDOWN_INDEX,
    IN_MLPLINEAROFFSETXDOWN,
    IN_MLPLINEARWEIGHTDOWN_COMPRESSINFO,

    IN_COS_TABLE,
    IN_SIN_TABLE,
    IN_SEQLEN,
    IN_ATTENTION_MASK,
    IN_PASTKEY,
    IN_PASTVALUE,
    IN_TOKEN_OFFSET,
    IN_BETA,
    IN_PLACE_HOLDER,
    IN_LAYER_ID,

    OUT_GLMLAYEROUT,

    INTERMEDIATE_INPUTNORMOUT,
    INTERMEDIATE_ATTENTION_OUTPUT,
    INTERMEDIATE_SELFRESIDUALMULSOUT,
    INTERMEDIATE_SELFRESIDUALADDOUT,
    INTERMEDIATE_SELFNORMOUT,
    INTERMEDIATE_MLPRESIDUALMULSOUT,
    INTERMEDIATE_MLPOUT,
};

static const uint64_t IN_TENSOR_COUNT = 39;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 7;
static const uint64_t NODE_COUNT = 8;

atb::Status CommonLayerFa(const CommonLayerParamFa &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.name = "CommonLayerFa";
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    atb::Node &inputNormNode = opGraph.nodes.at(nodeId++);
    atb::Node &flashAttentionWithROPENode = opGraph.nodes.at(nodeId++);
    atb::Node &selfResidualMulsNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfResidualAddNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfNormNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpResidualMulsNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpResidualAddNode = opGraph.nodes.at(nodeId++);

    atb::infer::LayerNormParam inputNormParam;
    inputNormParam.layerType = atb::infer::LayerNormParam::LayerNormType::LAYER_NORM_NORM;
    inputNormParam.normParam.beginNormAxis = 2;
    inputNormParam.normParam.beginParamsAxis = 1;
    if (param.quantmodel) {
        inputNormParam.normParam.quantInputScale = param.qkvInputScale;
        inputNormParam.normParam.quantInputOffset = param.qkvInputOffset;
        inputNormParam.normParam.quantType = atb::infer::QUANT_INT8;
        CREATE_OPERATION(inputNormParam, &inputNormNode.operation);
        inputNormNode.inTensorIds = {IN_HIDDENSTATES, IN_NORMWEIGHT, IN_BETA};
    } else {
        inputNormParam.normParam.epsilon = param.rmsNormEps;
        CREATE_OPERATION(inputNormParam, &inputNormNode.operation);
        inputNormNode.inTensorIds = {IN_HIDDENSTATES, IN_NORMWEIGHT, IN_NORMBIAS};
    }
    inputNormNode.outTensorIds = {INTERMEDIATE_INPUTNORMOUT};

    atb_speed::common::FlashAttentionWithPosEmbedding flashAttentionWithRoPEObj;
    atb_speed::common::FTWithROPEParam faWithROPEParam;
    // self attention param
    faWithROPEParam.isGroupedQueryAttention = false;
    faWithROPEParam.isCrossedWeight = true;
    faWithROPEParam.selfAttentionKvCacheParam.headDim = param.hiddenSizePerHead;
    faWithROPEParam.selfAttentionKvCacheParam.headNum = param.numHeadsPerPartition;
    faWithROPEParam.selfAttentionKvCacheParam.kvHeadNum = param.numHeadsPerPartition;
    faWithROPEParam.selfAttentionKvCacheParam.qScale = param.preScale;
    faWithROPEParam.selfAttentionKvCacheParam.qkScale = param.postScale;
    if (param.isEncoder) {
        faWithROPEParam.selfAttentionKvCacheParam.coderType = atb::infer::SelfAttentionParam::ENCODER;
    } else {
        faWithROPEParam.selfAttentionKvCacheParam.coderType = atb::infer::SelfAttentionParam::DECODER;
    }
    // RoPE param
    faWithROPEParam.rotaryCoeff = 2;
    faWithROPEParam.isHalfRotary = true;
    // self out linear param
    faWithROPEParam.selfOutLinearParam.commParam.rank = param.rank;
    faWithROPEParam.selfOutLinearParam.commParam.rankSize = param.rankSize;
    faWithROPEParam.selfOutLinearParam.commParam.backend = param.backend;
    faWithROPEParam.selfOutLinearParam.isBias = true;
    faWithROPEParam.selfOutLinearParam.isQuant = param.quantmodel;
    // mixedQKV linear param
    faWithROPEParam.mixdQkvLinearParam.isBias = true;
    faWithROPEParam.mixdQkvLinearParam.isQuant = param.quantmodel;
    if (param.quantmodel) {
        // mixed QKV linear
        faWithROPEParam.mixdQkvLinearParam.transposeB = true;
        faWithROPEParam.mixdQkvLinearParam.isSparse = param.isSparse;
        // self out linear
        faWithROPEParam.selfOutLinearParam.transposeB = true;
        faWithROPEParam.selfOutLinearParam.isSparse = param.isSparse;
        faWithROPEParam.selfOutLinearParam.quantParam.quantType = atb::infer::QUANT_INT8;
        faWithROPEParam.selfOutLinearParam.quantParam.isQuantOp = true;
        faWithROPEParam.selfOutLinearParam.quantParam.elewiseType =
                                                                atb::infer::ElewiseParam::ElewiseType::ELEWISE_QUANT;
        faWithROPEParam.selfOutLinearParam.quantParam.inputScale = param.denseInputScale;
        faWithROPEParam.selfOutLinearParam.quantParam.inputOffset = param.denseInputOffset;
    }
    flashAttentionWithRoPEObj.FlashAttentionWithPositionEmbeddingLayer(faWithROPEParam,
                                                                       &flashAttentionWithROPENode.operation);
    flashAttentionWithROPENode.inTensorIds = {INTERMEDIATE_INPUTNORMOUT,
                                              IN_QKVMIXDWEIGHT,
                                              IN_SELFOUTLINEARWEIGHT,
                                              IN_QKVMIXDBIAS,
                                              IN_SELFOUTLINEARBIAS,
                                              IN_QKVMIXDDEQSCALE,
                                              IN_SELFOUTLINEARDEQSCALE,
                                              IN_QKVMIXDWEIGHT_INDEX, // 稀疏独有权重
                                              IN_QKVMIXDOFFSETX,
                                              IN_QKVMIXDWEIGHT_COMPRESSINFO,
                                              IN_SELFOUTLINEARWEIGHT_INDEX,
                                              IN_SELFOUTLINEAROFFSETX,
                                              IN_SELFOUTLINEARWEIGHT_COMPRESSINFO,
                                              IN_COS_TABLE,
                                              IN_SIN_TABLE,
                                              IN_SEQLEN,
                                              IN_PASTKEY,
                                              IN_PASTVALUE,
                                              IN_ATTENTION_MASK,
                                              IN_TOKEN_OFFSET,
                                              IN_LAYER_ID,
                                              IN_PLACE_HOLDER};
    flashAttentionWithROPENode.outTensorIds = {INTERMEDIATE_ATTENTION_OUTPUT};

    atb::infer::ElewiseParam mulsParam;
    mulsParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MULS;
    mulsParam.mulsParam.varAttr = param.residualAddScale;
    CREATE_OPERATION(mulsParam, &selfResidualMulsNode.operation);
    selfResidualMulsNode.inTensorIds = {INTERMEDIATE_INPUTNORMOUT};
    selfResidualMulsNode.outTensorIds = {INTERMEDIATE_SELFRESIDUALMULSOUT};
    
    atb::infer::ElewiseParam AddParam;
    AddParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CREATE_OPERATION(AddParam, &selfResidualAddNode.operation);
    selfResidualAddNode.inTensorIds = {INTERMEDIATE_SELFRESIDUALMULSOUT, INTERMEDIATE_ATTENTION_OUTPUT};
    selfResidualAddNode.outTensorIds = {INTERMEDIATE_SELFRESIDUALADDOUT};
    
    atb::infer::LayerNormParam selfNormParam;
    selfNormParam.layerType = atb::infer::LayerNormParam::LayerNormType::LAYER_NORM_NORM;
    selfNormParam.normParam.beginNormAxis = 2;
    selfNormParam.normParam.beginParamsAxis = 1;
    if (param.quantmodel) {
        selfNormParam.normParam.quantInputScale = param.selfLnInputScale;
        selfNormParam.normParam.quantInputOffset = param.selfLnInputOffset;
        selfNormParam.normParam.quantType = atb::infer::QUANT_INT8;
        CREATE_OPERATION(selfNormParam, &selfNormNode.operation);
        selfNormNode.inTensorIds = {INTERMEDIATE_SELFRESIDUALADDOUT, IN_SELFOUTNORMWEIGHT, IN_BETA};
    } else {
        CREATE_OPERATION(selfNormParam, &selfNormNode.operation);
        selfNormNode.inTensorIds = {INTERMEDIATE_SELFRESIDUALADDOUT, IN_SELFOUTNORMWEIGHT, IN_SELFOUTNORMBIAS};
    }
    selfNormNode.outTensorIds = {INTERMEDIATE_SELFNORMOUT};

    atb_speed::common::MlpGateParamV2 mlpParam;
    mlpParam.isPack=false;
    mlpParam.noGate=true;
    mlpParam.isBias=true;
    mlpParam.isQuant=param.quantmodel;
    mlpParam.commDownParam.rank = param.rank;
    mlpParam.commDownParam.rankSize = param.rankSize;
    mlpParam.commDownParam.backend = param.backend;
    mlpParam.activationType = atb::infer::ActivationType::ACTIVATION_GELU;
    if (param.quantmodel) {
        mlpParam.quantUpParam.quantType = atb::infer::QUANT_INT8;
        mlpParam.quantUpParam.isQuantOp = false;
        mlpParam.transposeB=true;
        mlpParam.isSparse = param.isSparse;
        mlpParam.quantDownParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_QUANT;
        mlpParam.quantDownParam.inputScale = param.ffnOutInputScale;
        mlpParam.quantDownParam.inputOffset = param.ffnOutInputOffset;
    }
    atb_speed::common::MlpGateLayerV2(mlpParam, &mlpNode.operation);
    mlpNode.inTensorIds = {INTERMEDIATE_SELFNORMOUT, IN_MLPLINEARWEIGHTUP, IN_PLACE_HOLDER, IN_MLPLINEARWEIGHTDOWN,
                           IN_MLPLINEARDEQSCALETUP, IN_PLACE_HOLDER, IN_MLPLINEARDEQSCALEDOWN, IN_MLPLINEARBIASUP,
                           IN_PLACE_HOLDER, IN_MLPLINEARBIASDOWN, IN_MLPLINEARWEIGHT_INDEX, IN_PLACE_HOLDER,
                           IN_MLPLINEARWEIGHTDOWN_INDEX, IN_MLPLINEAROFFSETX, IN_PLACE_HOLDER, IN_MLPLINEAROFFSETXDOWN,
                           IN_MLPLINEARWEIGHT_COMPRESSINFO, IN_PLACE_HOLDER, IN_MLPLINEARWEIGHTDOWN_COMPRESSINFO};
    mlpNode.outTensorIds = {INTERMEDIATE_MLPOUT};

    atb::infer::ElewiseParam muls2Param;
    muls2Param.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MULS;
    muls2Param.mulsParam.varAttr = param.residualAddScale;
    CREATE_OPERATION(muls2Param, &mlpResidualMulsNode.operation);
    mlpResidualMulsNode.inTensorIds = {INTERMEDIATE_SELFNORMOUT};
    mlpResidualMulsNode.outTensorIds = {INTERMEDIATE_MLPRESIDUALMULSOUT};
    
    atb::infer::ElewiseParam Add2Param;
    Add2Param.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CREATE_OPERATION(Add2Param, &mlpResidualAddNode.operation);
    mlpResidualAddNode.inTensorIds = {INTERMEDIATE_MLPRESIDUALMULSOUT, INTERMEDIATE_MLPOUT};
    mlpResidualAddNode.outTensorIds = {OUT_GLMLAYEROUT};

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        const atb::TensorDesc &hiddenStateTensorDesc = inTensorDescs.at(IN_HIDDENSTATES);
        const size_t glmLayerOutID = 0;

        outTensorDescs.at(glmLayerOutID) = hiddenStateTensorDesc;

        return atb::NO_ERROR;
    };
    CREATE_OPERATION(opGraph, operation);

    return atb::NO_ERROR;
}
} // namespace chatglm_6b
} // namespace atb_speed