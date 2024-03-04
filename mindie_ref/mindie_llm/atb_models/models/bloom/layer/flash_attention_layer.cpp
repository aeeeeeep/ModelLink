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
#include "layers/mlp_gate_v2.h"
#include "layers/parallel_layer_v2.h"
#include "models/bloom/layer/flash_attention_layer.h"


namespace atb_speed {
namespace bloom_7b {
enum Bloom7BLayerInTensorId : int {
    IN_NORM_WEIGHT = 0,
    IN_NORM_BIAS,

    IN_QKVMIXED_WEIGHT,
    IN_QKVMIXED_BIAS,
    IN_QKVMIXED_DEQSCALE,

    IN_SELFOUTLINEAR_WEIGHT,
    IN_SELFOUTLINEAR_BIAS,
    IN_SELFOUTLINEAR_DEQSCALE,

    IN_SELFOUTNORM_WEIGHT,
    IN_SELFOUTNORM_BIAS,

    IN_HTO4H_WEIGHT,
    IN_HTO4H_BIAS,
    IN_HTO4H_DEQSCALE,

    IN_4HTOH_WEIGHT,
    IN_4HTOH_BIAS,
    IN_4HTOH_DEQSCALE,

    IN_HIDDEN_STATES,
    IN_ATTENTION_MASK,
    IN_CACHED_K,
    IN_CACHED_V,
    IN_TOKENOFFSET,
    IN_SEQLEN,
    IN_PLACE_HOLDER,
    IN_LAYERID,

    IN_TENSOR_MAX
};

enum Bloom7BLayerOutTensorId : int {
    OUT_LAYEROUT = IN_TENSOR_MAX,
    OUT_PLACEHOLDER_1,
    OUT_PLACEHOLDER_2,
    OUT_TENSOR_MAX
};

enum Bloom7BLayerIntermidateTensorId : int {
    INTERMIDATE_INPUTNORM_OUT = OUT_TENSOR_MAX,
    INTERMIDATE_MIXEDLINEAROUTQKV,
    INTERMIDATE_QUERY,
    INTERMIDATE_KEY,
    INTERMIDATE_VALUE,

    INTERMIDATE_SELFOUT,
    INTERMIDATE_SELFLINEAROUT,
    INTERMIDATE_SELFADDOUT,
    INTERMIDATE_SELFNORMOUT,
    INTERMIDATE_MLPOUT,

    INTERMIDATE_TENSOR_MAX,
};

static const uint64_t IN_TENSOR_COUNT = IN_TENSOR_MAX;
static const uint64_t OUT_TENSOR_COUNT = OUT_TENSOR_MAX - IN_TENSOR_MAX;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = INTERMIDATE_TENSOR_MAX - OUT_TENSOR_MAX;
static const uint64_t NODE_COUNT = 9;
static const uint64_t HIDDEN_STATES_DIM = 3;

void SqueezeThirdDim(const atb::Dims &oldShape, atb::Dims &newShape)
{
    newShape.dimNum = oldShape.dimNum - 1;
    std::copy(std::begin(oldShape.dims), std::end(oldShape.dims), std::begin(newShape.dims));
    newShape.dims[newShape.dimNum - 1] = oldShape.dims[oldShape.dimNum - 1]; // squeeze second last dim
}

void MulFirstSecondDim(const atb::Dims &oldShape, atb::Dims &newShape)
{
    newShape.dimNum = oldShape.dimNum - 1;
    std::copy(std::begin(oldShape.dims) + 1, std::end(oldShape.dims), std::begin(newShape.dims));
    newShape.dims[0] = oldShape.dims[0] * newShape.dims[0]; // mul first and second dim
}

atb::Status CommomLayer(const Bloom7bCommonLayerParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(param.quantmodel ? NODE_COUNT : NODE_COUNT + 2);
    opGraph.name = "Bloom7bCommonLayer";

    size_t nodeId = 0;
    atb::Node &inputNormNode = opGraph.nodes.at(nodeId++);
    atb::Node &mixdQkvLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &splitNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfAttentionFusionNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfOutLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfOutAddNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfNormNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpResidualAddNode = opGraph.nodes.at(nodeId++);

    atb::infer::LayerNormParam layerNormQuantParam;
    layerNormQuantParam.layerType = atb::infer::LayerNormParam::LAYER_NORM_NORM;
    const int32_t beginParamsAxis = 2;
    layerNormQuantParam.normParam.epsilon = param.layerNormEps;
    layerNormQuantParam.normParam.beginNormAxis = beginParamsAxis;
    layerNormQuantParam.normParam.beginParamsAxis = 1;
    if (param.quantmodel) {
        layerNormQuantParam.normParam.quantType = atb::infer::QUANT_INT8;
        layerNormQuantParam.normParam.quantInputScale = param.qkvInputScale;
        layerNormQuantParam.normParam.quantInputOffset = param.qkvInputOffset;
    }
    CREATE_OPERATION(layerNormQuantParam, &inputNormNode.operation);
    inputNormNode.inTensorIds = {IN_HIDDEN_STATES, IN_NORM_WEIGHT, IN_NORM_BIAS};
    if (param.quantmodel) {
        inputNormNode.outTensorIds = {OUT_PLACEHOLDER_1, INTERMIDATE_INPUTNORM_OUT};
    } else {
        inputNormNode.outTensorIds = {INTERMIDATE_INPUTNORM_OUT};
    }

    if (param.quantmodel) {
        atb::infer::LinearParam mixdQkvLinearParam;
        mixdQkvLinearParam.linearType = atb::infer::LinearType::LINEAR_INT8INT8_INT32_FP16;
        CREATE_OPERATION(mixdQkvLinearParam, &mixdQkvLinearNode.operation);
        mixdQkvLinearNode.inTensorIds = {
            INTERMIDATE_INPUTNORM_OUT, IN_QKVMIXED_WEIGHT, IN_QKVMIXED_BIAS, IN_QKVMIXED_DEQSCALE};
    } else {
        atb::infer::LinearParam mixdQkvLinearParam;
        CREATE_OPERATION(mixdQkvLinearParam, &mixdQkvLinearNode.operation);
        mixdQkvLinearNode.inTensorIds = {
            INTERMIDATE_INPUTNORM_OUT, IN_QKVMIXED_WEIGHT, IN_QKVMIXED_BIAS};
    }
    mixdQkvLinearNode.outTensorIds = {INTERMIDATE_MIXEDLINEAROUTQKV};

    atb::infer::SplitParam splitParam = {3, 3};
    CREATE_OPERATION(splitParam, &splitNode.operation);
    splitNode.inTensorIds = {INTERMIDATE_MIXEDLINEAROUTQKV};
    splitNode.outTensorIds = {INTERMIDATE_QUERY, INTERMIDATE_KEY, INTERMIDATE_VALUE};
    splitNode.inTensorReshapeFuncs.resize(splitNode.inTensorIds.size());
    splitNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        size_t dim = 0;
        newShape.dims[dim++] = oldShape.dims[0]; // batch
        newShape.dims[dim++] = oldShape.dims[1]; // seq_len
        newShape.dims[dim++] = param.headNum;    // head_num
        newShape.dims[dim++] = 3;                // 3 -> q, k, v
        newShape.dims[dim++] = param.dk;         // dk
        newShape.dimNum = dim;                   // [batch, seq_len, head_num, 3, dk]
    };

    atb::infer::SelfAttentionParam selfAttentionParam;
    selfAttentionParam.headNum = param.headNum;
    selfAttentionParam.qScale = 1.0f / std::sqrt(param.dk);
    selfAttentionParam.qkScale = 1.0f;
    selfAttentionParam.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_ALIBI;
    CREATE_OPERATION(selfAttentionParam, &selfAttentionFusionNode.operation);
    selfAttentionFusionNode.inTensorIds = {
        INTERMIDATE_QUERY, INTERMIDATE_KEY, INTERMIDATE_VALUE, IN_CACHED_K,
        IN_CACHED_V, IN_ATTENTION_MASK, IN_TOKENOFFSET, IN_SEQLEN, IN_LAYERID
        };
    selfAttentionFusionNode.outTensorIds = {INTERMIDATE_SELFOUT};
    selfAttentionFusionNode.inTensorReshapeFuncs.resize(selfAttentionFusionNode.inTensorIds.size());
    selfAttentionFusionNode.inTensorReshapeFuncs[0] = &SqueezeThirdDim;
    selfAttentionFusionNode.inTensorReshapeFuncs[1] = &SqueezeThirdDim;
    selfAttentionFusionNode.inTensorReshapeFuncs[2] = &SqueezeThirdDim;

    if (param.quantmodel) {
        atb_speed::common::ParallelParamV2 selfOutLinearParam;
        selfOutLinearParam.commParam.rank = param.rank;
        selfOutLinearParam.commParam.rankSize = param.rankSize;
        selfOutLinearParam.isBias = true;
        selfOutLinearParam.isQuant = true;
        selfOutLinearParam.transposeB = true;
        selfOutLinearParam.quantParam.quantType = atb::infer::QUANT_INT8;
        selfOutLinearParam.quantParam.isQuantOp = true;
        selfOutLinearParam.quantParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_QUANT;
        selfOutLinearParam.quantParam.inputScale = param.denseInputScale;
        selfOutLinearParam.quantParam.inputOffset = param.denseInputOffset;
        atb_speed::common::RowParallelLinearV2(selfOutLinearParam, &selfOutLinearNode.operation);
        selfOutLinearNode.inTensorIds = {INTERMIDATE_SELFOUT, IN_SELFOUTLINEAR_WEIGHT,
                                        IN_SELFOUTLINEAR_BIAS, IN_SELFOUTLINEAR_DEQSCALE,
                                        IN_PLACE_HOLDER, IN_PLACE_HOLDER, IN_PLACE_HOLDER};
        selfOutLinearNode.outTensorIds = {INTERMIDATE_SELFLINEAROUT};
    } else {
        atb_speed::common::ParallelParamV2 selfOutLinearParam;
        selfOutLinearParam.commParam.rank = param.rank;
        selfOutLinearParam.commParam.rankSize = param.rankSize;
        selfOutLinearParam.isBias = true;
        selfOutLinearParam.transposeB = true;
        atb_speed::common::RowParallelLinearV2(selfOutLinearParam, &selfOutLinearNode.operation);
        selfOutLinearNode.inTensorIds = {INTERMIDATE_SELFOUT, IN_SELFOUTLINEAR_WEIGHT,
                                        IN_SELFOUTLINEAR_BIAS, IN_PLACE_HOLDER, IN_PLACE_HOLDER,
                                        IN_PLACE_HOLDER, IN_PLACE_HOLDER};
        selfOutLinearNode.outTensorIds = {INTERMIDATE_SELFLINEAROUT};
        ATB_LOG(INFO) << "RowParallelLinearV2 " << selfOutLinearParam.commParam.rankSize<< "-";
        ATB_LOG(INFO) << selfOutLinearParam.transposeB;
    }

    atb::infer::ElewiseParam selfAddParam;
    selfAddParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CREATE_OPERATION(selfAddParam, &selfOutAddNode.operation);
    selfOutAddNode.inTensorIds = {INTERMIDATE_SELFLINEAROUT, IN_HIDDEN_STATES};
    selfOutAddNode.outTensorIds = {INTERMIDATE_SELFADDOUT};

    if (param.quantmodel) {
        atb::infer::LayerNormParam selfNormParam;
        selfNormParam.layerType = atb::infer::LayerNormParam::LAYER_NORM_NORM;
        selfNormParam.normParam.quantInputScale = param.selfLnInputScale;
        selfNormParam.normParam.quantInputOffset = param.selfLnInputOffset;
        selfNormParam.normParam.quantType = atb::infer::QUANT_INT8;
        selfNormParam.normParam.epsilon = param.layerNormEps;
        selfNormParam.normParam.beginNormAxis = beginParamsAxis;
        selfNormParam.normParam.beginParamsAxis = 1;
        CREATE_OPERATION(selfNormParam, &selfNormNode.operation);
        selfNormNode.inTensorIds = {INTERMIDATE_SELFADDOUT, IN_SELFOUTNORM_WEIGHT, IN_SELFOUTNORM_BIAS};
        selfNormNode.outTensorIds = {OUT_PLACEHOLDER_2, INTERMIDATE_SELFNORMOUT};

        atb_speed::common::MlpGateParamV2 mlpParam;
        mlpParam.isBias=true;
        mlpParam.isPack=false;
        mlpParam.isQuant=true;
        mlpParam.noGate = true;
        mlpParam.transposeB=true;
        mlpParam.commDownParam.rank = param.rank;
        mlpParam.commDownParam.rankSize = param.rankSize;
        mlpParam.quantUpParam.quantType = atb::infer::QUANT_INT8;
        mlpParam.quantUpParam.isQuantOp = false;
        mlpParam.activationType = atb::infer::ActivationType::ACTIVATION_GELU;
        mlpParam.quantDownParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_QUANT;
        mlpParam.quantDownParam.inputScale = param.ffnOutInputScale;
        mlpParam.quantDownParam.inputOffset = param.ffnOutInputOffset;
        atb_speed::common::MlpGateLayerV2(mlpParam, &mlpNode.operation);
        mlpNode.inTensorIds = {INTERMIDATE_SELFNORMOUT, IN_HTO4H_WEIGHT, IN_PLACE_HOLDER, IN_4HTOH_WEIGHT,
                            IN_HTO4H_DEQSCALE, IN_PLACE_HOLDER, IN_4HTOH_DEQSCALE, IN_HTO4H_BIAS,
                            IN_PLACE_HOLDER, IN_4HTOH_BIAS, IN_PLACE_HOLDER, IN_PLACE_HOLDER, IN_PLACE_HOLDER,
                            IN_PLACE_HOLDER, IN_PLACE_HOLDER, IN_PLACE_HOLDER, IN_PLACE_HOLDER, IN_PLACE_HOLDER,
                            IN_PLACE_HOLDER};
        mlpNode.outTensorIds = {INTERMIDATE_MLPOUT};
    } else {
        atb::infer::LayerNormParam selfNormParam;
        selfNormParam.layerType = atb::infer::LayerNormParam::LAYER_NORM_NORM;
        selfNormParam.normParam.epsilon = param.layerNormEps;
        selfNormParam.normParam.beginNormAxis = beginParamsAxis;
        selfNormParam.normParam.beginParamsAxis = 1;
        CREATE_OPERATION(selfNormParam, &selfNormNode.operation);
        selfNormNode.inTensorIds = {INTERMIDATE_SELFADDOUT, IN_SELFOUTNORM_WEIGHT, IN_SELFOUTNORM_BIAS};
        selfNormNode.outTensorIds = {INTERMIDATE_SELFNORMOUT};

        atb_speed::common::MlpGateParamV2 mlpParam;
        mlpParam.commDownParam.rank = param.rank;
        mlpParam.commDownParam.rankSize = param.rankSize;
        mlpParam.activationType = atb::infer::ActivationType::ACTIVATION_GELU;
        mlpParam.transposeB = true;
        mlpParam.isBias = true;
        mlpParam.noGate = true;
        mlpParam.isPack = false;
        atb_speed::common::MlpGateLayerV2(mlpParam, &mlpNode.operation);
        mlpNode.inTensorIds = {INTERMIDATE_SELFNORMOUT, IN_HTO4H_WEIGHT, IN_PLACE_HOLDER, IN_4HTOH_WEIGHT,
                            IN_PLACE_HOLDER, IN_PLACE_HOLDER, IN_PLACE_HOLDER, IN_HTO4H_BIAS, IN_PLACE_HOLDER,
                            IN_4HTOH_BIAS, IN_PLACE_HOLDER, IN_PLACE_HOLDER, IN_PLACE_HOLDER, IN_PLACE_HOLDER,
                            IN_PLACE_HOLDER, IN_PLACE_HOLDER, IN_PLACE_HOLDER, IN_PLACE_HOLDER, IN_PLACE_HOLDER};
        mlpNode.outTensorIds = {INTERMIDATE_MLPOUT};
    }

    atb::infer::ElewiseParam mlpResidualAddParam;
    mlpResidualAddParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CREATE_OPERATION(mlpResidualAddParam, &mlpResidualAddNode.operation);
    mlpResidualAddNode.inTensorIds = {INTERMIDATE_MLPOUT, INTERMIDATE_SELFADDOUT};
    mlpResidualAddNode.outTensorIds = {OUT_LAYEROUT};

    if (!param.quantmodel) {
        atb::Node &placeHolderNode1 = opGraph.nodes.at(nodeId++);
        atb::Node &placeHolderNode2 = opGraph.nodes.at(nodeId++);

        atb::infer::ElewiseParam fillPlaceHolderParam;
        fillPlaceHolderParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MULS;
        fillPlaceHolderParam.mulsParam.varAttr = 1.0;

        CREATE_OPERATION(fillPlaceHolderParam, &placeHolderNode1.operation);
        placeHolderNode1.inTensorIds = {OUT_LAYEROUT};
        placeHolderNode1.outTensorIds = {OUT_PLACEHOLDER_1};

        CREATE_OPERATION(fillPlaceHolderParam, &placeHolderNode2.operation);
        placeHolderNode2.inTensorIds = {OUT_LAYEROUT};
        placeHolderNode2.outTensorIds = {OUT_PLACEHOLDER_2};
    }

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        size_t dim = 0;
        outTensorDescs.at(dim++) = inTensorDescs.at(IN_HIDDEN_STATES);
        outTensorDescs.at(dim++) = inTensorDescs.at(IN_HIDDEN_STATES);
        outTensorDescs.at(dim++) = inTensorDescs.at(IN_HIDDEN_STATES);
        return atb::NO_ERROR;
    };

    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}

} // namespace bloom_7b
} // namespace atb_speed