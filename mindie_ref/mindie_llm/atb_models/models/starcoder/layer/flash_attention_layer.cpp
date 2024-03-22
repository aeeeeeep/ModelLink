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
#include "layers/mlp_gate_v2.h"
#include "layers/parallel_layer_v2.h"

namespace atb_speed {
namespace star_coder {
enum FlashAttentionLayerTensorId : int {
    IN_HIDDENSTATES = 0,
    IN_LN_1_WEIGTH,
    IN_LN_1_BIAS,
    IN_C_ATTN_WEIGHT,
    IN_C_ATTN_BIAS,
    IN_C_PROJ_WEIGHT,
    IN_C_PROJ_BIAS,
    IN_LN_2_WEIGHT,
    IN_LN_2_BIAS,
    IN_MLP_FC_WEIGHT,
    IN_MLP_FC_BIAS,
    IN_MLP_PROJ_WEIGHT,
    IN_MLP_PROJ_BIAS,
    IN_ATTENTIONMASK,
    IN_CACHEK,
    IN_CACHEV,
    IN_TOKENOFFSET, // 16
    IN_SEQLEN,      // 17
    IN_HOLDER,      // HOLDER
    IN_LAYERID,
    OUT_LAYEROUT,
    INTERMIDATE_INPUTNORMOUT,
    INTERMIDATE_QKV,
    INTERNAL_Q,
    INTERNAL_KV,
    INTERNAL_K,
    INTERNAL_V,
    INTERMIDATE_SELFOUT,
    INTERMIDATE_SELF_LINEAR_OUT,
    INTERMIDATE_SELF_RESIDUAL_ADD_OUT,
    INTERMEDIATE_LN_2_NORM_OUT,
    INTERMIDATE_MLP_OUT,
};

static const uint64_t IN_TENSOR_COUNT = 20;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 11;
static const uint64_t NODE_COUNT = 11;
static const uint64_t LAYER_NORM_AXIS_COUNT = 2;
static const uint64_t ATTENTION_DIM_3 = 3;

atb::Status FlashAttentionLayer(const FlashAttentionLayerParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.name = "StarCoder_FAPA_layer";
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    atb::Node &inputLayerNormNode = opGraph.nodes.at(nodeId++);
    atb::Node &cAttnLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &qPassSliceNode = opGraph.nodes.at(nodeId++);
    atb::Node &kVPassSliceNode = opGraph.nodes.at(nodeId++);
    atb::Node &splitKVNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfAttentionFaNode = opGraph.nodes.at(nodeId++);
    atb::Node &cProjLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfResidualAddNode = opGraph.nodes.at(nodeId++);
    atb::Node &postAttnLayerNormNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpNode = opGraph.nodes.at(nodeId++);
    atb::Node &attnResidualAddNode = opGraph.nodes.at(nodeId++);

    atb::SVector<int64_t> sliceOffsetQ = {0, 0, 0};
    atb::SVector<int64_t> sliceSizeQ = {-1, -1, param.headNum * param.dk};
    atb::SVector<int64_t> sliceOffsetKV = {0, 0, param.headNum * param.dk};
    atb::SVector<int64_t> sliceSizeKV = {-1, -1, 2 * param.dk};

    atb::infer::LayerNormParam layerNormParam;
    layerNormParam.layerType = atb::infer::LayerNormParam::LayerNormType::LAYER_NORM_NORM;
    layerNormParam.normParam.epsilon = param.layerNormEps;
    layerNormParam.normParam.beginNormAxis = LAYER_NORM_AXIS_COUNT;
    layerNormParam.normParam.beginParamsAxis = LAYER_NORM_AXIS_COUNT;
    CreateOperation(layerNormParam, &inputLayerNormNode.operation);
    inputLayerNormNode.inTensorIds = {IN_HIDDENSTATES, IN_LN_1_WEIGTH, IN_LN_1_BIAS};
    inputLayerNormNode.outTensorIds = {INTERMIDATE_INPUTNORMOUT};

    atb::infer::LinearParam linearBiasParam;
    CreateOperation(linearBiasParam, &cAttnLinearNode.operation);
    cAttnLinearNode.inTensorIds = { INTERMIDATE_INPUTNORMOUT, IN_C_ATTN_WEIGHT, IN_C_ATTN_BIAS };
    cAttnLinearNode.outTensorIds = { INTERMIDATE_QKV };

    atb::infer::SliceParam slicePassParam;
    slicePassParam.offsets = sliceOffsetQ;
    slicePassParam.size = sliceSizeQ;
    CreateOperation(slicePassParam, &qPassSliceNode.operation);
    qPassSliceNode.inTensorIds = {INTERMIDATE_QKV};
    qPassSliceNode.outTensorIds = {INTERNAL_Q};

    atb::infer::SliceParam slicePassKVParam;
    slicePassKVParam.offsets = sliceOffsetKV;
    slicePassKVParam.size = sliceSizeKV;
    CreateOperation(slicePassKVParam, &kVPassSliceNode.operation);
    kVPassSliceNode.inTensorIds = {INTERMIDATE_QKV};
    kVPassSliceNode.outTensorIds = {INTERNAL_KV};

    atb::infer::SplitParam splitParam;
    splitParam.splitDim = 2;
    splitParam.splitNum = 2;
    CreateOperation(splitParam, &splitKVNode.operation);
    splitKVNode.inTensorIds = {INTERNAL_KV};
    splitKVNode.outTensorIds = {INTERNAL_K, INTERNAL_V};

    atb::infer::SelfAttentionParam selfAttentionParam;
    selfAttentionParam.headNum = param.headNum;
    selfAttentionParam.qScale = 1.0 / sqrt(param.dk);
    selfAttentionParam.kvHeadNum = param.kvHead;
    selfAttentionParam.maskType = atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_NORM;
    if (param.isEncoder) {
        selfAttentionParam.calcType = atb::infer::SelfAttentionParam::ENCODER;
    } else {
        selfAttentionParam.calcType = atb::infer::SelfAttentionParam::DECODER;
    }
    CreateOperation(selfAttentionParam, &selfAttentionFaNode.operation);
    selfAttentionFaNode.inTensorIds = {INTERNAL_Q,
                                                  INTERNAL_K,
                                                  INTERNAL_V,
                                                  IN_CACHEK,
                                                  IN_CACHEV,
                                                  IN_ATTENTIONMASK,
                                                  IN_TOKENOFFSET,
                                                  IN_SEQLEN,
                                                  IN_LAYERID};
    selfAttentionFaNode.outTensorIds = {INTERMIDATE_SELFOUT};
    selfAttentionFaNode.inTensorReshapeFuncs.resize(selfAttentionFaNode.inTensorIds.size());
    selfAttentionFaNode.inTensorReshapeFuncs.at(0) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 4;
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = oldShape.dims[1];
        newShape.dims[2] = param.headNum;
        newShape.dims[ATTENTION_DIM_3] = oldShape.dims[2] / param.headNum;
    };
    selfAttentionFaNode.inTensorReshapeFuncs.at(1) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 4;
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = oldShape.dims[1];
        newShape.dims[2] = param.kvHead;
        newShape.dims[ATTENTION_DIM_3] = oldShape.dims[2];
    };
    selfAttentionFaNode.inTensorReshapeFuncs.at(2) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 4;
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = oldShape.dims[1];
        newShape.dims[2] = param.kvHead;
        newShape.dims[ATTENTION_DIM_3] = oldShape.dims[2];
    };

    atb_speed::common::ParallelParamV2 cProjLinearParam;
    cProjLinearParam.commParam.rank = param.rank;
    cProjLinearParam.commParam.rankSize = param.rankSize;
    cProjLinearParam.isBias = true;
    atb_speed::common::RowParallelLinearV2(cProjLinearParam, &cProjLinearNode.operation);
    cProjLinearNode.inTensorIds = {INTERMIDATE_SELFOUT, IN_C_PROJ_WEIGHT, IN_C_PROJ_BIAS,
                                    IN_HOLDER, IN_HOLDER, IN_HOLDER, IN_HOLDER};
    cProjLinearNode.outTensorIds = {INTERMIDATE_SELF_LINEAR_OUT};

    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CreateOperation(addParam, &selfResidualAddNode.operation);
    selfResidualAddNode.inTensorIds = { IN_HIDDENSTATES, INTERMIDATE_SELF_LINEAR_OUT };
    selfResidualAddNode.outTensorIds = { INTERMIDATE_SELF_RESIDUAL_ADD_OUT };

    CreateOperation(layerNormParam, &postAttnLayerNormNode.operation);
    postAttnLayerNormNode.inTensorIds = {INTERMIDATE_SELF_RESIDUAL_ADD_OUT, IN_LN_2_WEIGHT, IN_LN_2_BIAS};
    postAttnLayerNormNode.outTensorIds = {INTERMEDIATE_LN_2_NORM_OUT};

    atb_speed::common::MlpGateParamV2 mlpParam;
    mlpParam.commDownParam.rank = param.rank;
    mlpParam.commDownParam.rankSize = param.rankSize;
    mlpParam.activationType = atb::infer::ActivationType::ACTIVATION_GELU;
    mlpParam.transposeB = true;
    mlpParam.isBias = true;
    mlpParam.isPack = false;
    mlpParam.noGate = true;
    atb_speed::common::MlpGateLayerV2(mlpParam, &mlpNode.operation);
    mlpNode.inTensorIds = {INTERMEDIATE_LN_2_NORM_OUT,
                            IN_MLP_FC_WEIGHT, IN_HOLDER, IN_MLP_PROJ_WEIGHT,
                            IN_HOLDER, IN_HOLDER, IN_HOLDER,
                            IN_MLP_FC_BIAS, IN_HOLDER, IN_MLP_PROJ_BIAS,
                            IN_HOLDER, IN_HOLDER, IN_HOLDER,
                            IN_HOLDER, IN_HOLDER, IN_HOLDER,
                            IN_HOLDER, IN_HOLDER, IN_HOLDER};
    mlpNode.outTensorIds = {INTERMIDATE_MLP_OUT};

    CreateOperation(addParam, &attnResidualAddNode.operation);
    attnResidualAddNode.inTensorIds = {INTERMIDATE_SELF_RESIDUAL_ADD_OUT, INTERMIDATE_MLP_OUT};
    attnResidualAddNode.outTensorIds = {OUT_LAYEROUT};

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
        atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
    };

    atb::CreateOperation(opGraph, operation);
    return atb::NO_ERROR;
}

FlashAttentionHostBinder::FlashAttentionHostBinder() {}

FlashAttentionHostBinder::~FlashAttentionHostBinder() {}

void FlashAttentionHostBinder::ParseParam(const nlohmann::json &paramJson)
{
    tokenOffset_.clear();
    for (auto item : paramJson["tokenOffset"]) {
        tokenOffset_.push_back(item.get<int32_t>());
    }

    seqLen_.clear();
    for (auto item : paramJson["seqLen"]) {
        seqLen_.push_back(item.get<int32_t>());
    }
}

void FlashAttentionHostBinder::BindTensor(atb::VariantPack &variantPack)
{
    variantPack.inTensors.at(IN_TOKENOFFSET).hostData = tokenOffset_.data();
    variantPack.inTensors.at(IN_SEQLEN).hostData = seqLen_.data();
}

} // namespace star_coder
} // namespace atb_speed