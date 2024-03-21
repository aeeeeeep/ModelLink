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
#include "fusion_parallel_layer.h"
#include "atb_speed/log.h"
#include "layers/quant_parallel_layer.h"
#include "glm/130b/operation/fusion_mlp.h"
#include "glm/130b/operation/position_embedding.h"
#include "layers/plugin_op/w8a16_bias_operation.h"
#include "layers/plugin_op/w8a16_operation.h"

namespace atb_speed {
namespace glm130b {
static const uint64_t IN_TENSOR_COUNT = 29;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 12;
static const uint64_t NODE_COUNT = 11;
static const uint64_t DEFAULT_BEGIN_NORM_AXIS = 2;

enum class CoderTypes {
    UNDEFINED_TYPE,
    ENCODER_TYPE,
    DECODER_TYPE
};

void SqueezeLinearReshapeFunc(const atb::Dims &oldShape, atb::Dims &newShape)
{
    newShape.dimNum = 2;
    newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
    newShape.dims[1] = oldShape.dims[2] * oldShape.dims[3];
}

atb::Status CreateFusionParallelLayer(const FusionParallelLayerParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    atb::Node &inputNormNode = opGraph.nodes.at(nodeId++);
    atb::Node &mixdQkvLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &positionEmbeddingNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfAttentionKvCacheNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfOutLinearParallelNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfResidualNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfAddNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfNormNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpResidualNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpAddNode = opGraph.nodes.at(nodeId++);

    std::shared_ptr<int64_t> bsPtr = std::make_shared<int64_t>(0);

    atb::infer::LayerNormParam inputNormParam;
    inputNormParam.layerType = atb::infer::LayerNormParam::LAYER_NORM_NORM;
    inputNormParam.normParam.epsilon = param.layerNormEps;
    inputNormParam.normParam.beginNormAxis = DEFAULT_BEGIN_NORM_AXIS;
    inputNormParam.normParam.beginParamsAxis = 1;
    CreateOperation(inputNormParam, &inputNormNode.operation);
    inputNormNode.inTensorIds = {IN_HIDDENSTATES_ID, IN_NORMWEIGHT_ID,
                                 IN_NORMBIAS_ID};                // IN_HIDDENSTATES_ID: [bs, seq_len, hidden_size]
    inputNormNode.outTensorIds = {INTERMEDIATE_INPUTNORMOUT_ID}; // [bs, seq_len, hidden_size]

    mixdQkvLinearNode.operation = new atb_speed::common::W8A16BiasOperation("mixdQkvLinearNode");
    mixdQkvLinearNode.inTensorIds = {INTERMEDIATE_INPUTNORMOUT_ID, IN_QKVMIXEDWEIGHT_ID, IN_ANTIQUQNT_SCALE_QKV,
                                    IN_ANTIQUQNT_OFFSET_QKV, IN_QKVMIXEDBIAS_ID};
    mixdQkvLinearNode.outTensorIds = {INTERMEDIATE_MIXEDLINEAROUTQKV_ID}; // [bs, seq_len, 3 * hidden_size / world_size]
    mixdQkvLinearNode.inTensorReshapeFuncs.resize(mixdQkvLinearNode.inTensorIds.size());
    mixdQkvLinearNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        ATB_LOG(INFO) << "bs PTR start";
        *bsPtr = oldShape.dims[0];
        ATB_LOG(INFO) << "bs PTR end";
        newShape.dimNum = 2; // dimNum is 2
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
        newShape.dims[1] = oldShape.dims[2];
    };

    PositionEmbeddingParam positionEmbeddingParam;
    positionEmbeddingParam.headNum = param.headNum / param.rankSize;
    CreatePositionEmbedding(positionEmbeddingParam, &positionEmbeddingNode.operation);
    positionEmbeddingNode.inTensorIds = {INTERMEDIATE_MIXEDLINEAROUTQKV_ID, IN_COS_ID, IN_SIN_ID, IN_SEQLEN_ID};
    positionEmbeddingNode.outTensorIds = {INTERMEDIATE_POSITIONEMBEDQ_ID, INTERMEDIATE_POSITIONEMBEDK_ID,
                                          INTERMEDIATE_VALUE_ID}; // [bs, seq_len, headNum / rankSize, head_size]
    positionEmbeddingNode.inTensorReshapeFuncs.resize(positionEmbeddingNode.inTensorIds.size());
    positionEmbeddingNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 3; // dimNum is 3
        ATB_LOG(INFO) << "After bs PTR start";
        newShape.dims[0] = *bsPtr;
        ATB_LOG(INFO) << "After bs PTR end";
        newShape.dims[1] = oldShape.dims[0] / *bsPtr;
        newShape.dims[2] = oldShape.dims[1];
    };

    atb::infer::SelfAttentionParam selfAttentionParam;
    selfAttentionParam.headNum = param.headNum / param.rankSize;
    selfAttentionParam.qScale = param.qScale;
    selfAttentionParam.qkScale = param.qkScale;

    CreateOperation(selfAttentionParam, &selfAttentionKvCacheNode.operation);
    selfAttentionKvCacheNode.inTensorIds = {INTERMEDIATE_POSITIONEMBEDQ_ID,
                                            INTERMEDIATE_POSITIONEMBEDK_ID,
                                            INTERMEDIATE_VALUE_ID,
                                            IN_CACHEK_ID,
                                            IN_CACHEV_ID,
                                            IN_ATTENTIONMASK_ID,
                                            IN_TOKENOFFSET_ID,
                                            IN_SEQLEN_ID,
                                            IN_LAYERID_ID};
    selfAttentionKvCacheNode.outTensorIds = {INTERMEDIATE_SELFOUT_ID}; // [bs, seq_len, hidden_size / 8]
    selfAttentionKvCacheNode.inTensorReshapeFuncs.resize(selfAttentionKvCacheNode.inTensorIds.size());
    selfAttentionKvCacheNode.inTensorReshapeFuncs.at(0) = &SqueezeLinearReshapeFunc;
    selfAttentionKvCacheNode.inTensorReshapeFuncs.at(1) = &SqueezeLinearReshapeFunc;
    selfAttentionKvCacheNode.inTensorReshapeFuncs.at(2) = &SqueezeLinearReshapeFunc;

    atb_speed::common::QuantParallelParam selfOutLinearParallelParam;
    selfOutLinearParallelParam.rank = param.rank;
    selfOutLinearParallelParam.rankSize = param.rankSize;
    selfOutLinearParallelParam.isBias = true;
    selfOutLinearParallelParam.backend = param.backend;
    atb_speed::common::QuantRowParallelLinear(selfOutLinearParallelParam, &selfOutLinearParallelNode.operation);
    selfOutLinearParallelNode.inTensorIds = {INTERMEDIATE_SELFOUT_ID, IN_SELFOUTLINEARWEIGHT_ID,
                                             IN_ANTIQUQNT_SCALE_OUTLINEAR, IN_ANTIQUQNT_OFFSET_OUTLINEAR,
                                             IN_SELFOUTLINEARBIAS_ID};
    selfOutLinearParallelNode.outTensorIds = {INTERMEDIATE_SELFLINEAROUT_ID};
    selfOutLinearParallelNode.inTensorReshapeFuncs.resize(selfOutLinearParallelNode.inTensorIds.size());
    selfOutLinearParallelNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 3; // dimNum is 3
        ATB_LOG(INFO) << "After bs PTR start";
        newShape.dims[0] = *bsPtr;
        ATB_LOG(INFO) << "After bs PTR end";
        newShape.dims[1] = oldShape.dims[0] / *bsPtr;
        newShape.dims[2] = oldShape.dims[1];
    };

    atb::infer::ElewiseParam selfResidualParam;
    selfResidualParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MULS;
    selfResidualParam.mulsParam.varAttr = param.residualAddScale;
    CreateOperation(selfResidualParam, &selfResidualNode.operation);
    selfResidualNode.inTensorIds = {INTERMEDIATE_INPUTNORMOUT_ID};
    selfResidualNode.outTensorIds = {INTERMEDIATE_SELFRESIDUALOUT_ID}; // [bs, seq_len, hidden_size]

    atb::infer::ElewiseParam selfAddParam;
    selfAddParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CreateOperation(selfAddParam, &selfAddNode.operation);
    selfAddNode.inTensorIds = {INTERMEDIATE_SELFRESIDUALOUT_ID, INTERMEDIATE_SELFLINEAROUT_ID};
    selfAddNode.outTensorIds = {INTERMEDIATE_SELFADDOUT_ID}; // [bs, seq_len, hidden_size]

    atb::infer::LayerNormParam selfNormParam;
    selfNormParam.layerType = atb::infer::LayerNormParam::LAYER_NORM_NORM;
    selfNormParam.normParam.epsilon = param.layerNormEps;
    selfNormParam.normParam.beginNormAxis = DEFAULT_BEGIN_NORM_AXIS;
    selfNormParam.normParam.beginParamsAxis = 1;
    CreateOperation(selfNormParam, &selfNormNode.operation);
    selfNormNode.inTensorIds = {INTERMEDIATE_SELFADDOUT_ID, IN_SELFOUTNORMWEIGHT_ID, IN_SELFOUTNORMBIAS_ID};
    selfNormNode.outTensorIds = {INTERMEDIATE_SELFNORMOUT_ID}; // [bs, seq_len, hidden_size]

    atb_speed::glm130b::FusionMlpGlmParam mlpParam;
    mlpParam.rank = param.rank;
    mlpParam.rankSize = param.rankSize;
    mlpParam.activationType = atb::infer::ActivationType::ACTIVATION_SWISH;
    mlpParam.transposeB = false;
    mlpParam.isBias = true;
    mlpParam.backend = param.backend;
    atb_speed::glm130b::FusionMlpGlm(mlpParam, &mlpNode.operation);
    mlpNode.inTensorIds = {INTERMEDIATE_SELFNORMOUT_ID,
                           IN_MLPLINEARWEIGHT_ID, IN_MLPOUTLINEARWEIGHT_ID,
                           IN_MLPLINEARBIAS_ID, IN_MLPOUTLINEARBIAS_ID,
                           IN_ANTIQUQNT_SCALE_MLPUP, IN_ANTIQUQNT_OFFSET_MLPUP,
                           IN_ANTIQUQNT_SCALE_MLPDOWN, IN_ANTIQUQNT_OFFSET_MLPDOWN,
                           };
    mlpNode.outTensorIds = {INTERMEDIATE_MLPOUT_ID};

    atb::infer::ElewiseParam mlpResidualParam;
    mlpResidualParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MULS;
    mlpResidualParam.mulsParam.varAttr = param.residualAddScale;
    CreateOperation(mlpResidualParam, &mlpResidualNode.operation);
    mlpResidualNode.inTensorIds = {INTERMEDIATE_SELFNORMOUT_ID};
    mlpResidualNode.outTensorIds = {INTERMEDIATE_MLPRESIDUALOUT_ID};

    atb::infer::ElewiseParam mlpAddParam;
    mlpAddParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CreateOperation(mlpAddParam, &mlpAddNode.operation);
    mlpAddNode.inTensorIds = {INTERMEDIATE_MLPRESIDUALOUT_ID, INTERMEDIATE_MLPOUT_ID};
    mlpAddNode.outTensorIds = {OUT_LAYEROUT_ID};

    opGraph.inferShapeFunc = [&](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
    };

    atb::CreateOperation(opGraph, operation);
    return atb::NO_ERROR;
}

FusionParallelLayer::FusionParallelLayer() {}

FusionParallelLayer::~FusionParallelLayer() {}

void FusionParallelLayer::ParseParam(const nlohmann::json &paramJson)
{
    tokenOffset_.clear();
    for (auto item : paramJson["tokenOffset"]) {
        tokenOffset_.push_back(item.get<int>());
    }

    seqLen_.clear();
    for (auto item : paramJson["seqLen"]) {
        seqLen_.push_back(item.get<int>());
    }
}

void FusionParallelLayer::BindTensor(atb::VariantPack &variantPack)
{
    variantPack.inTensors.at(IN_TOKENOFFSET_ID).hostData = tokenOffset_.data();
    variantPack.inTensors.at(IN_SEQLEN_ID).hostData = seqLen_.data();
}
} // namespace glm130b
} // namespace atb_speed