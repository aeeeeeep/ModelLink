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
#include "layers/mlp_gate.h"
#include "layers/parallel_layer.h"
#include "models/gptneox/20b/operation/position_embedding.h"

namespace atb_speed {
namespace gptneox_20b {
enum LayerTensorId {
    IN_HIDDENSTATES = 0,
    IN_INPUTLAYERNORMWEIGTH,
    IN_INPUTLAYERNORMBIAS,
    IN_POSTATTNLAYERNORMWEIGHT,
    IN_POSTATTNLAYERNORMBIAS,
    IN_QKVWEIGHT,
    IN_QKVBIAS,
    IN_SELFOUTLINEARWEIGHT,
    IN_SELFOUTLINEARBIAS,
    IN_FFNLINEARWEIGHT,
    IN_FFNLINEARBIAS,
    IN_FFNOUTLINEARWEIGHT,
    IN_FFNOUTLINEARBIAS,
    IN_POSITIONIDS,
    IN_COSEMBED,
    IN_SINEMBED,
    IN_ATTENTIONMASK,
    IN_CACHEK,
    IN_CACHEV,
    IN_TOKENOFFSET,
    IN_SEQLEN,
    IN_LAYERID,

    OUT_GPTNEOXLAYEROUT,

    INTERMEDIATE_INPUTLAYERNORMOUT,
    INTERMEDIATE_MIXEDQKVLINEAROUT,
    INTERMEDIATE_QUERYEMBED,
    INTERMEDIATE_KEYEMBED,
    INTERMEDIATE_VALUE,
    INTERMEDIATE_SELFATTNOUT,
    INTERMEDIATE_SELFATTNLINEAROUT,
    INTERMEDIATE_POSTATTNLAYERNORMOUT,
    INTERMEDIATE_FFNLINEAROUT,
    INTERMEDIATE_FFNACTOUT,
    INTERMEDIATE_FFNOUTLINEAROUT,
    INTERMEDIATE_ATTNRESIDUALADDOUT,
};

static const uint64_t IN_TENSOR_COUNT = 22;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 12;
static const uint64_t NODE_COUNT = 11;
static const uint64_t LAYER_NORM_AXIS_NUM = 2;

atb::Status FlashAttentionKvCacheLayer(const LayerParam &param, atb::Operation **operation)
{
    ATB_LOG(INFO) << __func__ << " called, headNum: " << param.headNum;
    atb::GraphParam opGraph;
    opGraph.name = "FlashAttentionKvCacheLayer";
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    atb::Node &inputLayerNormNode = opGraph.nodes.at(nodeId++);
    atb::Node &qkvLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &positionEmbeddingNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfAttentionKvCacheFusionNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfAttnLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &postAttnLayerNormNode = opGraph.nodes.at(nodeId++);
    atb::Node &ffnLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &ffnActNode = opGraph.nodes.at(nodeId++);
    atb::Node &ffnOutLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &attnResidualAddNode = opGraph.nodes.at(nodeId++);
    atb::Node &ffnResidualAddNode = opGraph.nodes.at(nodeId++);

    // norm [1, n_tokens, hidden_size]
    atb::infer::LayerNormParam layerNormParam;
    layerNormParam.layerType = atb::infer::LayerNormParam::LayerNormType::LAYER_NORM_NORM;
    layerNormParam.normParam.epsilon = param.layerNormEps;
    layerNormParam.normParam.beginNormAxis = LAYER_NORM_AXIS_NUM;
    layerNormParam.normParam.beginParamsAxis = LAYER_NORM_AXIS_NUM;

    CREATE_OPERATION(layerNormParam, &inputLayerNormNode.operation);
    inputLayerNormNode.inTensorIds = {IN_HIDDENSTATES, IN_INPUTLAYERNORMWEIGTH, IN_INPUTLAYERNORMBIAS};
    inputLayerNormNode.outTensorIds = {INTERMEDIATE_INPUTLAYERNORMOUT};

    // qkv  [1, n_tokens, hidden_size] to [1, n_tokens, 3 * hidden_size]
    atb::infer::LinearParam linearParam = {false, false, true};
    CREATE_OPERATION(linearParam, &qkvLinearNode.operation);
    qkvLinearNode.inTensorIds = {INTERMEDIATE_INPUTLAYERNORMOUT, IN_QKVWEIGHT, IN_QKVBIAS};
    qkvLinearNode.outTensorIds = {INTERMEDIATE_MIXEDQKVLINEAROUT};

    // rope [1, n_tokens, hidden_size] to 3 * [1, n_tokens, hidden_size]
    atb_speed::gptneox_20b::PositionEmbeddingParam positionEmbeddingParam;
    positionEmbeddingParam.headNum = param.headNum;
    positionEmbeddingParam.dk = param.dk;
    positionEmbeddingParam.rotaryPct = param.rotaryPct;
    atb_speed::gptneox_20b::PositionEmbedding(positionEmbeddingParam, &positionEmbeddingNode.operation);
    positionEmbeddingNode.inTensorIds = {INTERMEDIATE_MIXEDQKVLINEAROUT, IN_POSITIONIDS, IN_COSEMBED, IN_SINEMBED};
    positionEmbeddingNode.outTensorIds = {INTERMEDIATE_QUERYEMBED, INTERMEDIATE_KEYEMBED, INTERMEDIATE_VALUE};

    atb::infer::SelfAttentionParam selfAttentionParam;
    selfAttentionParam.headDim = param.dk;
    selfAttentionParam.headNum = param.headNum;
    selfAttentionParam.qScale = param.qScale;
    selfAttentionParam.qkScale = param.qkScale;
    selfAttentionParam.isFusion = true;
    CREATE_OPERATION(selfAttentionParam, &selfAttentionKvCacheFusionNode.operation);
    selfAttentionKvCacheFusionNode.inTensorIds = {
        INTERMEDIATE_QUERYEMBED, INTERMEDIATE_KEYEMBED, INTERMEDIATE_VALUE, IN_CACHEK, IN_CACHEV,
        IN_ATTENTIONMASK, IN_TOKENOFFSET, IN_SEQLEN, IN_LAYERID};
    selfAttentionKvCacheFusionNode.outTensorIds = {INTERMEDIATE_SELFATTNOUT};

    // different parallel linear
    CREATE_OPERATION(linearParam, &selfAttnLinearNode.operation);
    selfAttnLinearNode.inTensorIds = {INTERMEDIATE_SELFATTNOUT, IN_SELFOUTLINEARWEIGHT, IN_SELFOUTLINEARBIAS};
    selfAttnLinearNode.outTensorIds = {INTERMEDIATE_SELFATTNLINEAROUT};

    CREATE_OPERATION(layerNormParam, &postAttnLayerNormNode.operation);
    postAttnLayerNormNode.inTensorIds = {IN_HIDDENSTATES, IN_POSTATTNLAYERNORMWEIGHT, IN_POSTATTNLAYERNORMBIAS};
    postAttnLayerNormNode.outTensorIds = {INTERMEDIATE_POSTATTNLAYERNORMOUT};

    CREATE_OPERATION(linearParam, &ffnLinearNode.operation);
    ffnLinearNode.inTensorIds = {INTERMEDIATE_POSTATTNLAYERNORMOUT, IN_FFNLINEARWEIGHT, IN_FFNLINEARBIAS};
    ffnLinearNode.outTensorIds = {INTERMEDIATE_FFNLINEAROUT};

    atb::infer::ActivationParam activationParam;
    activationParam.activationType = atb::infer::ActivationType::ACTIVATION_GELU;
    CREATE_OPERATION(activationParam, &ffnActNode.operation);
    ffnActNode.inTensorIds = {INTERMEDIATE_FFNLINEAROUT};
    ffnActNode.outTensorIds = {INTERMEDIATE_FFNACTOUT};

    CREATE_OPERATION(linearParam, &ffnOutLinearNode.operation);
    ffnOutLinearNode.inTensorIds = {INTERMEDIATE_FFNACTOUT, IN_FFNOUTLINEARWEIGHT, IN_FFNOUTLINEARBIAS};
    ffnOutLinearNode.outTensorIds = {INTERMEDIATE_FFNOUTLINEAROUT};

    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ELEWISE_ADD;
    CREATE_OPERATION(addParam, &attnResidualAddNode.operation);
    attnResidualAddNode.inTensorIds = {IN_HIDDENSTATES, INTERMEDIATE_SELFATTNLINEAROUT};
    attnResidualAddNode.outTensorIds = {INTERMEDIATE_ATTNRESIDUALADDOUT};

    CREATE_OPERATION(addParam, &ffnResidualAddNode.operation);
    ffnResidualAddNode.inTensorIds = {INTERMEDIATE_ATTNRESIDUALADDOUT, INTERMEDIATE_FFNOUTLINEAROUT};
    ffnResidualAddNode.outTensorIds = {OUT_GPTNEOXLAYEROUT};

    opGraph.inferShapeFunc = [](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
    };

    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}

atb::Operation *CreateFlashAttentionKvCacheRopeLayer(const nlohmann::json &paramJson)
{
    FlashAttentionKvCacheRopeParam param;
    param.layerNormEps = paramJson["layerNormEps"].get<float>();
    param.headNum = paramJson["headNum"].get<int>();
    param.dk = paramJson["dk"].get<int>();
    param.model = paramJson["model"].get<std::string>();
    param.qScale = paramJson["qScale"].get<float>();
    if (paramJson.contains("rotaryPct")) {
        param.rotaryPct = paramJson["rotaryPct"].get<float>();
    }
    if (paramJson.contains("isPrefill")) {
        param.isPrefill = paramJson["isPrefill"].get<bool>();
    }
    if (paramJson.contains("rank")) {
        param.rank = paramJson["rank"].get<int>();
    }
    if (paramJson.contains("rankSize")) {
        param.rankSize = paramJson["rankSize"].get<int>();
    }
    if (paramJson.contains("qkScale")) {
        param.qkScale = paramJson["qkScale"].get<int>();
    }

    ATB_LOG(INFO) << __func__ << " layerNormEps:" << param.layerNormEps << ", headNum:" << param.headNum
                  << ", dk:" << param.dk << ", model:" << param.model;
    atb::Operation *op;
    FlashAttentionKvCacheRopeLayer(param, &op);
    return op;
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
} // namespace gptneox_20b
} // namespace atb_speed