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
#include "pa_layer.h"

#include "models/gptneox/20b/operation/position_embedding_pa.h"

namespace atb_speed {
namespace gptneox_20b {
static const uint64_t IN_TENSOR_COUNT = 22;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 14;
static const uint64_t NODE_COUNT = 14;
static const uint64_t LAYER_NORM_AXIS_NUM = 1;

atb::Status PALayer(const PALayerParam &param, atb::Operation **operation)
{
    ATB_LOG(INFO) << __func__ << " called, headNum: " << param.headNum;
    atb::GraphParam opGraph;
    if (param.isPrefill) {
        opGraph.name = "Prefill_transformer_layer";
    } else {
        opGraph.name = "Decoder_transformer_layer";
    }

    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    atb::Node &inputLayerNormNode = opGraph.nodes.at(nodeId++);
    atb::Node &qkvLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &positionEmbeddingNode = opGraph.nodes.at(nodeId++);
    atb::Node &reshapeAndCacheNode = opGraph.nodes.at(nodeId++);
    atb::Node &mul0Node = opGraph.nodes.at(nodeId++);
    atb::Node &attentionNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfAttnLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &postAttnLayerNormNode = opGraph.nodes.at(nodeId++);

    atb::Node &ffnLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &ffnActNode = opGraph.nodes.at(nodeId++);
    atb::Node &ffnOutLinearNode = opGraph.nodes.at(nodeId++);

    atb::Node &ffnResidualAddNode = opGraph.nodes.at(nodeId++); // ffn add attention
    atb::Node &allReduceNode = opGraph.nodes.at(nodeId++);
    atb::Node &attnResidualAddNode = opGraph.nodes.at(nodeId++); // add hidden state

    // norm [n_tokens, hidden_size]
    atb::infer::LayerNormParam layerNormParam;
    layerNormParam.layerType = atb::infer::LayerNormParam::LayerNormType::LAYER_NORM_NORM;
    layerNormParam.normParam.epsilon = param.layerNormEps;
    layerNormParam.normParam.beginNormAxis = LAYER_NORM_AXIS_NUM;
    layerNormParam.normParam.beginParamsAxis = LAYER_NORM_AXIS_NUM;

    CREATE_OPERATION(layerNormParam, &inputLayerNormNode.operation);
    inputLayerNormNode.inTensorIds = { IN_HIDDENSTATES, IN_INPUTLAYERNORMWEIGTH, IN_INPUTLAYERNORMBIAS };
    inputLayerNormNode.outTensorIds = { INTERMEDIATE_INPUTLAYERNORMOUT };

    // qkv [n_tokens, hidden_size] to [n_tokens, 3 * hidden_size]
    atb::infer::LinearParam linearParam;
    CREATE_OPERATION(linearParam, &qkvLinearNode.operation);
    qkvLinearNode.inTensorIds = { INTERMEDIATE_INPUTLAYERNORMOUT, IN_QKVWEIGHT, IN_QKVBIAS };
    qkvLinearNode.outTensorIds = { INTERMEDIATE_MIXEDQKVLINEAROUT };

    // rope [n_tokens, hidden_size] to 3 * [n_tokens, hidden_size]
    atb_speed::gptneox_20b::PositionEmbeddingPAParam positionEmbeddingPAParam;
    positionEmbeddingPAParam.headNum = param.headNum;
    positionEmbeddingPAParam.dk = param.dk;
    positionEmbeddingPAParam.rotaryPct = param.rotaryPct;
    atb_speed::gptneox_20b::PositionEmbeddingPAOperation(positionEmbeddingPAParam, &positionEmbeddingNode.operation);
    positionEmbeddingNode.inTensorIds = { INTERMEDIATE_MIXEDQKVLINEAROUT, IN_COSEMBED, IN_SINEMBED, IN_INPUT_LENGTHS };
    positionEmbeddingNode.outTensorIds = { INTERMEDIATE_QUERYEMBED, INTERMEDIATE_KEYEMBED, INTERMEDIATE_VALUE };

    // self attention
    atb::infer::ReshapeAndCacheParam reshapeCacheParm;
    CREATE_OPERATION(reshapeCacheParm, &reshapeAndCacheNode.operation);
    reshapeAndCacheNode.inTensorIds = { INTERMEDIATE_KEYEMBED, INTERMEDIATE_VALUE, IN_CACHEK, IN_CACHEV, IN_SLOTS };
    reshapeAndCacheNode.outTensorIds = {};

    atb::infer::ElewiseParam Mul0Param;
    Mul0Param.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MULS;
    Mul0Param.mulsParam.varAttr = param.qScale;
    CreateOperation(Mul0Param, &mul0Node.operation);
    mul0Node.inTensorIds = { INTERMEDIATE_QUERYEMBED };
    mul0Node.outTensorIds = { INTERMEDIATE_QUERYEMBED_SCALED };
    mul0Node.inTensorReshapeFuncs.resize(mul0Node.inTensorIds.size());

    if (param.isPrefill) {
        atb::infer::SelfAttentionParam faEnParam;
        faEnParam.headNum = param.headNum;
        faEnParam.qkScale = param.qkScale;
        faEnParam.kvHeadNum = param.headNum;
        faEnParam.isEncoder = true;
        CREATE_OPERATION(faEnParam, &attentionNode.operation);
        attentionNode.inTensorIds = { INTERMEDIATE_QUERYEMBED_SCALED, INTERMEDIATE_KEYEMBED, INTERMEDIATE_VALUE,
            IN_ATTENTIONMASK, IN_INPUT_LENGTHS };
        attentionNode.outTensorIds = { INTERMEDIATE_SELFATTNOUT };
    } else {
        atb::infer::PagedAttentionParam paDeParam;
        paDeParam.headNum = param.headNum;
        paDeParam.qkScale = param.qkScale;
        paDeParam.kvHeadNum = param.headNum;
        CREATE_OPERATION(paDeParam, &attentionNode.operation);
        attentionNode.inTensorIds = { INTERMEDIATE_QUERYEMBED_SCALED, IN_CACHEK, IN_CACHEV, IN_BLOCK_TABLES,
            IN_INPUT_LENGTHS };
        attentionNode.outTensorIds = { INTERMEDIATE_SELFATTNOUT };
    }

    CREATE_OPERATION(linearParam, &selfAttnLinearNode.operation);
    selfAttnLinearNode.inTensorIds = { INTERMEDIATE_SELFATTNOUT, IN_SELFOUTLINEARWEIGHT, IN_SELFOUTLINEARBIAS };
    selfAttnLinearNode.outTensorIds = { INTERMEDIATE_SELFATTNLINEAROUT };
    selfAttnLinearNode.inTensorReshapeFuncs.resize(selfAttnLinearNode.inTensorIds.size());
    selfAttnLinearNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2;                                    // 2: dim num
        newShape.dims[0] = oldShape.dims[0];                    // 0: dim 0, n tokens
        newShape.dims[1] = oldShape.dims[1] * oldShape.dims[2]; // 1 hidden size: old 1, head num , old 2 head size
    };

    // mlp
    CREATE_OPERATION(layerNormParam, &postAttnLayerNormNode.operation);
    postAttnLayerNormNode.inTensorIds = { IN_HIDDENSTATES, IN_POSTATTNLAYERNORMWEIGHT, IN_POSTATTNLAYERNORMBIAS };
    postAttnLayerNormNode.outTensorIds = { INTERMEDIATE_POSTATTNLAYERNORMOUT };

    CREATE_OPERATION(linearParam, &ffnLinearNode.operation);
    ffnLinearNode.inTensorIds = { INTERMEDIATE_POSTATTNLAYERNORMOUT, IN_FFNLINEARWEIGHT, IN_FFNLINEARBIAS };
    ffnLinearNode.outTensorIds = { INTERMEDIATE_FFNLINEAROUT };

    atb::infer::ActivationParam activationParam;
    activationParam.activationType = atb::infer::ActivationType::ACTIVATION_GELU;
    CREATE_OPERATION(activationParam, &ffnActNode.operation);
    ffnActNode.inTensorIds = { INTERMEDIATE_FFNLINEAROUT };
    ffnActNode.outTensorIds = { INTERMEDIATE_FFNACTOUT };

    CREATE_OPERATION(linearParam, &ffnOutLinearNode.operation);
    ffnOutLinearNode.inTensorIds = { INTERMEDIATE_FFNACTOUT, IN_FFNOUTLINEARWEIGHT, IN_FFNOUTLINEARBIAS };
    ffnOutLinearNode.outTensorIds = { INTERMEDIATE_FFNOUTLINEAROUT };

    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CREATE_OPERATION(addParam, &ffnResidualAddNode.operation);
    ffnResidualAddNode.inTensorIds = { INTERMEDIATE_SELFATTNLINEAROUT, INTERMEDIATE_FFNOUTLINEAROUT };
    ffnResidualAddNode.outTensorIds = { INTERMEDIATE_ATTNMLPADDOUT };

    atb::infer::AllReduceParam allReduceParam;
    allReduceParam.rank = param.rank;
    allReduceParam.rankSize = param.rankSize;
    allReduceParam.backend = param.backend;
    CREATE_OPERATION(allReduceParam, &allReduceNode.operation);
    allReduceNode.inTensorIds = { INTERMEDIATE_ATTNMLPADDOUT };
    allReduceNode.outTensorIds = { INTERMEDIATE_ATTNMLP_ALLREDUCEOUT };

    CREATE_OPERATION(addParam, &attnResidualAddNode.operation);
    attnResidualAddNode.inTensorIds = { IN_HIDDENSTATES, INTERMEDIATE_ATTNMLP_ALLREDUCEOUT };
    attnResidualAddNode.outTensorIds = { OUT_GPTNEOXLAYEROUT };

    opGraph.inferShapeFunc = [](const atb::SVector<atb::TensorDesc> &inTensorDescs,
        atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
    };

    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}

atb::Operation *CreatePALayer(const nlohmann::json &paramJson)
{
    PALayerParam param;
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
        param.qkScale = paramJson["qkScale"].get<float>();
    }

    ATB_LOG(INFO) << __func__ << " layerNormEps:" << param.layerNormEps << ", headNum:" << param.headNum << ", dk:" <<
        param.dk << ", model:" << param.model;
    atb::Operation *op;
    PALayer(param, &op);
    return op;
}
} // namespace gptneox_20b
} // namespace atb_speed
