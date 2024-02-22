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
#include "models/qwen/14b/operation/rope.h"

namespace atb_speed {
namespace qwen_14b {
enum LayerPATensorId : int {
    IN_HIDDENSTATES = 0,

    IN_NORMWEIGHT,
    IN_QKVMIXEDLINEARWEIGHT,
    IN_QKVMIXEDLINEARBIAS,
    IN_SELFOUTLINEARWEIGHT,
    IN_SELFOUTNORMWEIGHT,

    IN_MLPW2W1WEIGHT,
    IN_MLPCPROJWEIGHT,

    IN_COS_EMBED,
    IN_SIN_EMBED,
    IN_ATTENTIONMASK,
    IN_K_CACHE,
    IN_V_CACHE,

    IN_BLOCK_TABLES,
    IN_SLOTS,
    IN_INPUT_LENGTHS,

    IN_HOLDER,

    OUT_LAYEROUT,

    INTERNAL_INPUTNORMOUT,
    INTERNAL_QKVMIXEDLINEAROUT,
    INTERNAL_Q,
    INTERNAL_K,
    INTERNAL_V,
    INTERNAL_QEMBED,
    INTERNAL_KEMBED,
    INTERNAL_SELFATTENTIONOUT,
    INTERNAL_SELFLINEAROUT,
    INTERNAL_ATTENTIONRESIDUALADDOUT,
    INTERNAL_SELFNORMOUT,
    INTERNAL_MLPOUT,
};

static const uint64_t IN_TENSOR_COUNT = 17;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERNAL_TENSOR_COUNT = 12;
static const uint64_t NODE_COUNT = 11;

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

atb::Operation *CreatePALayer(const nlohmann::json &paramJson)
{
    ATB_LOG(INFO) << GetFuncNameAndNameSpace(__PRETTY_FUNCTION__);
    atb::Operation *op;
    PALayer(paramJson.get<PALayerParam>(), &op);
    return op;
}

void reshapeHeads(const atb::Dims &oldShape, atb::Dims &newShape, int headNum)
{
    newShape.dimNum = 3;                           // dimNum: 3
    newShape.dims[0] = oldShape.dims[0];           // 0 dim: n tokens
    newShape.dims[1] = headNum;                    // 1 dim: head num
    newShape.dims[2] = oldShape.dims[1] / headNum; // 2 dim: head size, 1 hidden size
}  // -> [nTokens, headNum, headDim]

atb::Status PALayer(const PALayerParam &param, atb::Operation **operation)
{
    ATB_LOG(INFO) << __func__ << " called, headNum: " << param.headNum;
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERNAL_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);
    if (param.isPrefill) {
        opGraph.name = "Prefill_transformer_layer";
    } else {
        opGraph.name = "Decoder_transformer_layer";
    }

    size_t nodeId = 0;
    atb::Node &inputNormNode = opGraph.nodes.at(nodeId++);
    atb::Node &qkvLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &splitQKVNode = opGraph.nodes.at(nodeId++);
    atb::Node &ropeNode = opGraph.nodes.at(nodeId++);
    atb::Node &reshapeAndCacheNode = opGraph.nodes.at(nodeId++);
    atb::Node &attentionNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfOutLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &attentionResidualAddNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfNormNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpResidualAddNode = opGraph.nodes.at(nodeId++);

    // ln_1
    atb::infer::RmsNormParam rmsNormParam;
    rmsNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    rmsNormParam.normParam.epsilon = param.rmsNormEps;
    CreateOperation(rmsNormParam, &inputNormNode.operation);
    inputNormNode.inTensorIds = {IN_HIDDENSTATES, IN_NORMWEIGHT};
    inputNormNode.outTensorIds = {INTERNAL_INPUTNORMOUT};

    // c_attn
    atb::infer::LinearParam linearParam;
    CreateOperation(linearParam, &qkvLinearNode.operation);
    qkvLinearNode.inTensorIds = {INTERNAL_INPUTNORMOUT, IN_QKVMIXEDLINEARWEIGHT, IN_QKVMIXEDLINEARBIAS};
    qkvLinearNode.outTensorIds = {INTERNAL_QKVMIXEDLINEAROUT};

    // split
    atb::infer::SplitParam splitParam = {-1, 3};
    CreateOperation(splitParam, &splitQKVNode.operation);
    splitQKVNode.inTensorIds = {INTERNAL_QKVMIXEDLINEAROUT};
    splitQKVNode.outTensorIds = {INTERNAL_Q, INTERNAL_K, INTERNAL_V};

    // rope (q_embedding + k_embedding)
    atb::infer::RopeParam ropeParam;
    ropeParam.rotaryCoeff = 2;
    CREATE_OPERATION(ropeParam, &ropeNode.operation);
    ropeNode.inTensorIds = {INTERNAL_Q, INTERNAL_K, IN_COS_EMBED, IN_SIN_EMBED, IN_INPUT_LENGTHS};
    ropeNode.outTensorIds = {INTERNAL_QEMBED, INTERNAL_KEMBED};

    atb::infer::ReshapeAndCacheParam reshapeCacheParm;
    CreateOperation(reshapeCacheParm, &reshapeAndCacheNode.operation);
    reshapeAndCacheNode.inTensorIds = {INTERNAL_KEMBED, INTERNAL_V, IN_K_CACHE, IN_V_CACHE, IN_SLOTS};
    reshapeAndCacheNode.outTensorIds = {};
    reshapeAndCacheNode.inTensorReshapeFuncs.resize(reshapeAndCacheNode.inTensorIds.size());
    reshapeAndCacheNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        reshapeHeads(oldShape, newShape, param.headNum);
    };
    reshapeAndCacheNode.inTensorReshapeFuncs[1] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        reshapeHeads(oldShape, newShape, param.headNum);
    };

    // paged attention
    if (param.isPrefill) {
        atb::infer::SelfAttentionParam faEnParam;
        faEnParam.headNum = param.headNum;
        faEnParam.qkScale = 1.0 / sqrt(param.dk);
        faEnParam.kvHeadNum = param.headNum;
        faEnParam.isEncoder = true;
        CreateOperation(faEnParam, &attentionNode.operation);
        attentionNode.inTensorIds = {INTERNAL_QEMBED,
                                     INTERNAL_KEMBED,
                                     INTERNAL_V,
                                     IN_ATTENTIONMASK,
                                     IN_INPUT_LENGTHS};
        attentionNode.outTensorIds = {INTERNAL_SELFATTENTIONOUT};
        attentionNode.inTensorReshapeFuncs.resize(attentionNode.inTensorIds.size());
        attentionNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            reshapeHeads(oldShape, newShape, param.headNum);
        };
        attentionNode.inTensorReshapeFuncs[1] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            reshapeHeads(oldShape, newShape, param.headNum);
        };
        attentionNode.inTensorReshapeFuncs[2] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            reshapeHeads(oldShape, newShape, param.headNum);
        };
    } else {
        atb::infer::PagedAttentionParam paDeParam;
        paDeParam.headNum = param.headNum;
        paDeParam.qkScale = 1.0 / sqrt(param.dk);
        paDeParam.kvHeadNum = param.headNum;
        CreateOperation(paDeParam, &attentionNode.operation);
        attentionNode.inTensorIds = {INTERNAL_QEMBED,
                                     IN_K_CACHE,
                                     IN_V_CACHE,
                                     IN_BLOCK_TABLES,
                                     IN_INPUT_LENGTHS};
        attentionNode.outTensorIds = {INTERNAL_SELFATTENTIONOUT};
        attentionNode.inTensorReshapeFuncs.resize(attentionNode.inTensorIds.size());
        attentionNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            reshapeHeads(oldShape, newShape, param.headNum);
        };
    }

    // c_proj
    atb_speed::common::ParallelParamV2 selfOutLinearParam;
    selfOutLinearParam.commParam.rank = param.rank;
    selfOutLinearParam.commParam.rankSize = param.rankSize;
    selfOutLinearParam.commParam.backend = param.backend;
    selfOutLinearParam.isBias = false;
    atb_speed::common::RowParallelLinearV2(selfOutLinearParam, &selfOutLinearNode.operation);
    selfOutLinearNode.inTensorIds = {INTERNAL_SELFATTENTIONOUT,
                                     IN_SELFOUTLINEARWEIGHT,
                                     IN_HOLDER,
                                     IN_HOLDER,
                                     IN_HOLDER,
                                     IN_HOLDER,
                                     IN_HOLDER};
    selfOutLinearNode.outTensorIds = {INTERNAL_SELFLINEAROUT};
    selfOutLinearNode.inTensorReshapeFuncs.resize(selfOutLinearNode.inTensorIds.size());
    selfOutLinearNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2;
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = oldShape.dims[1] * oldShape.dims[2];
    };

    // residual
    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CreateOperation(addParam, &attentionResidualAddNode.operation);
    attentionResidualAddNode.inTensorIds = {IN_HIDDENSTATES, INTERNAL_SELFLINEAROUT};
    attentionResidualAddNode.outTensorIds = {INTERNAL_ATTENTIONRESIDUALADDOUT};

    // ln_2
    CreateOperation(rmsNormParam, &selfNormNode.operation);
    selfNormNode.inTensorIds = {INTERNAL_ATTENTIONRESIDUALADDOUT, IN_SELFOUTNORMWEIGHT};
    selfNormNode.outTensorIds = {INTERNAL_SELFNORMOUT};

    // mlp
    atb_speed::common::MlpGateParamV2 mlpParam;
    mlpParam.commDownParam.rank = param.rank;
    mlpParam.commDownParam.rankSize = param.rankSize;
    mlpParam.commDownParam.backend = param.backend;
    mlpParam.activationType = atb::infer::ActivationType::ACTIVATION_SWISH;
    mlpParam.transposeB = param.transposedWeight;
    mlpParam.isBias = false;
    mlpParam.isPack = true;
    atb_speed::common::MlpGateLayerV2(mlpParam, &mlpNode.operation);
    mlpNode.inTensorIds = {INTERNAL_SELFNORMOUT,
                           IN_MLPW2W1WEIGHT,
                           IN_HOLDER,
                           IN_MLPCPROJWEIGHT,
                           IN_HOLDER,
                           IN_HOLDER,
                           IN_HOLDER,
                           IN_HOLDER,
                           IN_HOLDER,
                           IN_HOLDER,
                           IN_HOLDER,
                           IN_HOLDER,
                           IN_HOLDER,
                           IN_HOLDER,
                           IN_HOLDER,
                           IN_HOLDER,
                           IN_HOLDER,
                           IN_HOLDER,
                           IN_HOLDER};
    mlpNode.outTensorIds = {INTERNAL_MLPOUT};

    // residual
    CreateOperation(addParam, &mlpResidualAddNode.operation);
    mlpResidualAddNode.inTensorIds = {INTERNAL_ATTENTIONRESIDUALADDOUT, INTERNAL_MLPOUT};
    mlpResidualAddNode.outTensorIds = {OUT_LAYEROUT};

    opGraph.inferShapeFunc = [](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
    };
    atb::CreateOperation(opGraph, operation);
    return atb::NO_ERROR;
}

FlashAttentionHostBinder::FlashAttentionHostBinder() = default;

FlashAttentionHostBinder::~FlashAttentionHostBinder() = default;

void FlashAttentionHostBinder::ParseParam(const nlohmann::json &paramJson)
{
    seqLen_.clear();
    for (const auto &item : paramJson["seqLen"]) {
        seqLen_.push_back(item.get<int32_t>());
    }
}

void FlashAttentionHostBinder::BindTensor(atb::VariantPack &variantPack)
{
    variantPack.inTensors.at(IN_INPUT_LENGTHS).hostData = seqLen_.data();
}
}  // namespace qwen_14b
}  // namespace atb_speed