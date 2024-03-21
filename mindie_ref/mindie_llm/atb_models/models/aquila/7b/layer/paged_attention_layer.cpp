/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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

namespace atb_speed {
namespace aquila_7b {
enum PagedAttentionRopeLayerTensorId : int {
    IN_HIDDEN_STATES = 0, // [batchSize, seqLen, hiddenSize]
    IN_NORM_WEIGHT,
    IN_Q_LINEAR_WEIGHT,
    IN_K_LINEAR_WEIGHT,
    IN_V_LINEAR_WEIGHT,
    IN_SELF_OUT_LINEAR_WEIGHT,
    IN_SELF_OUT_NORM_WEIGHT,
    IN_MLP_GATE_WEIGHT,
    IN_MLP_DOWN_WEIGHT,
    IN_MLP_UP_WEIGHT,
    IN_POSITION_IDS, // inputs
    IN_COS_EMBED, // 目前只支持FP16
    IN_SIN_EMBED,
    IN_ATTENTION_MASK,
    IN_K_CACHE,       // [36, 320, 128, 16]
    IN_V_CACHE,       // [36, 320, 128, 16]
    IN_BLOCK_TABLES,  // [4, 9]
    IN_SLOTS,         // [4096]
    IN_INPUT_LENGTHS, // [4]
    IN_HOLDER,
    OUT_LAYER_OUT,
    INTERNAL_INPUT_NORM_OUT,
    INTERNAL_Q_LINEAR_OUT,
    INTERNAL_K_LINEAR_OUT,
    INTERNAL_V_LINEAR_OUT,
    INTERNAL_Q_EMBED,
    INTERNAL_K_EMBED,
    INTERNAL_ATTENTION_OUT, // attention output
    INTERNAL_SELF_LINEAR_OUT,
    INTERNAL_ATTENTION_RESIDUAL_ADD_OUT,
    INTERNAL_SELF_NORM_OUT,
    INTERNAL_MLP_OUT,
};

static const uint64_t IN_TENSOR_COUNT = 20;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 11;
static const uint64_t NODE_COUNT = 12;

void from_json(const nlohmann::json &paramJson, PagedAttentionRopeLayerParam &param)
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

void reshapeHeads(const atb::Dims &oldShape, atb::Dims &newShape, int headNum)
{
    newShape.dimNum = 3;                           // dimNum: 3
    newShape.dims[0] = oldShape.dims[0];           // 0 dim: n tokens
    newShape.dims[1] = headNum;                    // 1 dim: head num
    newShape.dims[2] = oldShape.dims[1] / headNum; // 2 dim: head size, 1 hidden size
}

atb::Operation *CreatePagedAttentionRopeLayer(const nlohmann::json &paramJson)
{
    ATB_LOG(INFO) << GetFuncNameAndNameSpace(__PRETTY_FUNCTION__);
    atb::Operation *op;
    atb_speed::aquila_7b::PagedAttentionRopeLayer(paramJson.get<PagedAttentionRopeLayerParam>(), &op);
    return op;
}

atb::Status PagedAttentionRopeLayer(const PagedAttentionRopeLayerParam &param, atb::Operation **operation)
{
    ATB_LOG(INFO) << __func__ << " called, headNum: " << param.headNum;
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);
    if (param.isPrefill) {
        opGraph.name = "Prefill_transformer_layer";
    } else {
        opGraph.name = "Decoder_transformer_layer";
    }

    size_t nodeId = 0;
    atb::Node &inputNormNode = opGraph.nodes.at(nodeId++);
    atb::Node &qLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &kLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &vLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &ropeNode = opGraph.nodes.at(nodeId++);
    atb::Node &reshapeAndCacheNode = opGraph.nodes.at(nodeId++);
    atb::Node &attentionNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfOutLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &attentionResidualAddNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfNormNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpResidualAddNode = opGraph.nodes.at(nodeId++);

    // input_layernorm
    atb::infer::RmsNormParam rmsNormParam;
    rmsNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    rmsNormParam.normParam.epsilon = param.rmsNormEps;
    CREATE_OPERATION(rmsNormParam, &inputNormNode.operation);
    inputNormNode.inTensorIds = {IN_HIDDEN_STATES, IN_NORM_WEIGHT};
    inputNormNode.outTensorIds = {INTERNAL_INPUT_NORM_OUT};

    // q_proj
    atb::infer::LinearParam linearParam;
    linearParam.hasBias = false;
    CREATE_OPERATION(linearParam, &qLinearNode.operation);
    qLinearNode.inTensorIds = {INTERNAL_INPUT_NORM_OUT, IN_Q_LINEAR_WEIGHT};
    qLinearNode.outTensorIds = {INTERNAL_Q_LINEAR_OUT};

    // k_proj
    CREATE_OPERATION(linearParam, &kLinearNode.operation);
    kLinearNode.inTensorIds = {INTERNAL_INPUT_NORM_OUT, IN_K_LINEAR_WEIGHT};
    kLinearNode.outTensorIds = {INTERNAL_K_LINEAR_OUT};

    // v_proj
    CREATE_OPERATION(linearParam, &vLinearNode.operation);
    vLinearNode.inTensorIds = {INTERNAL_INPUT_NORM_OUT, IN_V_LINEAR_WEIGHT};
    vLinearNode.outTensorIds = {INTERNAL_V_LINEAR_OUT};

    // rope (q_embedding + k_embedding)
    atb::infer::RopeParam ropeParam;
    ropeParam.rotaryCoeff = 2;
    CREATE_OPERATION(ropeParam, &ropeNode.operation);
    ropeNode.inTensorIds = {INTERNAL_Q_LINEAR_OUT, INTERNAL_K_LINEAR_OUT, IN_COS_EMBED, IN_SIN_EMBED, IN_INPUT_LENGTHS};
    ropeNode.outTensorIds = {INTERNAL_Q_EMBED, INTERNAL_K_EMBED};

    atb::infer::ReshapeAndCacheParam reshapeCacheParm;
    CREATE_OPERATION(reshapeCacheParm, &reshapeAndCacheNode.operation);
    reshapeAndCacheNode.inTensorIds = {INTERNAL_K_EMBED, INTERNAL_V_LINEAR_OUT, IN_K_CACHE, IN_V_CACHE, IN_SLOTS};
    reshapeAndCacheNode.outTensorIds = {IN_K_CACHE, IN_V_CACHE};
    reshapeAndCacheNode.inTensorReshapeFuncs.resize(reshapeAndCacheNode.inTensorIds.size());
    reshapeAndCacheNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        reshapeHeads(oldShape, newShape, param.headNum);
    };
    reshapeAndCacheNode.inTensorReshapeFuncs[1] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        reshapeHeads(oldShape, newShape, param.headNum);
    };

    if (param.isPrefill) {
        atb::infer::SelfAttentionParam selfAttentionParam;
        selfAttentionParam.headNum = param.headNum;
        selfAttentionParam.qkScale = 1.0 / sqrt(param.dk);
        selfAttentionParam.kvHeadNum = param.headNum;
        selfAttentionParam.calcType = atb::infer::SelfAttentionParam::PA_ENCODER;
        selfAttentionParam.maskType = atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_NORM;
        CREATE_OPERATION(selfAttentionParam, &attentionNode.operation);
        attentionNode.inTensorIds = {INTERNAL_Q_EMBED, INTERNAL_K_EMBED, INTERNAL_V_LINEAR_OUT, IN_ATTENTION_MASK,
                                     IN_INPUT_LENGTHS};
        attentionNode.outTensorIds = {INTERNAL_ATTENTION_OUT};
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
        CREATE_OPERATION(paDeParam, &attentionNode.operation);
        attentionNode.inTensorIds = {INTERNAL_Q_EMBED, IN_K_CACHE, IN_V_CACHE, IN_BLOCK_TABLES, IN_INPUT_LENGTHS};
        attentionNode.outTensorIds = {INTERNAL_ATTENTION_OUT};
        attentionNode.inTensorReshapeFuncs.resize(attentionNode.inTensorIds.size());
        attentionNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            reshapeHeads(oldShape, newShape, param.headNum);
        };
    }

    // o_proj
    atb_speed::common::ParallelParamV2 selfOutLinearParam;
    selfOutLinearParam.commParam.rank = param.rank;
    selfOutLinearParam.commParam.rankSize = param.rankSize;
    selfOutLinearParam.commParam.backend = param.backend;
    selfOutLinearParam.isBias = false;
    selfOutLinearParam.transposeB = param.transposedWeight;
    atb_speed::common::RowParallelLinearV2(selfOutLinearParam, &selfOutLinearNode.operation);
    selfOutLinearNode.inTensorIds = {
        INTERNAL_ATTENTION_OUT, IN_SELF_OUT_LINEAR_WEIGHT, IN_HOLDER, IN_HOLDER, IN_HOLDER, IN_HOLDER, IN_HOLDER};
    selfOutLinearNode.outTensorIds = {INTERNAL_SELF_LINEAR_OUT};
    selfOutLinearNode.inTensorReshapeFuncs.resize(selfOutLinearNode.inTensorIds.size());
    selfOutLinearNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2;                                    // 2: dim num
        newShape.dims[0] = oldShape.dims[0];                    // 0: dim 0, n tokens
        newShape.dims[1] = oldShape.dims[1] * oldShape.dims[2]; // 1 hidden size: old 1, head num , old 2 head size
    };

    // residual
    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CREATE_OPERATION(addParam, &attentionResidualAddNode.operation);
    attentionResidualAddNode.inTensorIds = {IN_HIDDEN_STATES, INTERNAL_SELF_LINEAR_OUT};
    attentionResidualAddNode.outTensorIds = {INTERNAL_ATTENTION_RESIDUAL_ADD_OUT};

    // post_attention_layernorm
    CREATE_OPERATION(rmsNormParam, &selfNormNode.operation);
    selfNormNode.inTensorIds = {INTERNAL_ATTENTION_RESIDUAL_ADD_OUT, IN_SELF_OUT_NORM_WEIGHT};
    selfNormNode.outTensorIds = {INTERNAL_SELF_NORM_OUT};

    // mlp
    atb_speed::common::MlpGateParamV2 mlpParam;
    mlpParam.commDownParam.rank = param.rank;
    mlpParam.commDownParam.rankSize = param.rankSize;
    mlpParam.commDownParam.backend = param.backend;
    mlpParam.activationType = atb::infer::ActivationType::ACTIVATION_SWISH;
    mlpParam.transposeB = param.transposedWeight;
    mlpParam.isBias = false;
    mlpParam.isPack = false;
    atb_speed::common::MlpGateLayerV2(mlpParam, &mlpNode.operation);
    mlpNode.inTensorIds = {
        INTERNAL_SELF_NORM_OUT,
        IN_MLP_UP_WEIGHT,
        IN_MLP_GATE_WEIGHT,
        IN_MLP_DOWN_WEIGHT,
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
        IN_HOLDER,
    };
    mlpNode.outTensorIds = {INTERNAL_MLP_OUT};

    // residual
    CREATE_OPERATION(addParam, &mlpResidualAddNode.operation);
    mlpResidualAddNode.inTensorIds = {INTERNAL_ATTENTION_RESIDUAL_ADD_OUT, INTERNAL_MLP_OUT};
    mlpResidualAddNode.outTensorIds = {OUT_LAYER_OUT};

    opGraph.inferShapeFunc = [](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
    };
    CREATE_OPERATION(opGraph, operation);

    return atb::NO_ERROR;
}

PagedAttentionRopeLayerBinder::PagedAttentionRopeLayerBinder() = default;

PagedAttentionRopeLayerBinder::~PagedAttentionRopeLayerBinder() = default;

void PagedAttentionRopeLayerBinder::ParseParam(const nlohmann::json &paramJson)
{
    seqLen_.clear();
    for (const auto &item : paramJson["seqLen"]) {
        seqLen_.push_back(item.get<int32_t>());
    }
}

void PagedAttentionRopeLayerBinder::BindTensor(atb::VariantPack &variantPack)
{
    variantPack.inTensors.at(IN_INPUT_LENGTHS).hostData = seqLen_.data();
}
} // namespace aquila_7b
} // namespace atb_speed