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

namespace atb_speed {
namespace baichuan2_13b {
enum LayerPATensorId : int {
    IN_HIDDEN_STATES = 0,       // [4096, 5120]  [1, 5120]
    IN_NORM_WEIGHT,             // [5120]
    IN_QKV_MIXED_LINEAR_WEIGHT, // [15360, 5120]
    IN_SELF_OUT_LINEAR_WEIGHT,  // [5120, 5120]
    IN_SELF_OUT_NORM_WEIGHT,    // [5120]
    IN_MLP_GATE_UP_WEIGHT,      // [27392, 5120]
    IN_MLP_DOWN_WEIGHT,         // [5120, 13696]
    IN_ATTENTION_MASK,          // [160, 64, 1024, 16]  [40, 2, 16, 16]
    IN_K_CACHE,                 // [36, 320, 128, 16]
    IN_V_CACHE,                 // [36, 320, 128, 16]
    IN_BLOCK_TABLES,            // [4, 9]
    IN_SLOTS,                   // [4096]
    IN_INPUT_LENGTHS,           // [4]
    IN_HOLDER,

    OUT_LAYER_OUT,

    INTERNAL_INPUT_NORM_OUT,
    INTERNAL_QKV_MIXED_LINEAR_OUT,
    INTERNAL_MIXED_Q,
    INTERNAL_MIXED_K,
    INTERNAL_MIXED_V,

    INTERNAL_ATTENTION_OUT, // attention output
    INTERNAL_SELF_LINEAR_OUT,
    INTERNAL_SELF_RESIDUAL_ADD_OUT,
    INTERNAL_SELF_NORM_OUT,
    INTERNAL_MLP_OUT,
};

static const uint64_t IN_TENSOR_COUNT = 14;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERNAL_TENSOR_COUNT = 10;
static const uint64_t NODE_COUNT = 10;

void reshapeHeads(const atb::Dims &oldShape, atb::Dims &newShape, int headNum)
{
    newShape.dimNum = 3;                           // dimNum: 3
    newShape.dims[0] = oldShape.dims[0];           // 0 dim: n tokens
    newShape.dims[1] = headNum;                    // 1 dim: head num
    newShape.dims[2] = oldShape.dims[1] / headNum; // 2 dim: head size, 1 hidden size
}

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
    atb::Node &reshapeAndCacheNode = opGraph.nodes.at(nodeId++);
    atb::Node &attentionNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfOutLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfResidualAddNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfNormNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpResidualAddNode = opGraph.nodes.at(nodeId++);

    // norm [n_tokens, hidden_size]
    atb::infer::RmsNormParam rmsNormParam;
    rmsNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    rmsNormParam.normParam.epsilon = param.rmsNormEps;
    CREATE_OPERATION(rmsNormParam, &inputNormNode.operation);
    inputNormNode.inTensorIds = {IN_HIDDEN_STATES, IN_NORM_WEIGHT};
    inputNormNode.outTensorIds = {INTERNAL_INPUT_NORM_OUT};

    // qkv  [n_tokens, hidden_size] to [n_tokens, 3 * hidden_size]
    atb::infer::LinearParam linearParam;
    linearParam.transposeB = param.transposedWeight;
    linearParam.hasBias = false;
    CREATE_OPERATION(linearParam, &qkvLinearNode.operation);
    qkvLinearNode.inTensorIds = {INTERNAL_INPUT_NORM_OUT, IN_QKV_MIXED_LINEAR_WEIGHT};
    qkvLinearNode.outTensorIds = {INTERNAL_QKV_MIXED_LINEAR_OUT};

    // q/k/v [n_tokens, hidden_size]
    atb::infer::SplitParam splitParam = {-1, 3};
    CREATE_OPERATION(splitParam, &splitQKVNode.operation);
    splitQKVNode.inTensorIds = {INTERNAL_QKV_MIXED_LINEAR_OUT};
    splitQKVNode.outTensorIds = {INTERNAL_MIXED_Q, INTERNAL_MIXED_K, INTERNAL_MIXED_V};

    atb::infer::ReshapeAndCacheParam reshapeCacheParm;
    CREATE_OPERATION(reshapeCacheParm, &reshapeAndCacheNode.operation);
    reshapeAndCacheNode.inTensorIds = {INTERNAL_MIXED_K, INTERNAL_MIXED_V, IN_K_CACHE, IN_V_CACHE, IN_SLOTS};
    reshapeAndCacheNode.outTensorIds = {IN_K_CACHE, IN_V_CACHE};
    reshapeAndCacheNode.inTensorReshapeFuncs.resize(reshapeAndCacheNode.inTensorIds.size());
    reshapeAndCacheNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        reshapeHeads(oldShape, newShape, param.headNum);
    };
    reshapeAndCacheNode.inTensorReshapeFuncs[1] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        reshapeHeads(oldShape, newShape, param.headNum);
    };

    if (param.isPrefill) {
        atb::infer::SelfAttentionParam faEnParam;
        faEnParam.headNum = param.headNum;
        faEnParam.qkScale = 1.0 / sqrt(param.dk);
        faEnParam.kvHeadNum = param.headNum;
        faEnParam.calcType = atb::infer::SelfAttentionParam::PA_ENCODER;
        faEnParam.maskType = atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_ALIBI;
        CREATE_OPERATION(faEnParam, &attentionNode.operation);
        attentionNode.inTensorIds = {INTERNAL_MIXED_Q, INTERNAL_MIXED_K, INTERNAL_MIXED_V, IN_ATTENTION_MASK,
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
        paDeParam.maskType = atb::infer::PagedAttentionParam::MaskType::MASK_TYPE_ALIBI;
        CREATE_OPERATION(paDeParam, &attentionNode.operation);
        attentionNode.inTensorIds = {INTERNAL_MIXED_Q, IN_K_CACHE,       IN_V_CACHE,
                                     IN_BLOCK_TABLES,  IN_INPUT_LENGTHS, IN_ATTENTION_MASK};
        attentionNode.outTensorIds = {INTERNAL_ATTENTION_OUT};
        attentionNode.inTensorReshapeFuncs.resize(attentionNode.inTensorIds.size());
        attentionNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            reshapeHeads(oldShape, newShape, param.headNum);
        };
    }

    atb_speed::common::ParallelParamV2 selfOutLinearParam;
    selfOutLinearParam.commParam.rank = param.rank;
    selfOutLinearParam.commParam.rankSize = param.rankSize;
    selfOutLinearParam.isBias = false;
    selfOutLinearParam.commParam.backend = param.backend;
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

    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CREATE_OPERATION(addParam, &selfResidualAddNode.operation);
    selfResidualAddNode.inTensorIds = {IN_HIDDEN_STATES, INTERNAL_SELF_LINEAR_OUT};
    selfResidualAddNode.outTensorIds = {INTERNAL_SELF_RESIDUAL_ADD_OUT};

    CREATE_OPERATION(rmsNormParam, &selfNormNode.operation);
    selfNormNode.inTensorIds = {INTERNAL_SELF_RESIDUAL_ADD_OUT, IN_SELF_OUT_NORM_WEIGHT};
    selfNormNode.outTensorIds = {INTERNAL_SELF_NORM_OUT};

    atb_speed::common::MlpGateParamV2 mlpParam;
    mlpParam.commDownParam.rank = param.rank;
    mlpParam.commDownParam.rankSize = param.rankSize;
    mlpParam.activationType = atb::infer::ActivationType::ACTIVATION_SWISH;
    mlpParam.transposeB = param.transposedWeight;
    mlpParam.isBias = false;
    mlpParam.isPack = true;
    mlpParam.commDownParam.backend = param.backend;
    atb_speed::common::MlpGateLayerV2(mlpParam, &mlpNode.operation);
    mlpNode.inTensorIds = {INTERNAL_SELF_NORM_OUT,
                           IN_MLP_GATE_UP_WEIGHT,
                           IN_HOLDER,
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
                           IN_HOLDER};
    mlpNode.outTensorIds = {INTERNAL_MLP_OUT};

    CREATE_OPERATION(addParam, &mlpResidualAddNode.operation);
    mlpResidualAddNode.inTensorIds = {INTERNAL_SELF_RESIDUAL_ADD_OUT, INTERNAL_MLP_OUT};
    mlpResidualAddNode.outTensorIds = {OUT_LAYER_OUT};

    opGraph.inferShapeFunc = [](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
    };

    CREATE_OPERATION(opGraph, operation);
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
} // namespace baichuan2_13b
} // namespace atb_speed