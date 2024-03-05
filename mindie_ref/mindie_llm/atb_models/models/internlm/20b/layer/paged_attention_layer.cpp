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

#include "layers/mlp_gate.h"
#include "layers/parallel_layer.h"
#include "models/internlm/7b/operation/rope.h"

namespace atb_speed {
namespace internlm_20b {
enum PagedAttentionLayerTensorId : int {
    IN_HIDDENSTATES = 0,
    IN_NORMWEIGHT,
    IN_Q_LINEARWEIGHT,
    IN_K_LINEARWEIGHT,
    IN_V_LINEARWEIGHT,
    IN_SELFOUTLINEARWEIGHT,
    IN_SELFOUTNORMWEIGHT,
    IN_MLPGATEWEIGHT,
    IN_MLPDOWNWEIGHT,
    IN_MLPUPWEIGHT,
    IN_POSITIONIDS, // inputs
    IN_COS_EMBED,   // 目前只支持FP16
    IN_SIN_EMBED,
    IN_ATTENTION_MASK,
    IN_K_CACHE,
    IN_V_CACHE,
    IN_BLOCK_TABLES,
    IN_SLOTS,
    IN_INPUT_LENGTHS,
    IN_HOLDER,
    OUT_LAYEROUT,
    INTERMIDATE_INPUTNORMOUT,
    INTERMIDATE_Q_MIXEDLINEAROUT,
    INTERMIDATE_K_MIXEDLINEAROUT,
    INTERMIDATE_V_MIXEDLINEAROUT,
    INTERMIDATE_Q_POSITIONEMBED,
    INTERMIDATE_K_POSITIONEMBED,
    INTERMIDATE_SELFOUT,
    INTERMIDATE_SELFLINEAROUT,
    INTERMIDATE_SELFRESIDUALADDOUT,
    INTERMIDATE_SELFNORMOUT,
    INTERMIDATE_MLPOUT,
};

static const uint64_t IN_TENSOR_COUNT = 20;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 11;
static const uint64_t NODE_COUNT = 12;
static const uint64_t SELF_ATTENTION_V_INPUT_INDEX = 2;
static const uint64_t SELF_ATTENTION_V_INPUT_SIZE = 4;
static const uint64_t ROTARY_COEFF = 2;

void reshapeHeads(const atb::Dims &oldShape, atb::Dims &newShape, int headNum)
{
    newShape.dimNum = 3;                           // dimNum: 3
    newShape.dims[0] = oldShape.dims[0];           // 0 dim: n tokens
    newShape.dims[1] = headNum;                    // 1 dim: head num
    newShape.dims[2] = oldShape.dims[1] / headNum; // 2 dim: head size, 1 hidden size
}

atb::Status PagedAttentionLayer(const PagedAttentionLayerParam &param, atb::Operation **operation)
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
    atb::Node &selfResidualAddNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfNormNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpResidualAddNode = opGraph.nodes.at(nodeId++);

    atb::infer::RmsNormParam rmsNormParam;
    rmsNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    rmsNormParam.normParam.epsilon = param.rmsNormEps;
    CREATE_OPERATION(rmsNormParam, &inputNormNode.operation);
    inputNormNode.inTensorIds = {IN_HIDDENSTATES, IN_NORMWEIGHT};
    inputNormNode.outTensorIds = {INTERMIDATE_INPUTNORMOUT};

    atb::infer::LinearParam linearParam;
    linearParam.hasBias = false;
    CREATE_OPERATION(linearParam, &qLinearNode.operation);
    qLinearNode.inTensorIds = {INTERMIDATE_INPUTNORMOUT, IN_Q_LINEARWEIGHT};
    qLinearNode.outTensorIds = {INTERMIDATE_Q_MIXEDLINEAROUT};

    CREATE_OPERATION(linearParam, &kLinearNode.operation);
    kLinearNode.inTensorIds = {INTERMIDATE_INPUTNORMOUT, IN_K_LINEARWEIGHT};
    kLinearNode.outTensorIds = {INTERMIDATE_K_MIXEDLINEAROUT};

    CREATE_OPERATION(linearParam, &vLinearNode.operation);
    vLinearNode.inTensorIds = {INTERMIDATE_INPUTNORMOUT, IN_V_LINEARWEIGHT};
    vLinearNode.outTensorIds = {INTERMIDATE_V_MIXEDLINEAROUT};

    atb::infer::RopeParam ropeParam;
    ropeParam.rotaryCoeff = ROTARY_COEFF; // 旋转系数
    CreateOperation(ropeParam, &ropeNode.operation);
    ropeNode.inTensorIds = {INTERMIDATE_Q_MIXEDLINEAROUT, INTERMIDATE_K_MIXEDLINEAROUT, IN_COS_EMBED, IN_SIN_EMBED,
                            IN_INPUT_LENGTHS};
    ropeNode.outTensorIds = {INTERMIDATE_Q_POSITIONEMBED, INTERMIDATE_K_POSITIONEMBED};

    atb::infer::ReshapeAndCacheParam reshapeCacheParm;
    CREATE_OPERATION(reshapeCacheParm, &reshapeAndCacheNode.operation);
    reshapeAndCacheNode.inTensorIds = {INTERMIDATE_K_POSITIONEMBED, INTERMIDATE_V_MIXEDLINEAROUT, IN_K_CACHE,
                                       IN_V_CACHE, IN_SLOTS};
    reshapeAndCacheNode.outTensorIds = {};
    reshapeAndCacheNode.inTensorReshapeFuncs.resize(reshapeAndCacheNode.inTensorIds.size());
    reshapeAndCacheNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        reshapeHeads(oldShape, newShape, param.headNum);
    };
    reshapeAndCacheNode.inTensorReshapeFuncs[1] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        reshapeHeads(oldShape, newShape, param.headNum);
    };

    if (param.isPrefill) {
        atb::infer::SelfAttentionParam selfAttentionParam;
        selfAttentionParam.headDim = param.dk;
        selfAttentionParam.headNum = param.headNum;
        selfAttentionParam.kvHeadNum = param.headNum;
        selfAttentionParam.qkScale = 1.0 / sqrt(param.dk);
        selfAttentionParam.isEncoder = true;
        CREATE_OPERATION(selfAttentionParam, &attentionNode.operation);
        attentionNode.inTensorIds = {INTERMIDATE_Q_POSITIONEMBED, INTERMIDATE_K_POSITIONEMBED,
                                     INTERMIDATE_V_MIXEDLINEAROUT, IN_ATTENTION_MASK, IN_INPUT_LENGTHS};
        attentionNode.outTensorIds = {INTERMIDATE_SELFOUT};
    } else {
        atb::infer::PagedAttentionParam pagedAttentionDeParam;
        pagedAttentionDeParam.headNum = param.headNum;
        pagedAttentionDeParam.qkScale = 1.0 / sqrt(param.dk);
        pagedAttentionDeParam.kvHeadNum = param.headNum;
        CREATE_OPERATION(pagedAttentionDeParam, &attentionNode.operation);
        attentionNode.inTensorIds = {INTERMIDATE_Q_POSITIONEMBED, IN_K_CACHE, IN_V_CACHE, IN_BLOCK_TABLES,
                                     IN_INPUT_LENGTHS};
        attentionNode.outTensorIds = {INTERMIDATE_SELFOUT};
        attentionNode.inTensorReshapeFuncs.resize(attentionNode.inTensorIds.size());
        attentionNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            reshapeHeads(oldShape, newShape, param.headNum);
        };
    }

    atb_speed::common::ParallelParam selfOutLinearParam;
    selfOutLinearParam.rank = param.rank;
    selfOutLinearParam.rankSize = param.rankSize;
    atb_speed::common::RowParallelLinear(selfOutLinearParam, &selfOutLinearNode.operation);
    selfOutLinearNode.inTensorIds = {INTERMIDATE_SELFOUT, IN_SELFOUTLINEARWEIGHT};
    selfOutLinearNode.outTensorIds = {INTERMIDATE_SELFLINEAROUT};
    if (!param.isPrefill) {
        selfOutLinearNode.inTensorReshapeFuncs.resize(selfOutLinearNode.inTensorIds.size());
        selfOutLinearNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            newShape.dimNum = 2;                                    // 2: dim num
            newShape.dims[0] = oldShape.dims[0];                    // 0: dim 0, n tokens
            newShape.dims[1] = oldShape.dims[1] * oldShape.dims[2]; // 1 hidden size: old 1, head num , old 2 head size
        };
    }

    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CREATE_OPERATION(addParam, &selfResidualAddNode.operation);
    selfResidualAddNode.inTensorIds = {IN_HIDDENSTATES, INTERMIDATE_SELFLINEAROUT};
    selfResidualAddNode.outTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT};

    CREATE_OPERATION(rmsNormParam, &selfNormNode.operation);
    selfNormNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT, IN_SELFOUTNORMWEIGHT};
    selfNormNode.outTensorIds = {INTERMIDATE_SELFNORMOUT};

    atb_speed::common::MlpGateParam mlpParam;
    mlpParam.rank = param.rank;
    mlpParam.rankSize = param.rankSize;
    mlpParam.activationType = atb::infer::ActivationType::ACTIVATION_SWISH;
    mlpParam.transposeB = true;
    mlpParam.isBias = false;
    mlpParam.isPack = false;
    atb_speed::common::MlpGateLayer(mlpParam, &mlpNode.operation);
    mlpNode.inTensorIds = {INTERMIDATE_SELFNORMOUT, IN_MLPUPWEIGHT, IN_MLPGATEWEIGHT, IN_MLPDOWNWEIGHT};
    mlpNode.outTensorIds = {INTERMIDATE_MLPOUT};

    CREATE_OPERATION(addParam, &mlpResidualAddNode.operation);
    mlpResidualAddNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT, INTERMIDATE_MLPOUT};
    mlpResidualAddNode.outTensorIds = {OUT_LAYEROUT};

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
    };
    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}

PagedAttentionLayerBinder::PagedAttentionLayerBinder() = default;

PagedAttentionLayerBinder::~PagedAttentionLayerBinder() = default;

void PagedAttentionLayerBinder::ParseParam(const nlohmann::json &paramJson)
{
    seqLen_.clear();
    for (const auto &item : paramJson["seqLen"]) {
        seqLen_.push_back(item.get<int>());
    }
}

void PagedAttentionLayerBinder::BindTensor(atb::VariantPack &variantPack)
{
    variantPack.inTensors.at(IN_INPUT_LENGTHS).hostData = seqLen_.data();
}

void from_json(const nlohmann::json &paramJson, PagedAttentionLayerParam &param)
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
    if (paramJson.contains("isPrefill")) {
        paramJson.at("isPrefill").get_to(param.isPrefill);
    }
    if (paramJson.contains("backend")) {
        paramJson.at("backend").get_to(param.backend);
    }
}

atb::Operation *CreatePagedAttentionLayer(const nlohmann::json &paramJson)
{
    ATB_LOG(INFO) << GetFuncNameAndNameSpace(__PRETTY_FUNCTION__);
    atb::Operation *op;
    atb_speed::internlm_20b::PagedAttentionLayer(paramJson.get<PagedAttentionLayerParam>(), &op);
    return op;
}
} // namespace internlm_20b
} // namespace atb_speed