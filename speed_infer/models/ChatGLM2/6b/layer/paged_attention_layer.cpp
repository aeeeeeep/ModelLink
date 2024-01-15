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

#include "decoder_pa_layer.h"
#include "chatglm2/6b/operation/position_embedding_tgi.h"
#include "chatglm2/6b/operation/self_attention.h"
#include "layers/mlp_gate.h"
#include "layers/parallel_layer.h"

namespace atb_speed {
namespace chatglm2_6b {
static const uint64_t IN_TENSOR_COUNT = 17;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 8 + 2;
static const uint64_t NODE_COUNT = 8 + 2;

enum DecoderLayerTensorId : int {
    IN_HIDDENSTATES = 0,
    IN_NORMWEIGHT,
    IN_QKVMIXDWEIGHT,
    IN_QKVMIXDBIAS,
    IN_SELFOUTLINEARWEIGHT,
    IN_SELFOUTNORMWEIGHT,
    IN_MLPLINEARWEIGHTUP,
    IN_MLPLINEARWEIGHTDOWN,
    IN_ROPECACHE,
    IN_ATTENTION_MASK,
    IN_BLOCK_TABLES,
    IN_SLOTS,
    IN_INPUT_LENGTHS,
    IN_MAX_SEQLEN,
    IN_CURRENT_SEQLEN_PRE,
    IN_PASTKEY,
    IN_PASTVALUE,

    OUT_GLMLAYEROUT,

    INTERMIDATE_INPUTNORMOUT,
    INTERMIDATE_MIXEDLINEAROUTQKV,
    INTERMIDATE_POSITIONEMBEDQ,
    INTERMIDATE_POSITIONEMBEDK,
    INTERMIDATE_VALUE,
    INTERMIDATE_SELFOUT,
    INTERMIDATE_SELFLINEAROUT,
    INTERMIDATE_SELFRESIDUALADDOUT,
    INTERMIDATE_SELFNORMOUT,
    INTERMIDATE_MLPOUT,
};


void reshapeHeads(const atb::Dims &oldShape, atb::Dims &newShape, int headNum)
{
    newShape.dimNum = 3; // dimNum: 3
    newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1]; // 0 dim: n tokens
    newShape.dims[1] = headNum;  // 1 dim: head num
    newShape.dims[2] = oldShape.dims[3] * oldShape.dims[2] / headNum;  // 1 dim: head size
}

void SqueezeLinearReshapeFuncPA(const atb::Dims &oldShape, atb::Dims &newShape)
{
    newShape.dimNum = oldShape.dimNum;
    newShape.dims[0] = 1;
    newShape.dims[1] = oldShape.dims[0];
    newShape.dims[2] = oldShape.dims[1] * oldShape.dims[2];
}

atb::Status DecoderPALayer(const LayerParamPa &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.name = "DecoderPALayer";
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    atb::Node &inputNormNode = opGraph.nodes.at(nodeId++);
    atb::Node &mixdQkvLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &positionEmbeddingNode = opGraph.nodes.at(nodeId++);
    atb::Node &reshapeAndCacheNode = opGraph.nodes.at(nodeId++);
    atb::Node &attentionNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfOutLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfResidualAddNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfNormNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpResidualAddNode = opGraph.nodes.at(nodeId++);

    atb::infer::RmsNormParam inputNormParam;
    inputNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    inputNormParam.normParam.epsilon = param.rmsNormEps;
    CREATE_OPERATION(inputNormParam, &inputNormNode.operation);
    inputNormNode.inTensorIds = {IN_HIDDENSTATES, IN_NORMWEIGHT};
    inputNormNode.outTensorIds = {INTERMIDATE_INPUTNORMOUT};

    atb::infer::LinearParam mixdQkvLinearParam;
    CREATE_OPERATION(mixdQkvLinearParam, &mixdQkvLinearNode.operation);
    mixdQkvLinearNode.inTensorIds = {INTERMIDATE_INPUTNORMOUT, IN_QKVMIXDWEIGHT, IN_QKVMIXDBIAS};
    mixdQkvLinearNode.outTensorIds = {INTERMIDATE_MIXEDLINEAROUTQKV};

    atb_speed::chatglm2_6b::PositionEmbeddingParamTgi PositionEmbeddingParamTgi;
    PositionEmbeddingParamTgi.numHeadsPerPartition = param.numHeadsPerPartition;
    PositionEmbeddingParamTgi.hiddenSizePerHead = param.hiddenSizePerHead;
    PositionEmbeddingParamTgi.numGroupsPerPartition = param.numGroupsPerPartition;
    atb_speed::chatglm2_6b::PositionEmbeddingTgi(PositionEmbeddingParamTgi, &positionEmbeddingNode.operation);
    positionEmbeddingNode.inTensorIds = {INTERMIDATE_MIXEDLINEAROUTQKV, IN_ROPECACHE};
    positionEmbeddingNode.outTensorIds = {INTERMIDATE_POSITIONEMBEDQ, INTERMIDATE_POSITIONEMBEDK, INTERMIDATE_VALUE};

    atb::infer::ReshapeAndCacheParam reshapeCacheParm;
    CREATE_OPERATION(reshapeCacheParm, &reshapeAndCacheNode.operation);
    reshapeAndCacheNode.inTensorIds = {INTERMIDATE_POSITIONEMBEDK, INTERMIDATE_VALUE,
                                        IN_PASTKEY, IN_PASTVALUE, IN_SLOTS};
    reshapeAndCacheNode.outTensorIds = {};
    reshapeAndCacheNode.inTensorReshapeFuncs.resize(reshapeAndCacheNode.inTensorIds.size());
    reshapeAndCacheNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        reshapeHeads(oldShape, newShape, param.numGroupsPerPartition);
    };
    reshapeAndCacheNode.inTensorReshapeFuncs[1] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        reshapeHeads(oldShape, newShape, param.numGroupsPerPartition);
    };

    if (param.isPrefill) {
        atb::infer::SelfAttentionParam faEnParam;
        faEnParam.headNum = param.headNum;
        faEnParam.qkScale = 1.0 / sqrt(param.dk);
        faEnParam.kvHeadNum = param.numGroupsPerPartition;
        faEnParam.isEncoder = true;
        faEnParam.isFusion = true;
        CREATE_OPERATION(faEnParam, &attentionNode.operation);
        attentionNode.inTensorIds = {INTERMIDATE_POSITIONEMBEDQ, INTERMIDATE_POSITIONEMBEDK, INTERMIDATE_VALUE,
                                     IN_ATTENTION_MASK, IN_INPUT_LENGTHS};
        attentionNode.outTensorIds = {INTERMIDATE_SELFOUT};
        attentionNode.inTensorReshapeFuncs.resize(attentionNode.inTensorIds.size());
        attentionNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            reshapeHeads(oldShape, newShape, faEnParam.headNum);
        };
        attentionNode.inTensorReshapeFuncs[1] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            reshapeHeads(oldShape, newShape, faEnParam.kvHeadNum);
        };
        attentionNode.inTensorReshapeFuncs[2] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            reshapeHeads(oldShape, newShape, faEnParam.kvHeadNum);
        };
    } else {
        atb::infer::PagedAttentionParam paDeParam;
        paDeParam.headNum = param.headNum;
        paDeParam.qkScale = 1.0 / sqrt(param.dk);
        paDeParam.kvHeadNum = param.numGroupsPerPartition;
        CREATE_OPERATION(paDeParam, &attentionNode.operation);
        attentionNode.inTensorIds = {INTERMIDATE_POSITIONEMBEDQ, IN_PASTKEY, IN_PASTVALUE,
                                     IN_BLOCK_TABLES, IN_INPUT_LENGTHS};
        attentionNode.outTensorIds = {INTERMIDATE_SELFOUT};
        attentionNode.inTensorReshapeFuncs.resize(attentionNode.inTensorIds.size());
        attentionNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            reshapeHeads(oldShape, newShape, paDeParam.headNum);
        };
    }

    atb_speed::common::ParallelParam selfOutLinearParam;
    selfOutLinearParam.rank = param.rank;
    selfOutLinearParam.rankSize = param.rankSize;
    selfOutLinearParam.isBias = false;
    atb_speed::common::RowParallelLinear(selfOutLinearParam, &selfOutLinearNode.operation);
    selfOutLinearNode.inTensorIds = {INTERMIDATE_SELFOUT, IN_SELFOUTLINEARWEIGHT};
    selfOutLinearNode.outTensorIds = {INTERMIDATE_SELFLINEAROUT};
    selfOutLinearNode.inTensorReshapeFuncs.resize(selfOutLinearNode.inTensorIds.size());
    selfOutLinearNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 3;  // dimNum is 3
        newShape.dims[0] = 1;
        newShape.dims[1] = oldShape.dims[0];
        newShape.dims[2] = oldShape.dims[1] * oldShape.dims[2];
    };

    atb::infer::ElewiseParam AddParam;
    AddParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CREATE_OPERATION(AddParam, &selfResidualAddNode.operation);
    selfResidualAddNode.inTensorIds = {IN_HIDDENSTATES, INTERMIDATE_SELFLINEAROUT};
    selfResidualAddNode.outTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT};

    atb::infer::RmsNormParam selfNormParam;
    selfNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    selfNormParam.normParam.epsilon = param.rmsNormEps;
    CREATE_OPERATION(selfNormParam, &selfNormNode.operation);
    selfNormNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT, IN_SELFOUTNORMWEIGHT};
    selfNormNode.outTensorIds = {INTERMIDATE_SELFNORMOUT};

    atb_speed::common::MlpGateParam mlpParam;
    mlpParam.rank = param.rank;
    mlpParam.rankSize = param.rankSize;
    mlpParam.activationType = atb::infer::ActivationType::ACTIVATION_SWISH;
    mlpParam.transposeB = false;
    mlpParam.isBias = false;
    mlpParam.isPack = true;
    atb_speed::common::MlpGateLayer(mlpParam, &mlpNode.operation);
    mlpNode.inTensorIds = {INTERMIDATE_SELFNORMOUT, IN_MLPLINEARWEIGHTUP, IN_MLPLINEARWEIGHTDOWN};
    mlpNode.outTensorIds = {INTERMIDATE_MLPOUT};

    atb::infer::ElewiseParam Add2Param;
    Add2Param.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CREATE_OPERATION(Add2Param, &mlpResidualAddNode.operation);
    mlpResidualAddNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT, INTERMIDATE_MLPOUT};
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

FlashAttentionHostBinder::FlashAttentionHostBinder() {}

FlashAttentionHostBinder::~FlashAttentionHostBinder() {}

void FlashAttentionHostBinder::ParseParam(const nlohmann::json &paramJson)
{
    seqLen_.clear();
    for (auto item : paramJson["seqLen"]) {
        seqLen_.push_back(item.get<int32_t>());
    }
}

void FlashAttentionHostBinder::BindTensor(atb::VariantPack &variantPack)
{
    variantPack.inTensors.at(IN_INPUT_LENGTHS).hostData = seqLen_.data();
}
} // namespace chatglm2_6b
} // namespace atb_speed

