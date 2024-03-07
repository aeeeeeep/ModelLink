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

#include "atb_speed/log.h"
#include "paged_attention_layer.h"
#include "layers/mlp_gate.h"
#include "layers/parallel_layer.h"

namespace atb_speed {
namespace chatglm2_6b {
static const uint64_t IN_TENSOR_COUNT = 16;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 19;
static const uint64_t NODE_COUNT = 17;

enum DecoderLayerTensorId : int {
    IN_HIDDENSTATES = 0,
    IN_NORMWEIGHT,
    IN_QKVMIXDWEIGHT,
    IN_QKVMIXDBIAS,
    IN_SELFOUTLINEARWEIGHT,
    IN_SELFOUTNORMWEIGHT,
    IN_MLPLINEARWEIGHTUP,
    IN_MLPLINEARWEIGHTDOWN,
    IN_COS,
    IN_SIN,
    IN_ATTENTION_MASK,
    IN_BLOCK_TABLES,
    IN_SLOTS,
    IN_INPUT_LENGTHS,
    IN_K_CACHE, // kvcache
    IN_V_CACHE,

    OUT_GLMLAYEROUT,

    INTERMIDATE_INPUTNORMOUT,
    INTERMIDATE_MIXEDLINEAROUTQKV,
    INTERMEDIATE_QLAYER,
    INTERMEDIATE_KVLAYER,
    INTERMEDIATE_KLAYER,
    INTERMEDIATE_QCHUNK0,
    INTERMEDIATE_QCHUNK1,
    INTERMEDIATE_KCHUNK0,
    INTERMEDIATE_KCHUNK1,
    INTERMEDIATE_QOUT,
    INTERMEDIATE_KOUT,
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
    newShape.dims[0] = oldShape.dims[0]; // 0 dim: n tokens
    newShape.dims[1] = headNum; // 1 dim: head num
    newShape.dims[2] = oldShape.dims[1] / headNum; // 1 dim: head size
}

void SqueezeLinearReshapeFuncPA(const atb::Dims &oldShape, atb::Dims &newShape)
{
    newShape.dimNum = 2;
    newShape.dims[0] = oldShape.dims[0];
    newShape.dims[1] = oldShape.dims[1] * oldShape.dims[2];
}

atb::Status DecoderPALayer(const LayerParamPa &param, atb::Operation **operation)
{
    if (param.headNum == 0) {
        ATB_LOG(ERROR) << "headNum is 0, please input a correct value";
        return atb::ERROR_INVALID_PARAM;
    }

    atb::GraphParam opGraph;
    opGraph.name = "DecoderPALayer";
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    auto &inputNormNode = opGraph.nodes.at(nodeId++);
    auto &mixdQkvLinearNode = opGraph.nodes.at(nodeId++);
    auto &sliceQNode = opGraph.nodes[nodeId++];
    auto &sliceKVNode = opGraph.nodes[nodeId++];
    auto &splitKVNode = opGraph.nodes[nodeId++];
    auto &splitQNode = opGraph.nodes[nodeId++];
    auto &splitKNode = opGraph.nodes[nodeId++];
    auto &ropeNode = opGraph.nodes[nodeId++];
    auto &cat1Node = opGraph.nodes[nodeId++];
    auto &cat3Node = opGraph.nodes[nodeId++];
    auto &reshapeAndCacheNode = opGraph.nodes.at(nodeId++);
    auto &attentionNode = opGraph.nodes.at(nodeId++);
    auto &selfOutLinearNode = opGraph.nodes.at(nodeId++);
    auto &selfResidualAddNode = opGraph.nodes.at(nodeId++);
    auto &selfNormNode = opGraph.nodes.at(nodeId++);
    auto &mlpNode = opGraph.nodes.at(nodeId++);
    auto &mlpResidualAddNode = opGraph.nodes.at(nodeId++);

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

    atb::infer::SliceParam sliceQNodeParam;
    sliceQNodeParam.offsets = {0, 0};
    sliceQNodeParam.size = {-1, param.numHeadsPerPartition * param.hiddenSizePerHead};
    CREATE_OPERATION(sliceQNodeParam, &sliceQNode.operation);
    sliceQNode.inTensorIds = {INTERMIDATE_MIXEDLINEAROUTQKV};
    sliceQNode.outTensorIds = {INTERMEDIATE_QLAYER};

    atb::infer::SliceParam sliceKVNodeParam;
    sliceKVNodeParam.offsets = {0, param.numHeadsPerPartition * param.hiddenSizePerHead};
    sliceKVNodeParam.size = {-1, param.numGroupsPerPartition * param.hiddenSizePerHead * 2};
    CREATE_OPERATION(sliceKVNodeParam, &sliceKVNode.operation);
    sliceKVNode.inTensorIds = {INTERMIDATE_MIXEDLINEAROUTQKV};
    sliceKVNode.outTensorIds = {INTERMEDIATE_KVLAYER};

    atb::infer::SplitParam splitKVParam;
    splitKVParam.splitDim = -1;
    splitKVParam.splitNum = 2; // 2 means half split
    CREATE_OPERATION(splitKVParam, &splitKVNode.operation);
    splitKVNode.inTensorIds = {INTERMEDIATE_KVLAYER};
    splitKVNode.outTensorIds = {INTERMEDIATE_KLAYER, INTERMIDATE_VALUE};

    atb::infer::SplitParam splitQParam;
    splitQParam.splitDim = -1;
    splitQParam.splitNum = 2; // 2 means half split
    CREATE_OPERATION(splitQParam, &splitQNode.operation);
    splitQNode.inTensorIds = {INTERMEDIATE_QLAYER};
    splitQNode.outTensorIds = {INTERMEDIATE_QCHUNK0, INTERMEDIATE_QCHUNK1};
    splitQNode.inTensorReshapeFuncs.resize(splitQNode.inTensorIds.size());
    splitQNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        reshapeHeads(oldShape, newShape, param.numHeadsPerPartition);
    };

    atb::infer::SplitParam splitKParam;
    splitKParam.splitDim = -1;
    splitKParam.splitNum = 2; // 2 means half split
    CREATE_OPERATION(splitKParam, &splitKNode.operation);
    splitKNode.inTensorIds = {INTERMEDIATE_KLAYER};
    splitKNode.outTensorIds = {INTERMEDIATE_KCHUNK0, INTERMEDIATE_KCHUNK1};
    splitKNode.inTensorReshapeFuncs.resize(splitKNode.inTensorIds.size());
    splitKNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        reshapeHeads(oldShape, newShape, param.numGroupsPerPartition);
    };

    atb::infer::RopeParam ropeParam;
    ropeParam.rotaryCoeff = param.hiddenSizePerHead / 2; // 2 means half rotary
    CREATE_OPERATION(ropeParam, &ropeNode.operation);
    ropeNode.inTensorIds = {INTERMEDIATE_QCHUNK0, INTERMEDIATE_KCHUNK0, IN_COS, IN_SIN, IN_INPUT_LENGTHS};
    ropeNode.outTensorIds = {INTERMEDIATE_QOUT, INTERMEDIATE_KOUT};
    ropeNode.inTensorReshapeFuncs.resize(ropeNode.inTensorIds.size());
    ropeNode.inTensorReshapeFuncs.at(0) = &SqueezeLinearReshapeFuncPA;
    ropeNode.inTensorReshapeFuncs.at(1) = &SqueezeLinearReshapeFuncPA;
    ropeNode.inTensorReshapeFuncs.at(2) = &SqueezeLinearReshapeFuncPA; // reshape No.2 input
    ropeNode.inTensorReshapeFuncs.at(3) = &SqueezeLinearReshapeFuncPA; // reshape No.3 input

    atb::infer::ConcatParam cat1Param;
    cat1Param.concatDim = -1;
    CREATE_OPERATION(cat1Param, &cat1Node.operation);
    cat1Node.inTensorIds = {INTERMEDIATE_QOUT, INTERMEDIATE_QCHUNK1};
    cat1Node.outTensorIds = {INTERMIDATE_POSITIONEMBEDQ};
    cat1Node.inTensorReshapeFuncs.resize(cat1Node.inTensorIds.size());
    cat1Node.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        reshapeHeads(oldShape, newShape, param.numHeadsPerPartition);
    };

    atb::infer::ConcatParam cat3Param;
    cat3Param.concatDim = -1;
    CREATE_OPERATION(cat3Param, &cat3Node.operation);
    cat3Node.inTensorIds = {INTERMEDIATE_KOUT, INTERMEDIATE_KCHUNK1};
    cat3Node.outTensorIds = {INTERMIDATE_POSITIONEMBEDK};
    cat3Node.inTensorReshapeFuncs.resize(cat3Node.inTensorIds.size());
    cat3Node.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        reshapeHeads(oldShape, newShape, param.numGroupsPerPartition);
    };

    atb::infer::ReshapeAndCacheParam reshapeCacheParm;
    CREATE_OPERATION(reshapeCacheParm, &reshapeAndCacheNode.operation);
    reshapeAndCacheNode.inTensorIds = {INTERMIDATE_POSITIONEMBEDK, INTERMIDATE_VALUE,
                                        IN_K_CACHE, IN_V_CACHE, IN_SLOTS};
    reshapeAndCacheNode.outTensorIds = {};
    reshapeAndCacheNode.inTensorReshapeFuncs.resize(reshapeAndCacheNode.inTensorIds.size());
    reshapeAndCacheNode.inTensorReshapeFuncs[1] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        reshapeHeads(oldShape, newShape, param.numGroupsPerPartition);
    };

    if (param.isPrefill) {
        atb::infer::SelfAttentionParam faEnParam;
        faEnParam.headNum = param.headNum;
        faEnParam.qkScale = 1.0 / sqrt(param.dk);
        faEnParam.kvHeadNum = param.numGroupsPerPartition;
        faEnParam.isEncoder = true;
        CREATE_OPERATION(faEnParam, &attentionNode.operation);
        attentionNode.inTensorIds = {INTERMIDATE_POSITIONEMBEDQ, INTERMIDATE_POSITIONEMBEDK, INTERMIDATE_VALUE,
                                     IN_ATTENTION_MASK, IN_INPUT_LENGTHS};
        attentionNode.outTensorIds = {INTERMIDATE_SELFOUT};
        attentionNode.inTensorReshapeFuncs.resize(attentionNode.inTensorIds.size());
        attentionNode.inTensorReshapeFuncs[2] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            reshapeHeads(oldShape, newShape, faEnParam.kvHeadNum);
        };
    } else {
        atb::infer::PagedAttentionParam paDeParam;
        paDeParam.headNum = param.headNum;
        paDeParam.qkScale = 1.0 / sqrt(param.dk);
        paDeParam.kvHeadNum = param.numGroupsPerPartition;
        CREATE_OPERATION(paDeParam, &attentionNode.operation);
        attentionNode.inTensorIds = {INTERMIDATE_POSITIONEMBEDQ, IN_K_CACHE, IN_V_CACHE,
                                     IN_BLOCK_TABLES, IN_INPUT_LENGTHS};
        attentionNode.outTensorIds = {INTERMIDATE_SELFOUT};
        attentionNode.inTensorReshapeFuncs.resize(attentionNode.inTensorIds.size());
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
        newShape.dimNum = 2;  // dimNum is 2
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = oldShape.dims[1] * oldShape.dims[2]; // merge last 2 dims
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
    mlpParam.transposeB = true;
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

