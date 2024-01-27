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
#include "layers/mlp_gate.h"
#include "layers/parallel_layer.h"
#include "layers/plugin_op/w8a16_operation.h"
#include "models/contrib/operation/linear_parallel_w8a16.h"
#include "models/contrib/operation/mlp_w8a16.h"

namespace atb_speed {
namespace contrib {

static const uint64_t IN_TENSOR_COUNT = 32;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 11;
static const uint64_t NODE_COUNT = 12;
static uint64_t DIM3 = 3;

void reshapeHead(const atb::Dims &oldShape, atb::Dims &newShape, int headNum)
{
    newShape.dimNum = DIM3;
    newShape.dims[0] = oldShape.dims[0];           // 0 dim: n tokens
    newShape.dims[1] = headNum;                    // 1 dim: head num
    newShape.dims[2] = oldShape.dims[1] / headNum; // 2 dim: head size
}

atb::Status PagedAttentionLayer(const PagedAttentionLayerParam &param, atb::Operation **operation)
{
    std::shared_ptr<int64_t> batchNumPtr = std::make_shared<int64_t>(0);

    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);
    if (param.isPrefill) {
        opGraph.name = "PagedAttentionLayer_Prefill";
    } else {
        opGraph.name = "PagedAttentionLayer_Decode";
    }

    size_t nodeId = 0;
    atb::Node &inputNormNode = opGraph.nodes.at(nodeId++);
    atb::Node &mixdQLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &mixdKLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &mixdVLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &ropeNode = opGraph.nodes.at(nodeId++);
    atb::Node &reshapeAndCacheNode = opGraph.nodes.at(nodeId++);
    atb::Node &attentionNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfOutLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfResidualAddNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfNormNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpParallelNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpResidualAddNode = opGraph.nodes.at(nodeId++);

    atb::infer::RmsNormParam rmsNormParam;
    rmsNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    rmsNormParam.normParam.epsilon = param.rmsNormEps;
    CREATE_OPERATION(rmsNormParam, &inputNormNode.operation);
    inputNormNode.inTensorIds = {IN_HIDDENSTATES, IN_NORMWEIGHT};
    inputNormNode.outTensorIds = {INTERMIDATE_INPUTNORMOUT};

    mixdQLinearNode.operation = new atb_speed::common::W8A16Operation("mixdQLinearNode");
    mixdQLinearNode.inTensorIds = {INTERMIDATE_INPUTNORMOUT, IN_QMIXDWEIGHT, IN_Q_SCALE, IN_Q_OFFSET};
    mixdQLinearNode.outTensorIds = {INTERMIDATE_MIXEDQ};

    mixdKLinearNode.operation = new atb_speed::common::W8A16Operation("mixdKLinearNode");
    mixdKLinearNode.inTensorIds = {INTERMIDATE_INPUTNORMOUT, IN_KMIXDWEIGHT, IN_K_SCALE, IN_K_OFFSET};
    mixdKLinearNode.outTensorIds = {INTERMIDATE_MIXEDK};

    mixdVLinearNode.operation = new atb_speed::common::W8A16Operation("mixdVLinearNode");
    mixdVLinearNode.inTensorIds = {INTERMIDATE_INPUTNORMOUT, IN_VMIXDWEIGHT, IN_V_SCALE, IN_V_OFFSET};
    mixdVLinearNode.outTensorIds = {INTERMIDATE_MIXEDV};

    atb::infer::RopeParam ropeparam;
    ropeparam.rotaryCoeff = param.rotaryCoeff;
    CREATE_OPERATION(ropeparam, &ropeNode.operation);
    ropeNode.inTensorIds = {INTERMIDATE_MIXEDQ, INTERMIDATE_MIXEDK,
                            IN_COSTABLE, IN_SINTABLE, IN_SEQLEN};
    ropeNode.outTensorIds = {INTERMIDATE_POSITIONEMBEDQ, INTERMIDATE_POSITIONEMBEDK};

    atb::infer::ReshapeAndCacheParam reshapeCacheParm;
    CREATE_OPERATION(reshapeCacheParm, &reshapeAndCacheNode.operation);
    reshapeAndCacheNode.inTensorIds = {INTERMIDATE_POSITIONEMBEDK, INTERMIDATE_MIXEDV, IN_K_CACHE, IN_V_CACHE,
                                       IN_SLOTS};
    reshapeAndCacheNode.outTensorIds = {};
    reshapeAndCacheNode.inTensorReshapeFuncs.resize(reshapeAndCacheNode.inTensorIds.size());
    reshapeAndCacheNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        reshapeHead(oldShape, newShape, param.numHeadsPerPartition);
    };
    reshapeAndCacheNode.inTensorReshapeFuncs[1] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        reshapeHead(oldShape, newShape, param.numHeadsPerPartition);
    };

    if (param.isPrefill) {
        atb::infer::SelfAttentionParam faEnParam;
        faEnParam.headNum = param.headNum;
        faEnParam.kvHeadNum = param.numHeadsPerPartition;
        faEnParam.headDim = param.dk;
        faEnParam.qkScale = 1.0 / sqrt(param.dk);
        faEnParam.isEncoder = true;
        CREATE_OPERATION(faEnParam, &attentionNode.operation);

        attentionNode.inTensorIds = {INTERMIDATE_POSITIONEMBEDQ, INTERMIDATE_POSITIONEMBEDK, INTERMIDATE_MIXEDV,
            IN_ATTENTIONMASK, IN_SEQLEN};
        attentionNode.outTensorIds = {INTERMIDATE_ATTENTIONOUT};
        attentionNode.inTensorReshapeFuncs.resize(attentionNode.inTensorIds.size());
        attentionNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            reshapeHead(oldShape, newShape, param.headNum);
        };
        attentionNode.inTensorReshapeFuncs[1] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            reshapeHead(oldShape, newShape, param.numHeadsPerPartition);
        };
        attentionNode.inTensorReshapeFuncs[2] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            reshapeHead(oldShape, newShape, param.numHeadsPerPartition);
        };
    } else {
        atb::infer::PagedAttentionParam paDeParam;
        paDeParam.headNum = param.headNum;
        paDeParam.qkScale = 1.0 / sqrt(param.dk);
        paDeParam.kvHeadNum = param.numHeadsPerPartition;
        paDeParam.isSupportAlibi = param.isBF16;
        if (param.isBF16) {
            paDeParam.maskType = atb::infer::PagedAttentionParam::MaskType::MASK_TYPE_ALIBI;
            attentionNode.inTensorIds = {INTERMIDATE_POSITIONEMBEDQ, IN_K_CACHE, IN_V_CACHE, IN_BLOCK_TABLES,
                                         IN_SEQLEN, IN_ATTENTIONMASK};
        } else {
            attentionNode.inTensorIds = {INTERMIDATE_POSITIONEMBEDQ, IN_K_CACHE, IN_V_CACHE, IN_BLOCK_TABLES,
                                         IN_SEQLEN};
        }
        CREATE_OPERATION(paDeParam, &attentionNode.operation);
        attentionNode.outTensorIds = {INTERMIDATE_ATTENTIONOUT};
        attentionNode.inTensorReshapeFuncs.resize(attentionNode.inTensorIds.size());
        attentionNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            reshapeHead(oldShape, newShape, param.headNum);
        };
    }

    atb_speed::contrib::LinearParallelW8A16Param selfOutLinearParam;
    selfOutLinearParam.transWeight = true;
    selfOutLinearParam.rank = param.rank;
    selfOutLinearParam.rankSize = param.rankSize;
    selfOutLinearParam.rankRoot = 0;
    selfOutLinearParam.bias = "None";
    selfOutLinearParam.parallelType = "RowParallel";
    selfOutLinearParam.backend = param.backend;
    CreateLinearParallelW8A16(selfOutLinearParam, &selfOutLinearNode.operation);
    selfOutLinearNode.inTensorIds = {INTERMIDATE_ATTENTIONOUT, IN_SELFOUTLINEARWEIGHT,
                                             IN_SELFOUTLINEAR_SCALE, IN_SELFOUTLINEAR_OFFSET};
    selfOutLinearNode.outTensorIds = {INTERMIDATE_SELFLINEAROUT};
    selfOutLinearNode.inTensorReshapeFuncs.resize(selfOutLinearNode.inTensorIds.size());
    selfOutLinearNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2; // dimNum is 2
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = oldShape.dims[1] * oldShape.dims[2]; // dimNum is 2
    };

    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CREATE_OPERATION(addParam, &selfResidualAddNode.operation);
    selfResidualAddNode.inTensorIds = {IN_HIDDENSTATES, INTERMIDATE_SELFLINEAROUT};
    selfResidualAddNode.outTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT};

    CREATE_OPERATION(rmsNormParam, &selfNormNode.operation);
    selfNormNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT, IN_SELFOUTNORMWEIGHT};
    selfNormNode.outTensorIds = {INTERMIDATE_SELFNORMOUT};

    atb_speed::contrib::MlpW8A16Param mlpW8A16Param;
    mlpW8A16Param.rank = param.rank;
    mlpW8A16Param.rankSize = param.rankSize;
    mlpW8A16Param.rankRoot = 0;
    mlpW8A16Param.transposeB = true;
    mlpW8A16Param.hcclComm = nullptr;
    mlpW8A16Param.backend = param.backend;
    mlpW8A16Param.activationType = atb::infer::ActivationType::ACTIVATION_SWISH;
    CreateMlpW8A16Operation(mlpW8A16Param, &mlpParallelNode.operation);
    mlpParallelNode.inTensorIds = {
        INTERMIDATE_SELFNORMOUT, IN_MLPUPWEIGHT,   IN_MLPUP_SCALE,     IN_MLPUP_OFFSET,
        IN_MLPGATEWEIGHT,        IN_MLPGATE_SCALE,  IN_MLPGATE_OFFSET, IN_MLPDOWNWEIGHT,
        IN_MLPDOWN_SCALE,        IN_MLPDOWN_OFFSET};
    mlpParallelNode.outTensorIds = {INTERMIDATE_MLPOUT};

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

PagedAttentionLayerBinder::PagedAttentionLayerBinder() {}

PagedAttentionLayerBinder::~PagedAttentionLayerBinder() {}

void PagedAttentionLayerBinder::ParseParam(const nlohmann::json &paramJson)
{
    seqLen_.clear();
    for (auto item : paramJson["seqLen"]) {
        seqLen_.push_back(item.get<int>());
    }
}

void PagedAttentionLayerBinder::BindTensor(atb::VariantPack &variantPack)
{
    ATB_LOG(INFO) << "enter PagedAttentionLayer BindTensor";
    const uint32_t seqLenTensorId = IN_SEQLEN;
    variantPack.inTensors.at(seqLenTensorId).hostData = seqLen_.data();
}

} // namespace contrib
} // namespace atb_speed