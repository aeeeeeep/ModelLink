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
#include "fusion_pa_layer_w8a8.h"
#include "layers/mlp_gate.h"
#include "layers/parallel_layer.h"
#include "models/llama2/70b/operation/linear_parallel_w8a8.h"
#include "models/llama2/70b/operation/linear_w8a8.h"
#include "models/llama2/70b/operation/mlp_w8a8.h"
#include "models/llama2/70b/operation/position_embedding_fusion.h"
#include "models/llama2/70b/operation/self_attention.h"

namespace atb_speed {
namespace llama2_70b {

static const uint64_t IN_TENSOR_COUNT = 39;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 11;
static const uint64_t NODE_COUNT = 12;

void reshape(const atb::Dims &oldShape, atb::Dims &newShape, int headNum)
{
    newShape.dimNum = 3;                           // dimNum: 3
    newShape.dims[0] = oldShape.dims[0];           // 0 dim: n tokens
    newShape.dims[1] = headNum;                    // 1 dim: head num
    newShape.dims[2] = oldShape.dims[1] / headNum; // 2 dim: head size
}

atb::Status FusionPALayerW8A8(const FusionPALayerW8A8Param &param, atb::Operation **operation)
{
    std::shared_ptr<int64_t> batchNumPtr = std::make_shared<int64_t>(0);

    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);
    if (param.isPrefill) {
        opGraph.name = "fusion_pa_prefill_layer_w8a8";
    } else {
        opGraph.name = "fusion_pa_decoder_layer_w8a8";
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
    atb::Node &mlpParallelNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpResidualAddNode = opGraph.nodes.at(nodeId++);

    atb::infer::RmsNormParam rmsNormParam;
    rmsNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    rmsNormParam.normParam.epsilon = param.rmsNormEps;
    CreateOperation(rmsNormParam, &inputNormNode.operation);
    inputNormNode.inTensorIds = {IN_HIDDENSTATES, IN_NORMWEIGHT};
    inputNormNode.outTensorIds = {INTERMIDATE_INPUTNORMOUT};

    atb_speed::llama2_70b::LinearW8A8Param linearParam = {true};
    CreateLinearW8A8(linearParam, &qLinearNode.operation, atb_speed::llama2_70b::COLUMN_PARALLEL);
    qLinearNode.inTensorIds = {INTERMIDATE_INPUTNORMOUT, IN_QWEIGHT, IN_QSCALE, IN_QOFFSET, IN_QDESCALE};
    qLinearNode.outTensorIds = {INTERMIDATE_Q};

    CreateLinearW8A8(linearParam, &kLinearNode.operation, atb_speed::llama2_70b::COLUMN_PARALLEL);
    kLinearNode.inTensorIds = {INTERMIDATE_INPUTNORMOUT, IN_KWEIGHT, IN_KSCALE, IN_KOFFSET, IN_KDESCALE};
    kLinearNode.outTensorIds = {INTERMIDATE_K};
    kLinearNode.inTensorReshapeFuncs.resize(kLinearNode.inTensorIds.size());

    CreateLinearW8A8(linearParam, &vLinearNode.operation, atb_speed::llama2_70b::COLUMN_PARALLEL);
    vLinearNode.inTensorIds = {INTERMIDATE_INPUTNORMOUT, IN_VWEIGHT, IN_VSCALE, IN_VOFFSET, IN_VDESCALE};
    vLinearNode.outTensorIds = {INTERMIDATE_V};

    atb::infer::RopeParam ropeparam;
    ropeparam.rotaryCoeff = param.rotaryCoeff;
    CreateOperation(ropeparam, &ropeNode.operation);
    ropeNode.inTensorIds = {INTERMIDATE_Q, INTERMIDATE_K, IN_COSTABLE, IN_SINTABLE, IN_SEQLEN};
    ropeNode.outTensorIds = {INTERMIDATE_POSITIONEMBEDQ, INTERMIDATE_POSITIONEMBEDK};

    atb::infer::ReshapeAndCacheParam reshapeCacheParm;
    CreateOperation(reshapeCacheParm, &reshapeAndCacheNode.operation);
    reshapeAndCacheNode.inTensorIds = {INTERMIDATE_POSITIONEMBEDK, INTERMIDATE_V, IN_K_CACHE, IN_V_CACHE, IN_SLOTS};
    reshapeAndCacheNode.outTensorIds = {};
    reshapeAndCacheNode.inTensorReshapeFuncs.resize(reshapeAndCacheNode.inTensorIds.size());
    reshapeAndCacheNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        reshape(oldShape, newShape, param.numHeadsPerPartition);
    };
    reshapeAndCacheNode.inTensorReshapeFuncs[1] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        reshape(oldShape, newShape, param.numHeadsPerPartition);
    };

    if (param.isPrefill) {
        atb::infer::SelfAttentionParam faEnParam;
        faEnParam.headNum = param.headNum;
        faEnParam.kvHeadNum = param.numHeadsPerPartition;
        faEnParam.headDim = param.dk;
        faEnParam.qkScale = 1.0 / sqrt(param.dk);
        faEnParam.isEncoder = true;
        CreateOperation(faEnParam, &attentionNode.operation);

        attentionNode.inTensorIds = {INTERMIDATE_POSITIONEMBEDQ, INTERMIDATE_POSITIONEMBEDK, INTERMIDATE_V,
                                     IN_ATTENTIONMASK, IN_SEQLEN};
        attentionNode.outTensorIds = {INTERMIDATE_ATTENTIONOUT};
        attentionNode.inTensorReshapeFuncs.resize(attentionNode.inTensorIds.size());
        attentionNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            reshape(oldShape, newShape, param.headNum);
        };
        attentionNode.inTensorReshapeFuncs[1] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            reshape(oldShape, newShape, param.numHeadsPerPartition);
        };
        attentionNode.inTensorReshapeFuncs[2] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            reshape(oldShape, newShape, param.numHeadsPerPartition);
        };
    } else {
        atb::infer::PagedAttentionParam paDeParam;
        paDeParam.headNum = param.headNum;
        paDeParam.qkScale = 1.0 / sqrt(param.dk);
        paDeParam.kvHeadNum = param.numHeadsPerPartition;
        CreateOperation(paDeParam, &attentionNode.operation);
        attentionNode.inTensorIds = {INTERMIDATE_POSITIONEMBEDQ, IN_K_CACHE, IN_V_CACHE, IN_BLOCK_TABLES, IN_SEQLEN};
        attentionNode.outTensorIds = {INTERMIDATE_ATTENTIONOUT};
        attentionNode.inTensorReshapeFuncs.resize(attentionNode.inTensorIds.size());
        attentionNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            reshape(oldShape, newShape, param.headNum);
        };
    }

    atb_speed::llama2_70b::LinearParallelW8A8Param selfOutLinearParam;
    selfOutLinearParam.transWeight = param.transposedWeight;
    selfOutLinearParam.rank = param.rank;
    selfOutLinearParam.rankSize = param.rankSize;
    selfOutLinearParam.rankRoot = 0;
    selfOutLinearParam.bias = "None";
    selfOutLinearParam.parallelType = "RowParallel";
    selfOutLinearParam.backend = param.backend;
    CreateLinearParallelW8A8(selfOutLinearParam, &selfOutLinearNode.operation);
    selfOutLinearNode.inTensorIds = {INTERMIDATE_ATTENTIONOUT, IN_OWEIGHT, IN_OSCALE, IN_OOFFSET, IN_ODESCALE};
    selfOutLinearNode.outTensorIds = {INTERMIDATE_SELFLINEAROUT};
    selfOutLinearNode.inTensorReshapeFuncs.resize(selfOutLinearNode.inTensorIds.size());
    selfOutLinearNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2; // dimNum is 2
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = oldShape.dims[1] * oldShape.dims[2]; // dimNum is 2
    };

    // [1, 20, 8192]
    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CreateOperation(addParam, &selfResidualAddNode.operation);
    selfResidualAddNode.inTensorIds = {IN_HIDDENSTATES, INTERMIDATE_SELFLINEAROUT};
    selfResidualAddNode.outTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT};

    CreateOperation(rmsNormParam, &selfNormNode.operation);
    selfNormNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT, IN_SELFOUTNORMWEIGHT};
    selfNormNode.outTensorIds = {INTERMIDATE_SELFNORMOUT};

    atb_speed::llama2_70b::MlpW8A8Param mlpW8A8Param;
    mlpW8A8Param.rank = param.rank;
    mlpW8A8Param.rankSize = param.rankSize;
    mlpW8A8Param.rankRoot = 0;
    mlpW8A8Param.transposeB = false;
    mlpW8A8Param.hcclComm = nullptr;
    mlpW8A8Param.backend = param.backend;
    mlpW8A8Param.activationType = atb::infer::ActivationType::ACTIVATION_SWISH;
    CreateMlpW8A8Operation(mlpW8A8Param, &mlpParallelNode.operation);
    mlpParallelNode.inTensorIds = {
        INTERMIDATE_SELFNORMOUT, IN_MLPUPWEIGHT,   IN_MLPUPSCALE,     IN_MLPUPOFFSET,    IN_MLPUPDESCALE,
        IN_MLPGATEWEIGHT,        IN_MLPGATESCALE,  IN_MLPGATEOFFSET,  IN_MLPGATEDESCALE, IN_MLPDOWNWEIGHT,
        IN_MLPDOWNSCALE,         IN_MLPDOWNOFFSET, IN_MLPDOWNDESCALE,
    };
    mlpParallelNode.outTensorIds = {INTERMIDATE_MLPOUT};

    CreateOperation(addParam, &mlpResidualAddNode.operation);
    mlpResidualAddNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT, INTERMIDATE_MLPOUT};
    mlpResidualAddNode.outTensorIds = {OUT_LLAMA70BLAYEROUT};

    // [1,20,8192] [1, 20, 2, 128] [1, 20, 2, 128]
    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
    };

    atb::CreateOperation(opGraph, operation);
    return atb::NO_ERROR;
}

FusionPALayerW8A8Binder::FusionPALayerW8A8Binder() {}

FusionPALayerW8A8Binder::~FusionPALayerW8A8Binder() {}

void FusionPALayerW8A8Binder::ParseParam(const nlohmann::json &paramJson)
{
    seqLen_.clear();
    for (auto item : paramJson["seqLen"]) {
        seqLen_.push_back(item.get<int>());
    }
}

void FusionPALayerW8A8Binder::BindTensor(atb::VariantPack &variantPack)
{
    ATB_LOG(INFO) << "enter FusionPALayerW8A8Operation BindTensor";
    const uint32_t seqLenTensorId = IN_SEQLEN;
    variantPack.inTensors.at(seqLenTensorId).hostData = seqLen_.data();
}

} // namespace llama2_70b
} // namespace atb_speed