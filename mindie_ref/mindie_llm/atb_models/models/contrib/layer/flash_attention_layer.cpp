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
#include "flash_attention_layer.h"
#include "layers/mlp_gate.h"
#include "layers/plugin_op/w8a16_operation.h"
#include "models/contrib/operation/linear_parallel_w8a16.h"
#include "models/contrib/operation/mlp_w8a16.h"

namespace atb_speed {
namespace contrib {
enum FlashAttentionLayerTensorId : int {
    IN_HIDDENSTATES = 0,
    IN_NORMWEIGHT,
    IN_QMIXDWEIGHT,
    IN_KMIXDWEIGHT,
    IN_VMIXDWEIGHT,
    IN_SELFOUTLINEARWEIGHT,
    IN_SELFOUTNORMWEIGHT,
    IN_MLPGATEWEIGHT,
    IN_MLPUPWEIGHT,
    IN_MLPDOWNWEIGHT,
    IN_Q_SCALE,
    IN_Q_OFFSET,
    IN_K_SCALE,
    IN_K_OFFSET,
    IN_V_SCALE,
    IN_V_OFFSET,
    IN_SELFOUTLINEAR_SCALE,
    IN_SELFOUTLINEAR_OFFSET,
    IN_MLPGATE_SCALE,
    IN_MLPGATE_OFFSET,
    IN_MLPUP_SCALE,
    IN_MLPUP_OFFSET,
    IN_MLPDOWN_SCALE,
    IN_MLPDOWN_OFFSET,
    IN_COSTABLE,
    IN_SINTABLE,
    IN_ATTENTIONMASK,
    IN_CACHEK,
    IN_CACHEV,
    IN_TOKENOFFSET,
    IN_SEQLEN,
    IN_LAYERID,
    OUT_LAYEROUT,
    INTERMIDATE_INPUTNORMOUT,
    INTERMIDATE_MIXEDQ,
    INTERMIDATE_MIXEDK,
    INTERMIDATE_MIXEDV,
    INTERMIDATE_POSITIONEMBEDQ,
    INTERMIDATE_POSITIONEMBEDK,
    INTERMIDATE_SELFOUT,
    INTERMIDATE_SELFLINEAROUT,
    INTERMIDATE_SELFRESIDUALADDOUT,
    INTERMIDATE_SELFNORMOUT,
    INTERMIDATE_MLPOUT,
};

static const uint64_t IN_TENSOR_COUNT = 32;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 11;
static const uint64_t NODE_COUNT = 11;

atb::Status FlashAttentionLayer(const FlashAttentionLayerParam &param, atb::Operation **operation)
{
    std::shared_ptr<int64_t> batchNumPtr = std::make_shared<int64_t>(0);

    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);
    opGraph.name = "FlashAttentionLayerW8A16";
    size_t nodeId = 0;
    atb::Node &inputNormNode = opGraph.nodes.at(nodeId++);
    atb::Node &mixdQLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &mixdKLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &mixdVLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &ropeNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfAttentionNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfOutLinearParallelNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfResidualAddNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfNormNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpParallelNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpResidualAddNode = opGraph.nodes.at(nodeId++);

    // [b, s, h], [h] -> [b, s, h]
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

    // [b, s, h/w], [b, s, hd], [b * s, hd], [b * s, hd], [1] -> [b, s, h/w], [b, s, hd]
    atb::infer::RopeParam ropeParam;
    ropeParam.rotaryCoeff = param.rotaryCoeff;
    CREATE_OPERATION(ropeParam, &ropeNode.operation);
    ropeNode.inTensorIds = {INTERMIDATE_MIXEDQ, INTERMIDATE_MIXEDK, IN_COSTABLE, IN_SINTABLE, IN_SEQLEN};
    ropeNode.outTensorIds = {INTERMIDATE_POSITIONEMBEDQ, INTERMIDATE_POSITIONEMBEDK};
    ropeNode.inTensorReshapeFuncs.resize(ropeNode.inTensorIds.size());
    ropeNode.inTensorReshapeFuncs.at(0) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        *batchNumPtr = oldShape.dims[0];
        newShape.dimNum = 2;
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
        newShape.dims[1] = oldShape.dims[2];
    };
    ropeNode.inTensorReshapeFuncs.at(1) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2;
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
        newShape.dims[1] = oldShape.dims[2];
    };

    atb::infer::SelfAttentionParam selfAttentionParam;
    selfAttentionParam.headNum = param.headNum;
    selfAttentionParam.kvHeadNum = param.kvHeadNum;
    selfAttentionParam.headDim = param.dk;
    selfAttentionParam.qScale = 1.0 / sqrt(param.dk);
    CREATE_OPERATION(selfAttentionParam, &selfAttentionNode.operation);

    selfAttentionNode.inTensorIds = {INTERMIDATE_POSITIONEMBEDQ, // [b, s, h/w],
                                     INTERMIDATE_POSITIONEMBEDK, // [b, s, hd]
                                     INTERMIDATE_MIXEDV,         // [b, s, hd]
                                     IN_CACHEK,                  // [layers, b, max_seq, hd]
                                     IN_CACHEV,                  // [layers, b, max_seq, hd]
                                     IN_ATTENTIONMASK,           // [max_seq, max_seq]
                                     IN_TOKENOFFSET,             // [1]
                                     IN_SEQLEN,                  // [1]
                                     IN_LAYERID};                // [1]
    selfAttentionNode.outTensorIds = {INTERMIDATE_SELFOUT};      // [b, s, h/w]
    selfAttentionNode.inTensorReshapeFuncs.resize(selfAttentionNode.inTensorIds.size());
    selfAttentionNode.inTensorReshapeFuncs.at(2) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2;
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
        newShape.dims[1] = oldShape.dims[2];
    };

    atb_speed::contrib::LinearParallelW8A16Param selfOutLinearParam;
    selfOutLinearParam.transWeight = true;
    selfOutLinearParam.rank = param.rank;
    selfOutLinearParam.rankSize = param.rankSize;
    selfOutLinearParam.rankRoot = 0;
    selfOutLinearParam.bias = "None";
    selfOutLinearParam.parallelType = "RowParallel";
    selfOutLinearParam.backend = param.backend;
    CreateLinearParallelW8A16(selfOutLinearParam, &selfOutLinearParallelNode.operation);
    selfOutLinearParallelNode.inTensorIds = {INTERMIDATE_SELFOUT, IN_SELFOUTLINEARWEIGHT,
                                             IN_SELFOUTLINEAR_SCALE, IN_SELFOUTLINEAR_OFFSET};
    selfOutLinearParallelNode.outTensorIds = {INTERMIDATE_SELFLINEAROUT};
    selfOutLinearParallelNode.inTensorReshapeFuncs.resize(selfOutLinearParallelNode.inTensorIds.size());

    // [b, s, h] + [b, s, h] =  [b, s, h]
    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CREATE_OPERATION(addParam, &selfResidualAddNode.operation);
    selfResidualAddNode.inTensorIds = {IN_HIDDENSTATES, INTERMIDATE_SELFLINEAROUT};
    selfResidualAddNode.outTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT};
    selfResidualAddNode.inTensorReshapeFuncs.resize(selfResidualAddNode.inTensorIds.size());
    selfResidualAddNode.inTensorReshapeFuncs.at(1) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        int batchSize = *batchNumPtr;
        newShape.dimNum = 3;
        newShape.dims[0] = batchSize;
        newShape.dims[1] = oldShape.dims[0] / batchSize;
        newShape.dims[2] = oldShape.dims[1];
    };

    // [b, s, h], [h] -> [b, s, h]
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

    // [b, s, h] + [b, s, h] = [b, s, h]
    CREATE_OPERATION(addParam, &mlpResidualAddNode.operation);
    mlpResidualAddNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT, INTERMIDATE_MLPOUT};
    mlpResidualAddNode.outTensorIds = {OUT_LAYEROUT};

    // [b, s, h] -> [b, s, h]]
    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
    };

    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}

FlashAttentionLayerBinder::FlashAttentionLayerBinder() {}

FlashAttentionLayerBinder::~FlashAttentionLayerBinder() {}

void FlashAttentionLayerBinder::ParseParam(const nlohmann::json &paramJson)
{
    ATB_LOG(INFO) << "enter FusionLayerOperationW8A16 ParseParam tokenOffset";
    tokenOffset_.clear();
    for (auto item : paramJson["tokenOffset"]) {
        tokenOffset_.push_back(item.get<int>());
    }

    seqLen_.clear();
    for (auto item : paramJson["seqLen"]) {
        seqLen_.push_back(item.get<int>());
    }
    layerId_ = paramJson["layerId"].get<int>();
}

void FlashAttentionLayerBinder::BindTensor(atb::VariantPack &variantPack)
{
    ATB_LOG(INFO) << "enter FusionLayerOperationW8A16 BindTensor";
    const uint32_t tokenOffsetTensorId = IN_TOKENOFFSET;
    const uint32_t seqLenTensorId = IN_SEQLEN;
    const uint32_t layerIdTensorId = IN_LAYERID;
    variantPack.inTensors.at(tokenOffsetTensorId).hostData = tokenOffset_.data();
    variantPack.inTensors.at(seqLenTensorId).hostData = seqLen_.data();
    variantPack.inTensors.at(layerIdTensorId).hostData = &layerId_;
}

} // namespace contrib
} // namespace atb_speed