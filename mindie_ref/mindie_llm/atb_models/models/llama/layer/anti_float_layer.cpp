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
#include "anti_float_layer.h"
#include "layers/parallel_layer.h"
#include "models/llama/operation/anti_mlp.h"
#include "models/llama/operation/rope_fusion_operation.h"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-variable"

namespace atb_speed {
namespace llama {
const int ATTENTION_DIM_NUM = 4;
const int ATTENTION_DIM_2 = 2;
const int ATTENTION_DIM_3 = 3;

enum AntiFloatLayerTensorId {
    IN_HIDDENSTATES = 0,
    IN_NORMWEIGHT,
    IN_BETA,

    IN_QMIXDWEIGHT,
    IN_QMIXD_BIAS,
    IN_KMIXDWEIGHT,
    IN_KMIXD_BIAS,
    IN_VMIXDWEIGHT,
    IN_VMIXD_BIAS,
    
    IN_SELFOUTLINEARWEIGHT,
    IN_SELFOUTNORMWEIGHT,
    IN_SELFOUTBETA,

    IN_MLPGATEWEIGHT,
    IN_MLPGATE_BIAS,
    IN_MLPDOWNWEIGHT,
    IN_MLPUPWEIGHT,
    IN_MLPUP_BIAS,

    IN_POSITIONIDS,

    IN_COSTABLE,

    IN_SINTABLE,

    IN_ATTENTIONMASK,

    IN_CACHEK,

    IN_CACHEV,

    IN_TOKENOFFSET,

    IN_SEQLEN,

    IN_LAYERID,
    OUT_LLAMA7BLAYEROUT,
    INTERMIDATE_INPUTNORMOUT,
    INTERMIDATE_NORMADDOUT,
    INTERMIDATE_MIXEDQ,
    INTERMIDATE_MIXEDK,
    INTERMIDATE_MIXEDV,
    INTERMIDATE_POSITIONEMBEDQ,
    INTERMIDATE_POSITIONEMBEDK,
    INTERMIDATE_SELFOUT,
    INTERMIDATE_SELFLINEAROUT,
    INTERMIDATE_SELFRESIDUALADDOUT,
    INTERMIDATE_SELFNORMOUT,
    INTERMIDATE_SELFNORMADDOUT,
    INTERMIDATE_MLPOUT,
};

static const uint64_t IN_TENSOR_COUNT = 26;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 13;
static const uint64_t NODE_COUNT = 13;

atb::Status AntiFloatLayer(const AntiFloatLayerParam  &param,
    atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.name = "AntiFloatLayer";
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    atb::Node &inputNormNode = opGraph.nodes.at(nodeId++);
    atb::Node &InputNormAddNode = opGraph.nodes.at(nodeId++);
    atb::Node &mixdQLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &mixdKLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &mixdVLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &ropeNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfAttentionKvCacheNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfOutLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfResidualAddNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfNormNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfOutNormAddNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpResidualAddNode = opGraph.nodes.at(nodeId++);

    atb::infer::RmsNormParam rmsNormParam;
    rmsNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    rmsNormParam.normParam.epsilon = param.rmsNormEps;
    CREATE_OPERATION(rmsNormParam, &inputNormNode.operation);
    inputNormNode.inTensorIds = {IN_HIDDENSTATES, IN_NORMWEIGHT};
    inputNormNode.outTensorIds = {INTERMIDATE_INPUTNORMOUT};

    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CREATE_OPERATION(addParam, &InputNormAddNode.operation);
    InputNormAddNode.inTensorIds = {INTERMIDATE_INPUTNORMOUT, IN_BETA};
    InputNormAddNode.outTensorIds = {INTERMIDATE_NORMADDOUT};

    atb::infer::LinearParam linearParam;
    CREATE_OPERATION(linearParam, &mixdQLinearNode.operation);
    mixdQLinearNode.inTensorIds = {INTERMIDATE_NORMADDOUT, IN_QMIXDWEIGHT, IN_QMIXD_BIAS};
    mixdQLinearNode.outTensorIds = {INTERMIDATE_MIXEDQ};

    CREATE_OPERATION(linearParam, &mixdKLinearNode.operation);
    mixdKLinearNode.inTensorIds = {INTERMIDATE_NORMADDOUT, IN_KMIXDWEIGHT, IN_KMIXD_BIAS};
    mixdKLinearNode.outTensorIds = {INTERMIDATE_MIXEDK};

    CREATE_OPERATION(linearParam, &mixdVLinearNode.operation);
    mixdVLinearNode.inTensorIds = {INTERMIDATE_NORMADDOUT, IN_VMIXDWEIGHT, IN_VMIXD_BIAS};
    mixdVLinearNode.outTensorIds = {INTERMIDATE_MIXEDV};

    atb_speed::llama::RopeFusionParam ropeFusionParam;
    ropeFusionParam.headNum = param.headNum;
    atb_speed::llama::RopeFusionOperation(ropeFusionParam, &ropeNode.operation);
    ropeNode.inTensorIds = {INTERMIDATE_MIXEDQ, INTERMIDATE_MIXEDK, IN_POSITIONIDS,
                            IN_COSTABLE,        IN_SINTABLE,        IN_SEQLEN};
    ropeNode.outTensorIds = {INTERMIDATE_POSITIONEMBEDQ, INTERMIDATE_POSITIONEMBEDK};

    atb::infer::SelfAttentionParam selfAttentionKvCacheParam;
    selfAttentionKvCacheParam.headDim = param.dk;
    selfAttentionKvCacheParam.headNum = param.headNum;
    selfAttentionKvCacheParam.qScale = 1.0 / sqrt(param.dk);
    CREATE_OPERATION(selfAttentionKvCacheParam, &selfAttentionKvCacheNode.operation);
    selfAttentionKvCacheNode.inTensorIds = {INTERMIDATE_POSITIONEMBEDQ,
                                            INTERMIDATE_POSITIONEMBEDK,
                                            INTERMIDATE_MIXEDV,
                                            IN_CACHEK,
                                            IN_CACHEV,
                                            IN_ATTENTIONMASK,
                                            IN_TOKENOFFSET,
                                            IN_SEQLEN,
                                            IN_LAYERID};
    selfAttentionKvCacheNode.outTensorIds = {INTERMIDATE_SELFOUT};
    selfAttentionKvCacheNode.inTensorReshapeFuncs.resize(selfAttentionKvCacheNode.inTensorIds.size());
    selfAttentionKvCacheNode.inTensorReshapeFuncs.at(2) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = ATTENTION_DIM_NUM;
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = oldShape.dims[1];
        newShape.dims[ATTENTION_DIM_2] = param.headNum;
        newShape.dims[ATTENTION_DIM_3] = oldShape.dims[ATTENTION_DIM_2] / param.headNum;
    };

    atb_speed::common::ParallelParam selfOutLinearParam;
    selfOutLinearParam.rank = param.rank;
    selfOutLinearParam.rankSize = param.rankSize;
    selfOutLinearParam.isBias = false;
    atb_speed::common::RowParallelLinear(selfOutLinearParam, &selfOutLinearNode.operation);
    selfOutLinearNode.inTensorIds = {INTERMIDATE_SELFOUT, IN_SELFOUTLINEARWEIGHT};
    selfOutLinearNode.outTensorIds = {INTERMIDATE_SELFLINEAROUT};

    CREATE_OPERATION(addParam, &selfResidualAddNode.operation);
    selfResidualAddNode.inTensorIds = {IN_HIDDENSTATES, INTERMIDATE_SELFLINEAROUT};
    selfResidualAddNode.outTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT};

    CREATE_OPERATION(rmsNormParam, &selfNormNode.operation);
    selfNormNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT, IN_SELFOUTNORMWEIGHT};
    selfNormNode.outTensorIds = {INTERMIDATE_SELFNORMOUT};

    CREATE_OPERATION(addParam, &selfOutNormAddNode.operation);
    selfOutNormAddNode.inTensorIds = {INTERMIDATE_SELFNORMOUT, IN_SELFOUTBETA};
    selfOutNormAddNode.outTensorIds = {INTERMIDATE_SELFNORMADDOUT};

    atb_speed::llama::MlpGateParam mlpParam;
    mlpParam.rank = param.rank;
    mlpParam.rankSize = param.rankSize;
    mlpParam.activationType = atb::infer::ActivationType::ACTIVATION_SWISH;
    mlpParam.transposeB = true;
    mlpParam.isBias = true;
    mlpParam.isPack = false;
    atb_speed::llama::MlpGateLayer(mlpParam, &mlpNode.operation);
    mlpNode.inTensorIds = {INTERMIDATE_SELFNORMADDOUT, IN_MLPUPWEIGHT, IN_MLPGATEWEIGHT, IN_MLPDOWNWEIGHT,
                            IN_MLPUP_BIAS, IN_MLPGATE_BIAS};
    mlpNode.outTensorIds = {INTERMIDATE_MLPOUT};

    CREATE_OPERATION(addParam, &mlpResidualAddNode.operation);
    mlpResidualAddNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT, INTERMIDATE_MLPOUT};
    mlpResidualAddNode.outTensorIds = {OUT_LLAMA7BLAYEROUT};

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
    };

    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}

AntiFloatLayerBinder::AntiFloatLayerBinder() {}

AntiFloatLayerBinder::~AntiFloatLayerBinder() {}

void AntiFloatLayerBinder::ParseParam(const nlohmann::json &paramJson)
{
    tokenOffset_.clear();
    for (auto item : paramJson["tokenOffset"]) {
        tokenOffset_.push_back(item.get<int>());
    }

    seqLen_.clear();
    for (auto item : paramJson["seqLen"]) {
        seqLen_.push_back(item.get<int>());
    }
}

void AntiFloatLayerBinder::BindTensor(atb::VariantPack &variantPack)
{
    const uint32_t tokenOffsetTensorId = IN_TOKENOFFSET;
    const uint32_t seqLenTensorId = IN_SEQLEN;
    const uint32_t layerIdTensorId = IN_LAYERID;
    variantPack.inTensors.at(tokenOffsetTensorId).hostData = tokenOffset_.data();
    variantPack.inTensors.at(seqLenTensorId).hostData = seqLen_.data();
}

} // namespace llama
} // namespace atb_speed