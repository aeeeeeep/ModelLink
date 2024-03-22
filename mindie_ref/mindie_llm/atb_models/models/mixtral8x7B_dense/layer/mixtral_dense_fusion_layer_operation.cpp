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
#include "mixtral_dense_fusion_layer_operation.h"
#include "mixtral8x7B_dense/operation/mixtral_dense_multi_layer_linear.h"
#include "mixtral8x7B_dense/operation/mixtral_dense_moe.h"
#include "mixtral8x7B_dense/operation/mixtral_dense_position_embedding_1d_split_fusion_operation.h"

namespace atb_speed {
namespace mixtralDense {
static const uint64_t IN_TENSOR_COUNT = 33;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 12;
static const uint64_t NODE_COUNT = 10;

atb::Status MixtralDenseLayerFusionOperation(const MixtralDenseLayerFusionParam &param,
                                             atb::Operation **operation)
{
    std::shared_ptr<int64_t> batchDimPtr = std::make_shared<int64_t>(0);

    atb::GraphParam opGraph;
    opGraph.name = "MixtralDenseLayerParallelFusion";
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    atb::Node &inputNormNode = opGraph.nodes.at(nodeId++);
    atb::Node &mixdQKVLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &ropeNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfAttentionKvCacheNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfOutLinearParallelNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfResidualAddNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfNormNode = opGraph.nodes.at(nodeId++);
    atb::Node &moeNode = opGraph.nodes.at(nodeId++);
    atb::Node &moeAllReduceNode = opGraph.nodes.at(nodeId++);
    atb::Node &moeResidualAddNode = opGraph.nodes.at(nodeId++);

    atb::infer::RmsNormParam inputNormParam;
    inputNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    inputNormParam.normParam.epsilon = param.rmsNormEps;
    CreateOperation(inputNormParam, &inputNormNode.operation);
    if (inputNormNode.operation == nullptr) {
        ATB_LOG(ERROR) << "inputNormNode op is nullptr: ";
    }
    inputNormNode.inTensorIds = {IN_HIDDENSTATES, IN_NORMWEIGHT};
    inputNormNode.outTensorIds = {INTERMIDATE_INPUTNORMOUT};
    ATB_LOG(INFO) << "create input rmsnorm";

    atb_speed::mixtralDense::MixtralDenseMultiLayerLinearParam multiLayerLinearParam;
    multiLayerLinearParam.transpose = param.transpose;
    multiLayerLinearParam.headNum = param.headNum;
    multiLayerLinearParam.dk = param.dk;
    mixtralDense::CreateMixtralDenseMultiLayerLinearOperation(multiLayerLinearParam, &mixdQKVLinearNode.operation);
    if (mixdQKVLinearNode.operation == nullptr) {
        ATB_LOG(ERROR) << "mixdQKVLinearNode op is nullptr: ";
    }
    mixdQKVLinearNode.inTensorIds = {INTERMIDATE_INPUTNORMOUT, IN_QKVMIXDWEIGHT};
    mixdQKVLinearNode.outTensorIds = {INTERMIDATE_MIXEDQ, INTERMIDATE_MIXEDK, INTERMIDATE_MIXEDV};
    ATB_LOG(INFO) << "create input MultiLayerLinear";

    atb::infer::RopeParam ropeParam;
    ropeParam.rotaryCoeff = param.rotaryCoeff;
    CREATE_OPERATION(ropeParam, &ropeNode.operation);
    if (ropeNode.operation == nullptr) {
        ATB_LOG(ERROR) << "input PositionEmbedding op is nullptr: ";
    }
    ropeNode.inTensorIds = {INTERMIDATE_MIXEDQ, INTERMIDATE_MIXEDK, IN_COSTABLE, IN_SINTABLE, IN_SEQLEN};
    ropeNode.outTensorIds = {INTERMIDATE_POSITIONEMBEDQ, INTERMIDATE_POSITIONEMBEDK};
    ropeNode.inTensorReshapeFuncs.resize(ropeNode.inTensorIds.size());
    ropeNode.inTensorReshapeFuncs.at(0) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        *batchDimPtr = oldShape.dims[0];
        newShape.dimNum = 2;
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
        newShape.dims[1] = oldShape.dims[2];
    };
    ropeNode.inTensorReshapeFuncs.at(1) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2;
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
        newShape.dims[1] = oldShape.dims[2];
    };
    ropeNode.inTensorReshapeFuncs.at(2) = [](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2;
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
        newShape.dims[1] = oldShape.dims[2];
    };
    ropeNode.inTensorReshapeFuncs.at(3) = [](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2;
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
        newShape.dims[1] = oldShape.dims[2];
    };
    ATB_LOG(INFO) << "create input PositionEmbedding";

    atb::infer::SelfAttentionParam selfAttentionKvCacheParam;
    selfAttentionKvCacheParam.kvHeadNum = param.kvHeadNum;
    selfAttentionKvCacheParam.headNum = param.headNum;
    selfAttentionKvCacheParam.qkScale = param.qkScale;
    selfAttentionKvCacheParam.isTriuMask = param.isTriMask;
    selfAttentionKvCacheParam.maskType = atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_NORM;
    if (param.coderType == 0) {
        selfAttentionKvCacheParam.calcType = atb::infer::SelfAttentionParam::CalcType::UNDEFINED;
    } else if (param.coderType == 1) {
        selfAttentionKvCacheParam.calcType = atb::infer::SelfAttentionParam::CalcType::ENCODER;
    } else if (param.coderType == 2) {
        selfAttentionKvCacheParam.calcType = atb::infer::SelfAttentionParam::CalcType::DECODER;
    }
    CreateOperation(selfAttentionKvCacheParam, &selfAttentionKvCacheNode.operation);
    if (mixdQKVLinearNode.operation == nullptr) {
        ATB_LOG(ERROR) << "input SelfAttention op is nullptr: ";
    }
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
        newShape.dimNum = 2;
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
        newShape.dims[1] = oldShape.dims[2];
    };
    ATB_LOG(INFO) << "create input SelfAttention";

    atb::infer::LinearParallelParam selfOutLinearParallelParam;
    selfOutLinearParallelParam.transWeight = true;
    selfOutLinearParallelParam.rank = param.rank;
    selfOutLinearParallelParam.rankSize = param.rankSize;
    selfOutLinearParallelParam.rankRoot = 0;
    selfOutLinearParallelParam.bias = "None";
    selfOutLinearParallelParam.parallelType = "RowParallel";
    selfOutLinearParallelParam.backend = param.backend;
    CreateOperation(selfOutLinearParallelParam, &selfOutLinearParallelNode.operation);
    if (selfOutLinearParallelNode.operation == nullptr) {
        ATB_LOG(ERROR) << "selfOutLinearParallelNode op is nullptr: ";
    }
    selfOutLinearParallelNode.inTensorIds = {INTERMIDATE_SELFOUT, IN_SELFOUTLINEARWEIGHT};
    selfOutLinearParallelNode.outTensorIds = {INTERMIDATE_SELFLINEAROUT};
    ATB_LOG(INFO) << "create input rmsnorm";

    atb::infer::ElewiseParam selfResidualAddParam;
    selfResidualAddParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CreateOperation(selfResidualAddParam, &selfResidualAddNode.operation);
    if (selfOutLinearParallelNode.operation == nullptr) {
        ATB_LOG(ERROR) << "selfResidualAddNode op is nullptr: ";
    }
    selfResidualAddNode.inTensorIds = {IN_HIDDENSTATES, INTERMIDATE_SELFLINEAROUT};
    selfResidualAddNode.outTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT};
    selfResidualAddNode.inTensorReshapeFuncs.resize(selfResidualAddNode.inTensorIds.size());
    selfResidualAddNode.inTensorReshapeFuncs.at(1) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        int batchSize = *batchDimPtr;
        newShape.dimNum = 3;
        newShape.dims[0] = batchSize;
        newShape.dims[1] = oldShape.dims[0] / batchSize;
        newShape.dims[2] = oldShape.dims[1];
    };
    ATB_LOG(INFO) << "create input LinearParallel";

    atb::infer::RmsNormParam selfNormParam;
    selfNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    selfNormParam.normParam.epsilon = param.rmsNormEps;
    CreateOperation(selfNormParam, &selfNormNode.operation);
    if (selfOutLinearParallelNode.operation == nullptr) {
        ATB_LOG(ERROR) << "selfNormNode op is nullptr: ";
    }
    selfNormNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT, IN_SELFOUTNORMWEIGHT};
    selfNormNode.outTensorIds = {INTERMIDATE_SELFNORMOUT};
    ATB_LOG(INFO) << "create post rmsnorm";

    atb_speed::mixtralDense::MixtralDenseMoeParam mixtralDenseMoeParam;
    mixtralDenseMoeParam.transpose = param.transpose;
    mixtralDense::CreateMixtralDenseMoeOperation(mixtralDenseMoeParam, &moeNode.operation);
    if (selfOutLinearParallelNode.operation == nullptr) {
        ATB_LOG(ERROR) << "selfNormNode op is nullptr: ";
    }
    moeNode.inTensorIds = {INTERMIDATE_SELFNORMOUT,
                            IN_BLOCK_SPARSE_MOE_GATE_WEIGHT,
                            IN_MLPGATEUPWEIGHT_EXPERT_ONE,
                            IN_MLPDOWNWEIGHT_EXPERT_ONE,
                            IN_MLPGATEUPWEIGHT_EXPERT_TWO,
                            IN_MLPDOWNWEIGHT_EXPERT_TWO,
                            IN_MLPGATEUPWEIGHT_EXPERT_THREE,
                            IN_MLPDOWNWEIGHT_EXPERT_THREE,
                            IN_MLPGATEUPWEIGHT_EXPERT_FOUR,
                            IN_MLPDOWNWEIGHT_EXPERT_FOUR,
                            IN_MLPGATEUPWEIGHT_EXPERT_FIVE,
                            IN_MLPDOWNWEIGHT_EXPERT_FIVE,
                            IN_MLPGATEUPWEIGHT_EXPERT_SIX,
                            IN_MLPDOWNWEIGHT_EXPERT_SIX,
                            IN_MLPGATEUPWEIGHT_EXPERT_SEVEN,
                            IN_MLPDOWNWEIGHT_EXPERT_SEVEN,
                            IN_MLPGATEUPWEIGHT_EXPERT_EIGHT,
                            IN_MLPDOWNWEIGHT_EXPERT_EIGHT,
                            IN_ONE_HOT_ONE,
                            IN_ONE_HOT_ZERO,
                            IN_FINAL_HIDDEN_STATE};
    moeNode.outTensorIds = {INTERMIDATE_MOEOUT};
    ATB_LOG(INFO) << "create input Moe";

    atb::infer::AllReduceParam allReduceParam;
    allReduceParam.rank = param.rank;
    allReduceParam.rankSize = param.rankSize;
    allReduceParam.rankRoot = param.rankRoot;
    allReduceParam.backend = param.backend;
    allReduceParam.hcclComm = param.hcclComm;
    allReduceParam.rankTableFile = param.rankTableFile;
    CreateOperation(allReduceParam, &moeAllReduceNode.operation);
    if (selfOutLinearParallelNode.operation == nullptr) {
        ATB_LOG(ERROR) << "moeAllReduceNode op is nullptr: ";
    }
    moeAllReduceNode.inTensorIds = {INTERMIDATE_MOEOUT};
    moeAllReduceNode.outTensorIds = {INTERMIDATE_MOELINEARPARALLELOUT};
    moeAllReduceNode.inTensorReshapeFuncs.resize(moeAllReduceNode.inTensorIds.size());
    moeAllReduceNode.inTensorReshapeFuncs[0] = [](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2; // dimNum: 2
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
        newShape.dims[1] = oldShape.dims[2]; // 2 : the second dimension
    };
    ATB_LOG(INFO) << "create all reduce";

    atb::infer::ElewiseParam moeResidualAddParam;
    moeResidualAddParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CreateOperation(moeResidualAddParam, &moeResidualAddNode.operation);
    if (selfOutLinearParallelNode.operation == nullptr) {
        ATB_LOG(ERROR) << "moeResidualAddNode op is nullptr: ";
    }
    moeResidualAddNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT, INTERMIDATE_MOELINEARPARALLELOUT};
    moeResidualAddNode.outTensorIds = {OUT_MIXTRAL_DENSE_LAYEROUT};
    moeResidualAddNode.inTensorReshapeFuncs.resize(moeResidualAddNode.inTensorIds.size());
    moeResidualAddNode.inTensorReshapeFuncs.at(1) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        int batchSize = *batchDimPtr;
        newShape.dimNum = 3;
        newShape.dims[0] = batchSize;
        newShape.dims[1] = oldShape.dims[0] / batchSize;
        newShape.dims[2] = oldShape.dims[1];
    };
    ATB_LOG(INFO) << "create input ResidualAdd";

    opGraph.inferShapeFunc = [&](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                    atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
    };

    return atb::CreateOperation(opGraph, operation);
}

MixtralDenseLayerFusionBinder::MixtralDenseLayerFusionBinder() {}

MixtralDenseLayerFusionBinder::~MixtralDenseLayerFusionBinder() {}

void MixtralDenseLayerFusionBinder::ParseParam(const nlohmann::json &paramJson)
{
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

void MixtralDenseLayerFusionBinder::BindTensor(atb::VariantPack &variantPack)
{
    const uint32_t tokenOffsetTensorId = IN_TOKENOFFSET;
    const uint32_t seqLenTensorId = IN_SEQLEN;
    const uint32_t layerIdTensorId = IN_LAYERID;
    variantPack.inTensors.at(tokenOffsetTensorId).hostData = tokenOffset_.data();
    variantPack.inTensors.at(seqLenTensorId).hostData = seqLen_.data();
    variantPack.inTensors.at(layerIdTensorId).hostData = &layerId_;
}
}
} // namespace atb_speed
