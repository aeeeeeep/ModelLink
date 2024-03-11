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
#include "deepseek_dense_fusion_layer_operation.h"
#include "deepseek/16b_dense/operation/deepseek_dense_multi_layer_linear.h"
#include "deepseek/16b_dense/operation/deepseek_dense_moe.h"
#include "deepseek/16b_dense/operation/deepseek_dense_position_embedding_1d_split_fusion_operation.h"
#include "deepseek/16b_dense/operation/deepseek_dense_mlp_without_expert.h"

namespace atb_speed {
namespace deepseekDense {
static const uint64_t IN_TENSOR_COUNT = 147;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 14;
static const uint64_t NODE_COUNT = 12;

atb::Status DeepseekDenseLayerFusionOperation(const DeepseekDenseLayerFusionParam &param, atb::Operation **operation)
{
    std::shared_ptr<int64_t> batchDimPtr = std::make_shared<int64_t>(0);

    atb::GraphParam opGraph;
    opGraph.name = "DeepseekDenseLayerParallelFusion";
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
    atb::Node &sharedMlpExpertNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpAddNode = opGraph.nodes.at(nodeId++);
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

    atb_speed::deepseekDense::DeepseekDenseMultiLayerLinearParam multiLayerLinearParam;
    multiLayerLinearParam.transpose = param.transpose;
    multiLayerLinearParam.headNum = param.headNum;
    multiLayerLinearParam.dk = param.dk;
    deepseekDense::CreateDeepseekDenseMultiLayerLinearOperation(multiLayerLinearParam, &mixdQKVLinearNode.operation);
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
    selfOutLinearParallelParam.rankTableFile = param.rankTableFile;
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
    if (selfResidualAddNode.operation == nullptr) {
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
    if (selfNormNode.operation == nullptr) {
        ATB_LOG(ERROR) << "selfNormNode op is nullptr: ";
    }
    selfNormNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT, IN_SELFOUTNORMWEIGHT};
    selfNormNode.outTensorIds = {INTERMIDATE_SELFNORMOUT};
    ATB_LOG(INFO) << "create post rmsnorm";

    atb_speed::deepseekDense::DeepseekDenseMoeParam deepseekDenseMoeParam;
    deepseekDenseMoeParam.transpose = param.transpose;
    deepseekDenseMoeParam.numOfExperts = param.numOfExperts;
    deepseekDense::CreateDeepseekDenseMoeOperation(deepseekDenseMoeParam, &moeNode.operation);
    if (moeNode.operation == nullptr) {
        ATB_LOG(ERROR) << "selfNormNode op is nullptr: ";
    }
    moeNode.inTensorIds = {
        INTERMIDATE_SELFNORMOUT,
        IN_BLOCK_SPARSE_MOE_GATE_WEIGHT,
        IN_MLPGATEUPWEIGHT_EXPERT_ZERO,
        IN_MLPDOWNWEIGHT_EXPERT_ZERO,
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
        IN_MLPGATEUPWEIGHT_EXPERT_NINE,
        IN_MLPDOWNWEIGHT_EXPERT_NINE,
        IN_MLPGATEUPWEIGHT_EXPERT_TEN,
        IN_MLPDOWNWEIGHT_EXPERT_TEN,
        IN_MLPGATEUPWEIGHT_EXPERT_ELEVEN,
        IN_MLPDOWNWEIGHT_EXPERT_ELEVEN,
        IN_MLPGATEUPWEIGHT_EXPERT_TWELVE,
        IN_MLPDOWNWEIGHT_EXPERT_TWELVE,
        IN_MLPGATEUPWEIGHT_EXPERT_THIRTEEN,
        IN_MLPDOWNWEIGHT_EXPERT_THIRTEEN,
        IN_MLPGATEUPWEIGHT_EXPERT_FOURTEEN,
        IN_MLPDOWNWEIGHT_EXPERT_FOURTEEN,
        IN_MLPGATEUPWEIGHT_EXPERT_FIFTEENN,
        IN_MLPDOWNWEIGHT_EXPERT_FIFTEEN,
        IN_MLPGATEUPWEIGHT_EXPERT_SIXTEEN,
        IN_MLPDOWNWEIGHT_EXPERT_SIXTEEN,
        IN_MLPGATEUPWEIGHT_EXPERT_SEVENTEEN,
        IN_MLPDOWNWEIGHT_EXPERT_SEVENTEEN,
        IN_MLPGATEUPWEIGHT_EXPERT_EIGHTEEN,
        IN_MLPDOWNWEIGHT_EXPERT_EIGHTEEN,
        IN_MLPGATEUPWEIGHT_EXPERT_NINETEEN,
        IN_MLPDOWNWEIGHT_EXPERT_NINETEEN,
        IN_MLPGATEUPWEIGHT_EXPERT_TWENTY,
        IN_MLPDOWNWEIGHT_EXPERT_TWENTY,
        IN_MLPGATEUPWEIGHT_EXPERT_TWENTY_ONE,
        IN_MLPDOWNWEIGHT_EXPERT_TWENTY_ONE,
        IN_MLPGATEUPWEIGHT_EXPERT_TWENTY_TWO,
        IN_MLPDOWNWEIGHT_EXPERT_TWENTY_TWO,
        IN_MLPGATEUPWEIGHT_EXPERT_TWENTY_THREE,
        IN_MLPDOWNWEIGHT_EXPERT_TWENTY_THREE,
        IN_MLPGATEUPWEIGHT_EXPERT_TWENTY_FOUR,
        IN_MLPDOWNWEIGHT_EXPERT_TWENTY_FOUR,
        IN_MLPGATEUPWEIGHT_EXPERT_TWENTY_FIVE,
        IN_MLPDOWNWEIGHT_EXPERT_TWENTY_FIVE,
        IN_MLPGATEUPWEIGHT_EXPERT_TWENTY_SIX,
        IN_MLPDOWNWEIGHT_EXPERT_TWENTY_SIX,
        IN_MLPGATEUPWEIGHT_EXPERT_TWENTY_SEVEN,
        IN_MLPDOWNWEIGHT_EXPERT_TWENTY_SEVEN,
        IN_MLPGATEUPWEIGHT_EXPERT_TWENTY_EIGHT,
        IN_MLPDOWNWEIGHT_EXPERT_TWENTY_EIGHT,
        IN_MLPGATEUPWEIGHT_EXPERT_TWENTY_NINE,
        IN_MLPDOWNWEIGHT_EXPERT_TWENTY_NINE,
        IN_MLPGATEUPWEIGHT_EXPERT_THIRTY,
        IN_MLPDOWNWEIGHT_EXPERT_THIRTY,
        IN_MLPGATEUPWEIGHT_EXPERT_THIRTY_ONE,
        IN_MLPDOWNWEIGHT_EXPERT_THIRTY_ONE,
        IN_MLPGATEUPWEIGHT_EXPERT_THIRTY_TWO,
        IN_MLPDOWNWEIGHT_EXPERT_THIRTY_TWO,
        IN_MLPGATEUPWEIGHT_EXPERT_THIRTY_THREE,
        IN_MLPDOWNWEIGHT_EXPERT_THIRTY_THREE,
        IN_MLPGATEUPWEIGHT_EXPERT_THIRTY_FOUR,
        IN_MLPDOWNWEIGHT_EXPERT_THIRTY_FOUR,
        IN_MLPGATEUPWEIGHT_EXPERT_THIRTY_FIVE,
        IN_MLPDOWNWEIGHT_EXPERT_THIRTY_FIVE,
        IN_MLPGATEUPWEIGHT_EXPERT_THIRTY_SIX,
        IN_MLPDOWNWEIGHT_EXPERT_THIRTY_SIX,
        IN_MLPGATEUPWEIGHT_EXPERT_THIRTY_SEVEN,
        IN_MLPDOWNWEIGHT_EXPERT_THIRTY_SEVEN,
        IN_MLPGATEUPWEIGHT_EXPERT_THIRTY_EIGHT,
        IN_MLPDOWNWEIGHT_EXPERT_THIRTY_EIGHT,
        IN_MLPGATEUPWEIGHT_EXPERT_THIRTY_NINE,
        IN_MLPDOWNWEIGHT_EXPERT_THIRTY_NINE,
        IN_MLPGATEUPWEIGHT_EXPERT_FOURTY,
        IN_MLPDOWNWEIGHT_EXPERT_FOURTY,
        IN_MLPGATEUPWEIGHT_EXPERT_FOURTY_ONE,
        IN_MLPDOWNWEIGHT_EXPERT_FOURTY_ONE,
        IN_MLPGATEUPWEIGHT_EXPERT_FOURTY_TWO,
        IN_MLPDOWNWEIGHT_EXPERT_FOURTY_TWO,
        IN_MLPGATEUPWEIGHT_EXPERT_FOURTY_THREE,
        IN_MLPDOWNWEIGHT_EXPERT_FOURTY_THREE,
        IN_MLPGATEUPWEIGHT_EXPERT_FOURTY_FOUR,
        IN_MLPDOWNWEIGHT_EXPERT_FOURTY_FOUR,
        IN_MLPGATEUPWEIGHT_EXPERT_FOURTY_FIVE,
        IN_MLPDOWNWEIGHT_EXPERT_FOURTY_FIVE,
        IN_MLPGATEUPWEIGHT_EXPERT_FOURTY_SIX,
        IN_MLPDOWNWEIGHT_EXPERT_FOURTY_SIX,
        IN_MLPGATEUPWEIGHT_EXPERT_FOURTY_SEVEN,
        IN_MLPDOWNWEIGHT_EXPERT_FOURTY_SEVEN,
        IN_MLPGATEUPWEIGHT_EXPERT_FOURTY_EIGHT,
        IN_MLPDOWNWEIGHT_EXPERT_FOURTY_EIGHT,
        IN_MLPGATEUPWEIGHT_EXPERT_FOURTY_NINE,
        IN_MLPDOWNWEIGHT_EXPERT_FOURTY_NINE,
        IN_MLPGATEUPWEIGHT_EXPERT_FIFTY,
        IN_MLPDOWNWEIGHT_EXPERT_FIFTY,
        IN_MLPGATEUPWEIGHT_EXPERT_FIFTY_ONE,
        IN_MLPDOWNWEIGHT_EXPERT_FIFTY_ONE,
        IN_MLPGATEUPWEIGHT_EXPERT_FIFTY_TWO,
        IN_MLPDOWNWEIGHT_EXPERT_FIFTY_TWO,
        IN_MLPGATEUPWEIGHT_EXPERT_FIFTY_THREE,
        IN_MLPDOWNWEIGHT_EXPERT_FIFTY_THREE,
        IN_MLPGATEUPWEIGHT_EXPERT_FIFTY_FOUR,
        IN_MLPDOWNWEIGHT_EXPERT_FIFTY_FOUR,
        IN_MLPGATEUPWEIGHT_EXPERT_FIFTY_FIVE,
        IN_MLPDOWNWEIGHT_EXPERT_FIFTY_FIVE,
        IN_MLPGATEUPWEIGHT_EXPERT_FIFTY_SIX,
        IN_MLPDOWNWEIGHT_EXPERT_FIFTY_SIX,
        IN_MLPGATEUPWEIGHT_EXPERT_FIFTY_SEVEN,
        IN_MLPDOWNWEIGHT_EXPERT_FIFTY_SEVEN,
        IN_MLPGATEUPWEIGHT_EXPERT_FIFTY_EIGHT,
        IN_MLPDOWNWEIGHT_EXPERT_FIFTY_EIGHT,
        IN_MLPGATEUPWEIGHT_EXPERT_FIFTY_NINEE,
        IN_MLPDOWNWEIGHT_EXPERT_FIFTY_NINE,
        IN_MLPGATEUPWEIGHT_EXPERT_SIXTY,
        IN_MLPDOWNWEIGHT_EXPERT_SIXTY,
        IN_MLPGATEUPWEIGHT_EXPERT_SIXTY_ONE,
        IN_MLPDOWNWEIGHT_EXPERT_SIXTY_ONE,
        IN_MLPGATEUPWEIGHT_EXPERT_SIXTY_TWO,
        IN_MLPDOWNWEIGHT_EXPERT_SIXTY_TWO,
        IN_MLPGATEUPWEIGHT_EXPERT_SIXTY_THREE,
        IN_MLPDOWNWEIGHT_EXPERT_SIXTY_THREE,
        IN_ONE_HOT_ONE,
        IN_ONE_HOT_ZERO,
        IN_FINAL_HIDDEN_STATE};
    moeNode.outTensorIds = {INTERMIDATE_MOEOUT};
    ATB_LOG(INFO) << "create input Moe";

    atb_speed::deepseekDense::DeepseekDenseMlpWithoutExpertParam sharedMlpExpertParam;
    sharedMlpExpertParam.transpose = param.transpose;
    deepseekDense::CreateDeepseekDenseMlpWithoutExpertOperation(
        sharedMlpExpertParam, &sharedMlpExpertNode.operation);
    sharedMlpExpertNode.inTensorIds = {INTERMIDATE_SELFNORMOUT,
                                  IN_MLPGATEUPWEIGHT_SHARED_EXPERT,
                                  IN_MLPDOWNWEIGHT_EXPERT_SHARED_EXPERT};
    sharedMlpExpertNode.outTensorIds = {INTERMIDATE_HIDDEN_STATE_SHARED_EXPERTS};
    ATB_LOG(INFO) << "shared expert calculation success";

    atb::infer::ElewiseParam sharedMlpAddParam;
    sharedMlpAddParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CreateOperation(sharedMlpAddParam, &mlpAddNode.operation);
    mlpAddNode.inTensorIds = {INTERMIDATE_HIDDEN_STATE_SHARED_EXPERTS, INTERMIDATE_MOEOUT};
    mlpAddNode.outTensorIds = {INTERMIDATE_MOEOUT_ALL};

    atb::infer::AllReduceParam allReduceParam;
    allReduceParam.rank = param.rank;
    allReduceParam.rankSize = param.rankSize;
    allReduceParam.rankRoot = param.rankRoot;
    allReduceParam.backend = param.backend;
    allReduceParam.hcclComm = param.hcclComm;
    allReduceParam.rankTableFile = param.rankTableFile;
    CreateOperation(allReduceParam, &moeAllReduceNode.operation);
    if (moeAllReduceNode.operation == nullptr) {
        ATB_LOG(ERROR) << "moeAllReduceNode op is nullptr: ";
    }
    moeAllReduceNode.inTensorIds = {INTERMIDATE_MOEOUT_ALL};
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
    moeResidualAddNode.outTensorIds = {OUT_DEEPSEEK_DENSE_LAYEROUT};
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

DeepseekDenseLayerFusionBinder::DeepseekDenseLayerFusionBinder() {}

DeepseekDenseLayerFusionBinder::~DeepseekDenseLayerFusionBinder() {}

void DeepseekDenseLayerFusionBinder::ParseParam(const nlohmann::json &paramJson)
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

void DeepseekDenseLayerFusionBinder::BindTensor(atb::VariantPack &variantPack)
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
