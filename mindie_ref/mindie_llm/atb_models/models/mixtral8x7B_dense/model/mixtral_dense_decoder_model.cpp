/*
* Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
*
*  * Licensed under the Apache License, Version 2.0 (the "License");
*  * you may not use this file except in compliance with the License.
*  * You may obtain a copy of the License at
*  *
*  * http://www.apache.org/licenses/LICENSE-2.0
*  *
*  * Unless required by applicable law or agreed to in writing, software
*  * distributed under the License is distributed on an "AS IS" BASIS,
*  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  * See the License for the specific language governing permissions and
*  * limitations under the License.
*  */
#include <nlohmann/json.hpp>
#include <atb/atb_infer.h>
#include <cmath>
#include "mixtral_dense_decoder_model.h"
#include "models/mixtral8x7B_dense/operation/mixtral_dense_layer_embedding.h"
#include "models/mixtral8x7B_dense/layer/mixtral_dense_fusion_layer_operation.h"

namespace atb_speed {
namespace mixtralDense {
const int WEIGHT_COUNT_PER_LAYER = 21;
const int WORDEMBEDDINGNODE_WEIGHT_COUNT = 0;
const int FINALNORMNODE_WEIGHT_COUNT = 0;
const int OPERATION_COUNT_BEFORE_LAYER = 1;
const int OPERATION_COUNT_AFTER_LAYER = 0;

enum InTensorId {
    IN_TENSOR_HIDDENSTATSE = 0,
    IN_MOE_FINAL_HIDDEN_STATE,
    IN_ONE_HOT_SCALER_ONE,
    IN_ONE_HOT_SCALER_ZERO,
    IN_TENSOR_POSITIONID,
    IN_TENSOR_COSTABLE,
    IN_TENSOR_SINTABLE,
    IN_TENSOR_ATTENTIONMASK,
    IN_TENSOR_KCACHE,
    IN_TENSOR_VCACHE,
    IN_TENSOR_TOKENOFFSET,
    IN_TENSOR_SEQLEN,
    IN_TENSOR_LAYERID_BASE,
};

enum InternelTensorId {
    INTERNEL_TENSOR_COS = 0,
    INTERNEL_TENSOR_SIN,
    INTERNEL_TENSOR_LAYEROUT_BASE,
};

void MixtralDenseDecoderModel::Param::FromString(const std::string &param)
{
    ATB_LOG(INFO) << "Start loading params";
    nlohmann::json paramJson = nlohmann::json::parse(param);
    dk = paramJson["dk"].get<int>();
    headNum = paramJson["headNum"].get<int>();
    rmsNormEps = paramJson["rmsNormEps"].get<float>();
    rank = paramJson["rank"].get<int>();
    rankSize = paramJson["rankSize"].get<int>();
    layerNum = paramJson["layerNum"].get<int>();
    backend = paramJson["backend"].get<std::string>();
    coderType = paramJson["coderType"].get<int>();
    isTriMask = paramJson["isTriMask"].get<int>();
    kvHeadNum = paramJson["kvHeadNum"].get<int>();

    ATB_LOG(INFO) << "Start loading tokenOffset params";
    for (auto item : paramJson["tokenOffset"]) {
        tokenOffset.push_back(item.get<int>());
    }

    ATB_LOG(INFO) << "Start loading seqLen params";
    for (auto item : paramJson["seqLen"]) {
        seqLen.push_back(item.get<int>());
    }
    ATB_LOG(INFO) << "MixtralDenseDecoderModel param rmsNormEps:" << rmsNormEps << ", headNum:" << headNum
                    << ", model:" << model << ", dk:" << dk << ", layerNum:" << layerNum
                    << ", rotaryCoeff:" << rotaryCoeff << ", rank:" << rank << ", rankSize:" << rankSize
                    << ", tokenOffset:" << tokenOffset << ", seqLen:" << seqLen << ", coderType:" << coderType
                    << "isTriMask" << isTriMask << ", kvHeadNum:" << kvHeadNum;
}

MixtralDenseDecoderModel::MixtralDenseDecoderModel(const std::string &param) : Model("MixtralDenseDecoderModel", param)
{
    param_.FromString(param);
}

MixtralDenseDecoderModel::~MixtralDenseDecoderModel() {}

uint32_t MixtralDenseDecoderModel::GetInputNum() const { return graph_.inTensors.size(); }

uint32_t MixtralDenseDecoderModel::GetOutputNum() const { return graph_.outTensors.size(); }

atb::Status MixtralDenseDecoderModel::InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
                                                 std::vector<atb::TensorDesc> &outTensorDescs)
{
    if (outTensorDescs.size() != GetOutputNum()) {
        return atb::ERROR_INVALID_GRAPH;
    }
    outTensorDescs.at(0) = inTensorDescs.at(0);

    return atb::NO_ERROR;
}

int64_t MixtralDenseDecoderModel::BuildGraph()
{
    ATB_LOG(INFO) << "MixtralDenseDecoderModel build graph begin";
    const int weightTensorSize =
        WORDEMBEDDINGNODE_WEIGHT_COUNT + WEIGHT_COUNT_PER_LAYER * param_.layerNum + FINALNORMNODE_WEIGHT_COUNT;
    graph_.weightTensors.resize(weightTensorSize);

    graph_.inTensors.resize(IN_TENSOR_LAYERID_BASE + param_.layerNum);
    graph_.outTensors.resize(1);

    const int nodeSize = param_.layerNum + OPERATION_COUNT_BEFORE_LAYER + OPERATION_COUNT_AFTER_LAYER;
    graph_.nodes.resize(nodeSize);

    const int internalTensorSize = INTERNEL_TENSOR_LAYEROUT_BASE + param_.layerNum - 1;
    graph_.internalTensors.resize(internalTensorSize);

    int nodeId = 0;
    atb::Operation *op = nullptr;
    auto &embeddingNode = graph_.nodes.at(nodeId++);
    atb_speed::mixtralDense::LayerEmbeddingParam layerEmbeddingParam;
    atb_speed::mixtralDense::LayerEmbedding(layerEmbeddingParam, &op);
    embeddingNode.operation.reset(op);
    embeddingNode.inTensors = {&graph_.inTensors.at(IN_TENSOR_COSTABLE),
                                &graph_.inTensors.at(IN_TENSOR_SINTABLE),
                                &graph_.inTensors.at(IN_TENSOR_POSITIONID)};
    embeddingNode.outTensors = {&graph_.internalTensors.at(INTERNEL_TENSOR_COS),
                                &graph_.internalTensors.at(INTERNEL_TENSOR_SIN)};

    atb::Tensor *firstInTensor = &graph_.inTensors.at(IN_TENSOR_HIDDENSTATSE);
    ATB_LOG(INFO) << "===========layerNum===========> " << param_.layerNum;
    for (int layerId = 0; layerId < param_.layerNum; ++layerId) {
        auto &layerNode = graph_.nodes.at(nodeId++);
        atb_speed::mixtralDense::MixtralDenseLayerFusionParam opParam;
        opParam.headNum = param_.headNum;
        opParam.rmsNormEps = param_.rmsNormEps;
        opParam.dk = param_.dk;
        opParam.layerId = layerId;
        opParam.rank = param_.rank;
        opParam.kvHeadNum = param_.kvHeadNum;
        opParam.rankSize = param_.rankSize;
        opParam.backend = param_.backend;
        opParam.qkScale = 1.0 / (sqrt(param_.dk));
        opParam.coderType = param_.coderType;
        opParam.isTriMask = param_.isTriMask;

        ATB_LOG(INFO) << "Model params init success: " << layerId;

        atb::Operation *op = nullptr;
        mixtralDense::MixtralDenseLayerFusionOperation(opParam, &op);
        if (op == nullptr) {
            ATB_LOG(ERROR) << "Layer op is nullptr: " << layerId;
        }
        layerNode.operation.reset(op);
        layerNode.inTensors.resize(layerNode.operation->GetInputNum());
        size_t inTensorId = 0;
        layerNode.inTensors.at(inTensorId++) = firstInTensor;
        for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
            layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(
                layerId * WEIGHT_COUNT_PER_LAYER + weightTensorId + WORDEMBEDDINGNODE_WEIGHT_COUNT);
        }
        ATB_LOG(INFO) << "Model Layer set weight success: " << layerId;
        layerNode.inTensors.at(inTensorId++) = &graph_.internalTensors.at(INTERNEL_TENSOR_COS); // cosTable
        layerNode.inTensors.at(inTensorId++) = &graph_.internalTensors.at(INTERNEL_TENSOR_SIN); // sinTable
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_MOE_FINAL_HIDDEN_STATE); // Hidden_state for mpe
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_ONE_HOT_SCALER_ONE);     // scaler tensor for one hot
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_ONE_HOT_SCALER_ZERO);    // scaler tensor for one hot
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ATTENTIONMASK);   // attentionMaskTensor
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_KCACHE);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_VCACHE);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_TOKENOFFSET);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SEQLEN); // seqLen
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_LAYERID_BASE + layerId);
        ATB_LOG(INFO) << "Model Layer set intensors success: " << layerId;
        if (layerId < param_.layerNum - 1) {
            layerNode.outTensors = {&graph_.internalTensors.at(INTERNEL_TENSOR_LAYEROUT_BASE + layerId)};
        } else {
            layerNode.outTensors = {&graph_.outTensors.at(0)};
        }
        firstInTensor = layerNode.outTensors.at(0);
        ATB_LOG(INFO) << "Layers created" << layerId;
    }
    ATB_LOG(INFO) << "MixtralDenseDecoderModel build graph success";
    return atb::NO_ERROR;
}

atb::Status MixtralDenseDecoderModel::ParseParam(const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);
    tokenOffset_.clear();
    for (auto item : paramJson["tokenOffset"]) {
        tokenOffset_.push_back(item.get<int>());
    }

    seqLen_.clear();
    for (auto item : paramJson["seqLen"]) {
        seqLen_.push_back(item.get<int>());
    }

    ATB_LOG(INFO) << "MixtralDenseDecoderModel ParseParam tokenOffset:" << tokenOffset_ << ", seqLen:" << seqLen_;

    return atb::NO_ERROR;
}

atb::Status MixtralDenseDecoderModel::BindParamHostTensor(uint32_t nodeId)
{
    if (nodeId < OPERATION_COUNT_BEFORE_LAYER || static_cast<int32_t>(nodeId) >= OPERATION_COUNT_BEFORE_LAYER + param_.layerNum) {
        return atb::NO_ERROR;
    }

    layerId_ = nodeId - OPERATION_COUNT_BEFORE_LAYER;
    auto &node = graph_.nodes.at(nodeId);

    const uint32_t seqLenTensorId = mixtralDense::MixtralDenseFusionTensorId::IN_SEQLEN;
    const uint32_t tokenOffsetTensorId = mixtralDense::MixtralDenseFusionTensorId::IN_TOKENOFFSET;
    const uint32_t layerIdTensorId = mixtralDense::MixtralDenseFusionTensorId::IN_LAYERID;
    node.variantPack.inTensors.at(seqLenTensorId).hostData = seqLen_.data();
    node.variantPack.inTensors.at(tokenOffsetTensorId).hostData = tokenOffset_.data();
    node.variantPack.inTensors.at(layerIdTensorId).hostData = &layerId_;

    return atb::NO_ERROR;
}
}
} // namespace atb_speed