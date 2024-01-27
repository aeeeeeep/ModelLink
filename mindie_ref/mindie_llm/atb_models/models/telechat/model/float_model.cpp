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
#include "float_model.h"
#include <nlohmann/json.hpp>
#include <atb/atb_infer.h>
#include "telechat/layer/embedding_layer.h"
#include "telechat/layer/float_layer.h"

namespace atb_speed {
namespace telechat {

const int WEIGHT_COUNT_PER_LAYER = 10;
const int WORD_EMBEDDING_WEIGHT_COUNT = 1;
const int FINAL_RMSNORM_WEIGHT_COUNT = 1;
const int OUT_LM_HEAD_WEIGHT_COUNT = 1;
const int INTERNAL_TENSOR_COUNT_BEFORE_LAYER = 3;
const int OPERATION_COUNT_BEFORE_LAYER = 1;
const int OPERATION_COUNT_AFTER_LAYER = 2;
const int OUT_TENSOR_NUM = 1;

enum InTensorId {
    IN_TENSOR_INPUT_IDS = 0,
    IN_TENSOR_POSITIONIDS,
    IN_TENSOR_COSTABLE,
    IN_TENSOR_SINTABLE,
    IN_TENSOR_ATTENTIONMASK,
    IN_TENSOR_PAST_KEY,
    IN_TENSOR_PAST_VALUE,
    IN_TOKENOFFSET,
    IN_SEQLEN,
    IN_HOLDER,
    IN_MAX_TENSOR
};

void FloatFAModel::FloatFAParam::FromString(const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);
    rmsNormEps = paramJson["rmsNormEps"].get<float>();
    headNum = paramJson["headNum"].get<int>();
    dk = paramJson["dk"].get<int>();
    layerNum = paramJson["layerNum"].get<int>();
    rank = paramJson["rank"].get<int>();
    rankSize = paramJson["rankSize"].get<int>();
    ATB_LOG(INFO) << "FloatFAModel param rmsNormEps:" << rmsNormEps << ", headNum:" << headNum << ", dk:" << dk
                  << ", layerNum:" << layerNum;
}

FloatFAModel::FloatFAModel(const std::string &param) : Model("FloatFAModel", param)
{
    param_.FromString(param);
}

FloatFAModel::~FloatFAModel() {}

uint32_t FloatFAModel::GetInputNum() const
{
    return graph_.inTensors.size();
}
uint32_t FloatFAModel::GetOutputNum() const
{
    return graph_.outTensors.size();
}

atb::Status FloatFAModel::InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
                                     std::vector<atb::TensorDesc> &outTensorDescs)
{
    if (outTensorDescs.size() != GetOutputNum()) {
        return atb::ERROR_INVALID_GRAPH;
    }

    const atb::TensorDesc &keyTensorDesc = inTensorDescs.at(IN_TENSOR_PAST_KEY);
    const atb::TensorDesc &valueTensorDesc = inTensorDescs.at(IN_TENSOR_PAST_KEY + param_.layerNum);
    // bs seqlen vocabsize
    outTensorDescs.at(0) = graph_.weightTensors.at(0).desc;
    outTensorDescs.at(0).shape.dimNum = 3;
    outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(IN_TENSOR_INPUT_IDS).shape.dims[0];
    outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(IN_TENSOR_INPUT_IDS).shape.dims[1];
    outTensorDescs.at(0).shape.dims[2] = graph_.weightTensors.at(graph_.weightTensors.size() - 1).desc.shape.dims[1];

    return atb::NO_ERROR;
}

int64_t FloatFAModel::BuildGraph()
{
    ATB_LOG(INFO) << "Enter Telechat FloatFAModel BuildGraph";

    const int weightTensorSize = WORD_EMBEDDING_WEIGHT_COUNT + WEIGHT_COUNT_PER_LAYER * param_.layerNum +
                                 FINAL_RMSNORM_WEIGHT_COUNT + OUT_LM_HEAD_WEIGHT_COUNT;
    graph_.weightTensors.resize(weightTensorSize);

    const int inTensorSize = IN_MAX_TENSOR + param_.layerNum;
    graph_.inTensors.resize(inTensorSize);

    const int outTensorSize = OUT_TENSOR_NUM;
    graph_.outTensors.resize(outTensorSize);

    const int nodeSize = OPERATION_COUNT_BEFORE_LAYER + param_.layerNum + OPERATION_COUNT_AFTER_LAYER;
    graph_.nodes.resize(nodeSize);

    const int internalTensorSize = graph_.nodes.size() + 1;
    graph_.internalTensors.resize(internalTensorSize);

    int nodeId = 0;
    atb::Operation *op = nullptr;

    auto &embeddingNode = graph_.nodes.at(nodeId++);
    EmbeddingLayerParam embeddingLayerParam;
    EmbeddingLayer(embeddingLayerParam, &op);
    embeddingNode.operation.reset(op);
    embeddingNode.inTensors = { &graph_.weightTensors.at(0), &graph_.inTensors.at(IN_TENSOR_INPUT_IDS),
                                &graph_.inTensors.at(IN_TENSOR_COSTABLE), &graph_.inTensors.at(IN_TENSOR_SINTABLE),
                                &graph_.inTensors.at(IN_TENSOR_POSITIONIDS) };
    embeddingNode.outTensors = { &graph_.internalTensors.at(0), &graph_.internalTensors.at(1),
                                 &graph_.internalTensors.at(2) };
    atb::Tensor *firstInTensor = &graph_.internalTensors.at(0);
    atb::Tensor *cosEmbedTensor = &graph_.internalTensors.at(1);
    atb::Tensor *sinEmbedTensor = &graph_.internalTensors.at(2);
    for (int layerId = 0; layerId < param_.layerNum; ++layerId) {
        auto &layerNode = graph_.nodes.at(nodeId++);

        FloatFALayerParam layerParam;

        layerParam.rmsNormEps = param_.rmsNormEps;
        layerParam.headNum = param_.headNum;
        layerParam.dk = param_.dk;
        layerParam.rank = param_.rank;
        layerParam.rankSize = param_.rankSize;

        FloatFALayer(layerParam, &op);
        layerNode.operation.reset(op);
        layerNode.inTensors.resize(layerNode.operation->GetInputNum());

        size_t inTensorId = 0;
        layerNode.inTensors.at(inTensorId++) = firstInTensor;
        for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
            layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(
                WORD_EMBEDDING_WEIGHT_COUNT + layerId * WEIGHT_COUNT_PER_LAYER + weightTensorId);
        }
        layerNode.inTensors.at(inTensorId++) = cosEmbedTensor;
        layerNode.inTensors.at(inTensorId++) = sinEmbedTensor;
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ATTENTIONMASK);  // attentionMaskTensor
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PAST_KEY);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PAST_VALUE);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TOKENOFFSET);           // tokenoffset
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_SEQLEN);                // seqlen
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_HOLDER);                // holder
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_MAX_TENSOR + layerId);  // layerid

        layerNode.outTensors = { &graph_.internalTensors.at(INTERNAL_TENSOR_COUNT_BEFORE_LAYER + layerId) };
        firstInTensor = layerNode.outTensors.at(0);
    }

    auto &finalRmsNormNode = graph_.nodes.at(nodeId++);
    atb::infer::RmsNormParam rmsNormParam;
    rmsNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    rmsNormParam.normParam.epsilon = param_.rmsNormEps;
    CREATE_OPERATION(rmsNormParam, &op);
    finalRmsNormNode.operation.reset(op);
    const int finalRmsNormWeightTensorId =
        graph_.weightTensors.size() - FINAL_RMSNORM_WEIGHT_COUNT - OUT_LM_HEAD_WEIGHT_COUNT;
    const int finalRmsNormOutTensorId = internalTensorSize - 1;
    finalRmsNormNode.inTensors = { firstInTensor, &graph_.weightTensors.at(finalRmsNormWeightTensorId) };
    finalRmsNormNode.outTensors = { &graph_.internalTensors.at(finalRmsNormOutTensorId) };

    auto &lmHeadNode = graph_.nodes.at(nodeId++);
    atb::infer::LinearParam linearParam = { false, true, false };
    CREATE_OPERATION(linearParam, &op);
    lmHeadNode.operation.reset(op);
    const int lmHeadWeightTensorId = graph_.weightTensors.size() - OUT_LM_HEAD_WEIGHT_COUNT;
    lmHeadNode.inTensors = { &graph_.internalTensors.at(finalRmsNormOutTensorId),
                             &graph_.weightTensors.at(lmHeadWeightTensorId) };
    lmHeadNode.outTensors = { &graph_.outTensors.at(0) };
}

atb::Status FloatFAModel::ParseParam(const std::string &param)
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

    ATB_LOG(INFO) << "ParseParam tokenOffset:" << tokenOffset_ << ", seqLen:" << seqLen_;

    return atb::NO_ERROR;
}

atb::Status FloatFAModel::BindParamHostTensor(uint32_t nodeId)
{
    if (nodeId < OPERATION_COUNT_BEFORE_LAYER || nodeId >= OPERATION_COUNT_BEFORE_LAYER + param_.layerNum) {
        return atb::NO_ERROR;
    }

    auto &node = graph_.nodes.at(nodeId);

    const uint32_t tokenOffsetTensorId = 16;

    const uint32_t seqLenTensorId = 17;

    node.variantPack.inTensors.at(tokenOffsetTensorId).hostData = tokenOffset_.data();
    node.variantPack.inTensors.at(seqLenTensorId).hostData = seqLen_.data();

    return atb::NO_ERROR;
}
}  // namespace telechat
}  // namespace atb_speed