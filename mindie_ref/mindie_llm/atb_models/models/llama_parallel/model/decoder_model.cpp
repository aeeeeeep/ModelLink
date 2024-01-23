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
#include "nlohmann/json.hpp"
#include "atb/atb_infer.h"
#include "atb_speed/log.h"
#include "models/llama_parallel/operation/embedding.h"
#include "models/llama_parallel/operation/rms_norm.h"
#include "models/llama_parallel/operation/lmhead.h"
#include "models/llama_parallel/layer/decoder_layer.h"
#include "models/llama_parallel/model/decoder_model.h"

namespace atb_speed {
namespace llama_parallel {

// Weight count
const int WEIGHT_COUNT_PER_LAYER = 30;
const int WEIGHT_COUNT_WORD_EMBEDDINGNODE = 1;
const int WEIGHT_COUNT_POST_NORM = 1;
const int WEIGHT_COUNT_LM_HEAD = 1;

// Operation count
const int OPERATION_COUNT_BEFORE_LAYER = 1;
const int OPERATION_COUNT_AFTER_LAYER = 2;  // RmsNorm + LmHead

void DecoderModel::Param::FromString(const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);
    isFA = paramJson["isFA"].get<bool>();
    isPrefill = paramJson["isPrefill"].get<bool>();
    isBF16 = paramJson["isBF16"].get<bool>();
    isPack = paramJson["isPack"].get<bool>();
    isEmbeddingParallel = paramJson["isEmbeddingParallel"].get<bool>();
    isLmHeadParallel = paramJson["isLmHeadParallel"].get<bool>();
    quantType = paramJson["quantType"].get<int>();
    rmsNormEps = paramJson["rmsNormEps"].get<float>();
    numAttentionHeadsPerRank = paramJson["numAttentionHeadsPerRank"].get<int>();
    hiddenSizePerAttentionHead = paramJson["hiddenSizePerAttentionHead"].get<int>();
    numHiddenLayers = paramJson["numHiddenLayers"].get<int>();
    numKeyValueHeadsPerRank = paramJson["numKeyValueHeadsPerRank"].get<int>();
    rank = paramJson["rank"].get<int>();
    worldSize = paramJson["worldSize"].get<int>();
    backend = paramJson["backend"].get<std::string>();
    for (auto item : paramJson["tokenOffset"]) {
        tokenOffset.push_back(item.get<int>());
    }
    for (auto item : paramJson["seqLen"]) {
        seqLen.push_back(item.get<int>());
    }
    ATB_LOG(INFO) << "DecoderModel param" << ", isFA:" << isFA << ", isPrefill:" << isPrefill << ", isBF16:" << isBF16 << ", isPack:" << isPack
                  << ", isEmbeddingParallel: " << isEmbeddingParallel << ", isLmHeadParallel: " << isLmHeadParallel
                  << ", quantType:" << quantType << ", rmsNormEps:" << rmsNormEps << ", numAttentionHeadsPerRank:" << numAttentionHeadsPerRank
                  << ", hiddenSizePerAttentionHead:" << hiddenSizePerAttentionHead << ", numHiddenLayers:" << numHiddenLayers
                  << ", numKeyValueHeadsPerRank:" << numKeyValueHeadsPerRank
                  << ", rank:" << rank << ", worldSize:" << worldSize << ", backend:" << backend
                  << ", tokenOffset:" << tokenOffset << ", seqLen:" << seqLen;
}

DecoderModel::DecoderModel(const std::string &param) : Model("DecoderModel", param)
{
    param_.FromString(param);
    modelName_ += param_.isPrefill ? "_Prefill" : "_Decoder";
}

DecoderModel::~DecoderModel() {}

uint32_t DecoderModel::GetInputNum() const { return graph_.inTensors.size(); }

uint32_t DecoderModel::GetOutputNum() const { return graph_.outTensors.size(); }

atb::Status DecoderModel::InferShape(const std::vector<atb::TensorDesc> &inTensorDescs, std::vector<atb::TensorDesc> &outTensorDescs)
{
    ATB_LOG(INFO) << "Enter DecoderModel InferShape";
    if (outTensorDescs.size() != GetOutputNum()) {
        return atb::ERROR_INVALID_GRAPH;
    }

    const int64_t vocabSizePerRank = graph_.weightTensors.at(graph_.weightTensors.size() - 1).desc.shape.dims[0];
    // FA: [batchSize, seqLen, vocabSize] PA: [seqLen, vocabSisze]
    outTensorDescs.at(0).dtype = graph_.weightTensors.at(graph_.weightTensors.size() - 1).desc.dtype;
    outTensorDescs.at(0).format = graph_.weightTensors.at(graph_.weightTensors.size() - 1).desc.format;
    outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum + 1;

    if (param_.isFA) {
        outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
        outTensorDescs.at(0).shape.dims[1] = 1;
    } else {
        outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
        if (param_.isPrefill) {
            outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(graph_.inTensors.size() - 1).shape.dims[0];
        }
    }

    if (param_.isLmHeadParallel) {
        outTensorDescs.at(0).shape.dims[outTensorDescs.at(0).shape.dimNum - 1] = vocabSizePerRank * param_.worldSize;
    } else {
        outTensorDescs.at(0).shape.dims[outTensorDescs.at(0).shape.dimNum - 1] = vocabSizePerRank;
    }

    return atb::NO_ERROR;
}

int64_t DecoderModel::BuildGraph()
{
    // define inTensor
    int inTensorIdx = 0;
    int IN_TENSOR_INPUT_IDS = inTensorIdx++;          // idx: 0, shape: FA: [batchSize, seqLen] PA: [seqLen]
    int IN_TENSOR_POSITION_IDS = inTensorIdx++;       // idx: 1, shape: FA: [batchSize, seqLen] PA: [seqLen]
    int IN_TENSOR_COS_TABLE = inTensorIdx++;          // idx: 2, shape: FA: [maxPositionEmbeddings, hiddenSizePerAttentionHead] PA: [maxInputLength, hiddenSizePerAttentionHead]
    int IN_TENSOR_SIN_TABLE = inTensorIdx++;          // idx: 3, shape: FA: [maxPositionEmbeddings, hiddenSizePerAttentionHead] PA: [maxInputLength, hiddenSizePerAttentionHead]
    int IN_TENSOR_ATTENTION_MASK = inTensorIdx++;     // idx: 4, shape: FA: [batchSize, maxPositionEmbeddings, maxPositionEmbeddings] PA: [maxInputLength, maxInputLength]
    int IN_TENSOR_BLOCK_TABLES = inTensorIdx++;       // idx: 5, shape: [4, 9]; PA所需入参
    int IN_TENSOR_SLOTS = inTensorIdx++;              // idx: 6, shape: [seqLen]; PA所需入参
    int IN_TENSOR_KV_CACHE_IDX = inTensorIdx++;       // idx: 7, shape: [1]; FA所需入参
    int IN_TENSOR_TOKEN_OFFSET = inTensorIdx++;       // idx: 8, shape: [batchSize]; FA所需入参
    int IN_TENSOR_PLACE_HOLDER = inTensorIdx++;       // idx: 9, shape: [1]
    int IN_TENSOR_BETA = inTensorIdx++;               // idx: 10, shape: [hiddenSize]; Quant所需参数
    int IN_TENSOR_SEQ_LEN = inTensorIdx++;            // idx: 11, shape: FA: [batchSize] PA: [4]
    int IN_TENSOR_LOGTIS_INDICES = inTensorIdx++;     // idx: 12, shape: FA: [batchSize]  PA: [4]

    // define internelTensor
    int internelTensorIdx = 0;
    int INTERNEL_TENSOR_HIDDEN_STATES = internelTensorIdx++;              // idx: 0, shape: FA: [batchSize, seqLen, hiddenSize] PA: [seqLen, hiddenSize]
    int INTERNEL_TENSOR_COS_TABLE = internelTensorIdx++;                  // idx: 1, shape: [batchSize * seqLen, hiddenSizePerAttentionHead]
    int INTERNEL_TENSOR_SIN_TABLE = internelTensorIdx++;                  // idx: 2, shape: [batchSize * seqLen, hiddenSizePerAttentionHead]
    int INTERNEL_TENSOR_LAYER_OUT_BASE = internelTensorIdx++;             // idx: [3, 3 + numHiddenLayers), shape: FA: [batchSize, seqLen, hiddenSize] PA: [seqLen, hiddenSize]
    internelTensorIdx = internelTensorIdx + param_.numHiddenLayers - 1;
    int INTERNEL_TENSOR_FINAL_NORM_OUT = internelTensorIdx++;             // idx: 3 + numHiddenLayers, shape: FA: [batchSize, seqLen, hiddenSize] PA: [seqLen, hiddenSize]

    // set size
    const int weightTensorSize =
        WEIGHT_COUNT_WORD_EMBEDDINGNODE + WEIGHT_COUNT_PER_LAYER * param_.numHiddenLayers + WEIGHT_COUNT_POST_NORM + WEIGHT_COUNT_LM_HEAD;
    graph_.weightTensors.resize(weightTensorSize);

    graph_.inTensors.resize(inTensorIdx);
    graph_.outTensors.resize(1);
    graph_.internalTensors.resize(internelTensorIdx);

    graph_.kCacheTensors.resize(param_.numHiddenLayers);
    graph_.vCacheTensors.resize(param_.numHiddenLayers);

    const int nodeSize = param_.numHiddenLayers + OPERATION_COUNT_BEFORE_LAYER + OPERATION_COUNT_AFTER_LAYER;
    graph_.nodes.resize(nodeSize);

    ATB_LOG(INFO) << "DecoderModel build graph begin";
    int nodeId = 0;
    atb::Operation *op = nullptr;

    auto &embeddingNode = graph_.nodes.at(nodeId++);
    atb_speed::llama_parallel::EmbeddingParam embeddingParam;
    embeddingParam.unpadInputs = !param_.isFA;
    if (param_.isEmbeddingParallel) {
        embeddingParam.rank = param_.rank;
        embeddingParam.worldSize = param_.worldSize;
    };
    embeddingParam.backend = param_.backend;
    atb_speed::llama_parallel::Embedding(embeddingParam, &op);
    embeddingNode.operation.reset(op);
    embeddingNode.inTensors = {&graph_.weightTensors.at(0),                    // shape: [vocabSize + 1, hiddenSize]
                               &graph_.inTensors.at(IN_TENSOR_INPUT_IDS),
                               &graph_.inTensors.at(IN_TENSOR_COS_TABLE),
                               &graph_.inTensors.at(IN_TENSOR_SIN_TABLE),
                               &graph_.inTensors.at(IN_TENSOR_POSITION_IDS)};
    embeddingNode.outTensors = {&graph_.internalTensors.at(INTERNEL_TENSOR_HIDDEN_STATES),
                                &graph_.internalTensors.at(INTERNEL_TENSOR_COS_TABLE),
                                &graph_.internalTensors.at(INTERNEL_TENSOR_SIN_TABLE)};

    atb::Tensor *firstInTensor = &graph_.internalTensors.at(INTERNEL_TENSOR_HIDDEN_STATES);
    for (int layerId = 0; layerId < param_.numHiddenLayers; ++layerId) {
        auto &layerNode = graph_.nodes.at(nodeId++);
        atb_speed::llama_parallel::DecoderLayerParam layerParam;
        layerParam.isFA = param_.isFA;
        layerParam.isPrefill = param_.isPrefill;
        layerParam.isBF16 = param_.isBF16;
        layerParam.isPack = param_.isPack;
        layerParam.quantType = param_.quantType;
        layerParam.rmsNormEps = param_.rmsNormEps;
        layerParam.numAttentionHeadsPerRank = param_.numAttentionHeadsPerRank;
        layerParam.hiddenSizePerAttentionHead = param_.hiddenSizePerAttentionHead;
        layerParam.numKeyValueHeadsPerRank = param_.numKeyValueHeadsPerRank;
        layerParam.rank = param_.rank;
        layerParam.worldSize = param_.worldSize;
        layerParam.backend = param_.backend;
        atb_speed::llama_parallel::DecoderLayer(layerParam, &op);

        layerNode.operation.reset(op);
        layerNode.inTensors.resize(layerNode.operation->GetInputNum());
        size_t inTensorId = 0;
        layerNode.inTensors.at(inTensorId++) = firstInTensor;
        for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
            layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(
                layerId * WEIGHT_COUNT_PER_LAYER + weightTensorId + WEIGHT_COUNT_WORD_EMBEDDINGNODE);
        }
        layerNode.inTensors.at(inTensorId++) = &graph_.internalTensors.at(INTERNEL_TENSOR_COS_TABLE);
        layerNode.inTensors.at(inTensorId++) = &graph_.internalTensors.at(INTERNEL_TENSOR_SIN_TABLE);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ATTENTION_MASK);
        layerNode.inTensors.at(inTensorId++) = &graph_.kCacheTensors.at(layerId);
        layerNode.inTensors.at(inTensorId++) = &graph_.vCacheTensors.at(layerId);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SEQ_LEN);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PLACE_HOLDER);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_BETA);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_TOKEN_OFFSET);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_KV_CACHE_IDX);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_BLOCK_TABLES);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SLOTS);

        layerNode.outTensors = {&graph_.internalTensors.at(INTERNEL_TENSOR_LAYER_OUT_BASE + layerId)};
        firstInTensor = layerNode.outTensors.at(0);
    }

    auto &finalNormNode = graph_.nodes.at(nodeId++);
    atb_speed::llama_parallel::FusionRmsNormParam fusionRmsNormParam;
    fusionRmsNormParam.quantType = param_.quantType;
    fusionRmsNormParam.rmsNormEps = param_.rmsNormEps;
    FusionRmsNorm(fusionRmsNormParam, &op);
    finalNormNode.operation.reset(op);
    const int finalLayerNormWeightTensorId =
        graph_.weightTensors.size() - WEIGHT_COUNT_POST_NORM - WEIGHT_COUNT_LM_HEAD;
    finalNormNode.inTensors = {
        firstInTensor,
        &graph_.weightTensors.at(finalLayerNormWeightTensorId), &graph_.inTensors.at(IN_TENSOR_BETA)  // shape: [hiddenSize]
    };
    finalNormNode.outTensors = {&graph_.internalTensors.at(INTERNEL_TENSOR_FINAL_NORM_OUT)};  // shape: FA: [batchSize, seqLen, hiddenSize] PA: [seqLen, hiddenSize]

    auto &lmHeadNode = graph_.nodes.at(nodeId++);
    atb_speed::llama_parallel::LmHeadParam lmHeadParam;
    lmHeadParam.unpadInputs = !param_.isFA;
    lmHeadParam.gatherAhead = param_.isPrefill;
    lmHeadParam.hiddenSizePerAttentionHead = param_.hiddenSizePerAttentionHead;
    lmHeadParam.linearParallelParam.fusionLinearParam.quantType = false;  // LmHead未接入量化
    if (param_.isLmHeadParallel && param_.worldSize > 1) {
        lmHeadParam.linearParallelParam.parallelType = atb_speed::llama_parallel::COLUMN_PARALLEL;
        lmHeadParam.linearParallelParam.rank = param_.rank;
        lmHeadParam.linearParallelParam.worldSize = param_.worldSize;
        lmHeadParam.linearParallelParam.backend = param_.backend;
    }
    LmHead(lmHeadParam, &op);
    lmHeadNode.operation.reset(op);
    const int finalLinearWeightTensorId = graph_.weightTensors.size() - WEIGHT_COUNT_LM_HEAD;
    lmHeadNode.inTensors = {
        &graph_.internalTensors.at(INTERNEL_TENSOR_FINAL_NORM_OUT),
        &graph_.weightTensors.at(finalLinearWeightTensorId), // shape: [vocabSizePerRank, hiddenSize]
        &graph_.inTensors.at(IN_TENSOR_PLACE_HOLDER),        // LmHead未接入量化，量化权重使用placeholder代替
        &graph_.inTensors.at(IN_TENSOR_PLACE_HOLDER),
        &graph_.inTensors.at(IN_TENSOR_PLACE_HOLDER),
        &graph_.inTensors.at(IN_TENSOR_LOGTIS_INDICES)
    };
    lmHeadNode.outTensors = {&graph_.outTensors.at(0)};      // shpae: FA: [batchSize, seqLen, vocabSize] PA: [seqLen, vocabSize]

    ATB_LOG(INFO) << "DecoderModel build graph success";
    return atb::NO_ERROR;
}

atb::Status DecoderModel::ParseParam(const std::string &param)
{
    ATB_LOG(INFO) << "ParseParam start.";
    nlohmann::json paramJson = nlohmann::json::parse(param);

    tokenOffset_.clear();
    for (auto item : paramJson["tokenOffset"]) {
        tokenOffset_.push_back(item.get<int>());
        ATB_LOG(INFO) << "token offset value: " << item;
    }

    seqLen_.clear();
    for (auto item : paramJson["seqLen"]) {
        seqLen_.push_back(item.get<int>());
        ATB_LOG(INFO) << "Prefill" << paramJson["isPrefill"] << "seqLen value: " << item;
    }
    ATB_LOG(INFO) << "ParseParam end.";
    return atb::NO_ERROR;
}

atb::Status DecoderModel::BindParamHostTensor(uint32_t nodeId)
{
    ATB_LOG(INFO) << "BindParamHostTensor";
    ATB_LOG(INFO) << "nodeId = " << nodeId;

    if (nodeId < OPERATION_COUNT_BEFORE_LAYER || nodeId >= OPERATION_COUNT_BEFORE_LAYER + param_.numHiddenLayers) {
        return atb::NO_ERROR;
    }

    auto &node = graph_.nodes.at(nodeId);
    const uint32_t tokenOffsetTensorId = DecoderLayerTensorIdx::IN_TOKEN_OFFSET;
    const uint32_t seqLenTensorId = DecoderLayerTensorIdx::IN_SEQ_LEN;
    node.variantPack.inTensors.at(tokenOffsetTensorId).hostData = tokenOffset_.data();
    node.variantPack.inTensors.at(seqLenTensorId).hostData = seqLen_.data();
    ATB_LOG(INFO) << "BindParamHostTensor end";
    return atb::NO_ERROR;
}
} // namespace llama_parallel
} // namespace atb_speed