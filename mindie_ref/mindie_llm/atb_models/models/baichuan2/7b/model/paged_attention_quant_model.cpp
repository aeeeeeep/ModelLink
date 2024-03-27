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
#include "paged_attention_quant_model.h"
#include "vector"
#include <atb/atb_infer.h>
#include <nlohmann/json.hpp>
#include "layers/operations/word_embedding.h"
#include "layers/operations/positional_embedding.h"
#include "layers/operations/lmhead.h"
#include "models/baichuan2/7b/layer/paged_attention_quant_layer.h"
#include "parallel_lmhead.h"
#include "atb_speed/utils/model_factory.h"

namespace atb_speed {
namespace baichuan2_7b {
REGISTER_MODEL(baichuan2_7b, PagedAttentionQuantModel);

const int WEIGHT_COUNT_PER_LAYER = 50;
const int WEIGHT_COUNT_WORD_EMBEDDINGNODE = 1;
const int WEIGHT_COUNT_POST_NORM = 1;
const int WEIGHT_COUNT_LM_HEAD = 1;

// Operation count
const int OPERATION_COUNT_BEFORE_LAYER = 2; // Word Embedding + Positional Embedding
const int OPERATION_COUNT_AFTER_LAYER = 2;  // RmsNorm + LmHead

enum PAModelInTensorId : int {
    IN_TENSOR_INPUTIDS = 0,
    IN_TENSOR_POSITIONID,
    IN_TENSOR_COSEMBED,
    IN_TENSOR_SINEMBED,
    IN_TENSOR_ATTENTIONMASK,
    IN_TENSOR_BLOCK_TABLES,
    IN_TENSOR_SLOTS,
    IN_TENSOR_INPUT_LENGTHS,
    IN_TENSOR_LOGTIS_INDICES,
    IN_HOLDER,
    IN_TENSOR_MAX,
};

enum OutTensorId : int {
    OUT_TENSOR_HIDDENSTATES = 0,
    OUT_TENSOR_MAX,
};

void PagedAttentionQuantModel::Param::FromString(const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);
    isFA = paramJson["isFA"].get<bool>();
    isPrefill = paramJson["isPrefill"].get<bool>();
    isBF16 = paramJson["isBF16"].get<bool>();
    isEmbeddingParallel = paramJson["isEmbeddingParallel"].get<bool>();
    isLmHeadParallel = paramJson["isLmHeadParallel"].get<bool>();
    supportSwiGLU = paramJson["supportSwiGLU"].get<bool>();
    rmsNormEps = paramJson["rmsNormEps"].get<float>();
    numAttentionHeadsPerRank = paramJson["numAttentionHeadsPerRank"].get<int>();
    hiddenSizePerAttentionHead = paramJson["hiddenSizePerAttentionHead"].get<int>();
    numHiddenLayers = paramJson["numHiddenLayers"].get<int>();
    numKeyValueHeadsPerRank = paramJson["numKeyValueHeadsPerRank"].get<int>();
    rank = paramJson["rank"].get<int>();
    rankSize = paramJson["rankSize"].get<int>();
    backend = paramJson["backend"].get<std::string>();
    for (auto item : paramJson["tokenOffset"]) {
        tokenOffset.push_back(item.get<int>());
    }
    for (auto item : paramJson["seqLen"]) {
        seqLen.push_back(item.get<int>());
    }
    for (auto item : paramJson["packQuantType"]) {
        packQuantType.push_back(item.get<std::vector<int>>());
    }
    for (auto item : paramJson["linearQuantType"]) {
        linearQuantType.push_back(item.get<std::vector<int>>());
    }
    ATB_LOG(INFO) << "DecoderModel param"
                  << ", isFA:" << isFA << ", isPrefill:" << isPrefill << ", isBF16:" << isBF16 <<
        ", isEmbeddingParallel: " << isEmbeddingParallel << ", isLmHeadParallel: " << isLmHeadParallel <<
        ", supportSwiGLU: " << supportSwiGLU << ", rmsNormEps:" << rmsNormEps << ", numAttentionHeadsPerRank:" <<
        numAttentionHeadsPerRank << ", hiddenSizePerAttentionHead:" << hiddenSizePerAttentionHead <<
        ", numHiddenLayers:" << numHiddenLayers << ", numKeyValueHeadsPerRank:" << numKeyValueHeadsPerRank <<
        ", rank:" << rank << ", rankSize:" << rankSize << ", backend:" << backend << ", tokenOffset:" << tokenOffset <<
        ", seqLen:" << seqLen;
}

PagedAttentionQuantModel::PagedAttentionQuantModel(const std::string &param) : Model("Baichuan2_7BPAModel", param)
{
    param_.FromString(param);
    modelName_ += param_.isPrefill ? "_Prefill" : "_Decoder";
}

PagedAttentionQuantModel::~PagedAttentionQuantModel() = default;

uint32_t PagedAttentionQuantModel::GetInputNum() const
{
    return graph_.inTensors.size();
}

uint32_t PagedAttentionQuantModel::GetOutputNum() const
{
    return graph_.outTensors.size();
}

atb::Status PagedAttentionQuantModel::InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
    std::vector<atb::TensorDesc> &outTensorDescs)
{
    ATB_LOG(INFO) << "Enter DecoderModel InferShape";
    if (outTensorDescs.size() != GetOutputNum()) {
        return atb::ERROR_INVALID_GRAPH;
    }

    const int64_t vocabSizePerRank = graph_.weightTensors.at(graph_.weightTensors.size() - 1).desc.shape.dims[0];
    // FA: [batchSize, seqLen, vocabSize] PA: [seqLen, vocabSisze]
    outTensorDescs.at(0).dtype = graph_.weightTensors.at(graph_.weightTensors.size() - 1).desc.dtype;
    outTensorDescs.at(0).format = graph_.weightTensors.at(0).desc.format;
    outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum + 1;

    outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
    if (param_.isFA) { // unpadInputs = false
        outTensorDescs.at(0).shape.dims[1] =
            param_.isPrefill ? inTensorDescs.at(graph_.inTensors.size() - 1).shape.dims[0] : 1;
    } else { // unpadInputs = true
        if (param_.isPrefill) {
            outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(graph_.inTensors.size() - 1).shape.dims[0];
        }
    }

    if (param_.isLmHeadParallel) {
        outTensorDescs.at(0).shape.dims[outTensorDescs.at(0).shape.dimNum - 1] = vocabSizePerRank * param_.rankSize;
    } else {
        outTensorDescs.at(0).shape.dims[outTensorDescs.at(0).shape.dimNum - 1] = vocabSizePerRank;
    }

    return atb::NO_ERROR;
}

int64_t PagedAttentionQuantModel::BuildGraph()
{
    // define inTensor
    int inTensorIdx = 0;
    // idx: 0, shape: FA: [batchSize, seqLen] PA: [seqLen]
    int IN_TENSOR_INPUT_IDS = inTensorIdx++;
    // idx: 1, shape: FA: [batchSize, seqLen] PA: [seqLen]
    int IN_TENSOR_POSITION_IDS = inTensorIdx++;
    // idx: 2, shape: FA: [maxPositionEmbeddings, hiddenSizePerAttentionHead]
    // PA: [maxInputLength, hiddenSizePerAttentionHead]
    int IN_TENSOR_COS_TABLE = inTensorIdx++;
    // idx: 3, shape: FA: [maxPositionEmbeddings, hiddenSizePerAttentionHead]
    // PA: [maxInputLength, hiddenSizePerAttentionHead]
    int IN_TENSOR_SIN_TABLE = inTensorIdx++;
    // idx: 4, shape: FA: [batchSize, maxPositionEmbeddings, maxPositionEmbeddings]
    // PA: [maxInputLength, maxInputLength]
    int IN_TENSOR_ATTENTION_MASK = inTensorIdx++;
    // idx: 5, shape: [4, 9]; PA所需入参
    int IN_TENSOR_BLOCK_TABLES = inTensorIdx++;
    // idx: 6, shape: [seqLen]; PA所需入参
    int IN_TENSOR_SLOTS = inTensorIdx++;
    // idx: 7, shape: [1]; FA所需入参
    int IN_TENSOR_KV_CACHE_IDX = inTensorIdx++;
    // idx: 8, shape: [batchSize]; FA所需入参
    int IN_TENSOR_TOKEN_OFFSET = inTensorIdx++;
    // idx: 9, shape: [1]
    int IN_TENSOR_PLACE_HOLDER = inTensorIdx++;
    // idx: 10, shape: FA: [batchSize] PA: [4]
    int IN_TENSOR_SEQ_LEN = inTensorIdx++;
    // idx: 11, shape: FA: [batchSize]  PA: [4]
    int IN_TENSOR_LOGTIS_INDICES = inTensorIdx++;

    // define internelTensor
    int internelTensorIdx = 0;
    // idx: 0, shape: FA: [batchSize, seqLen, hiddenSize] PA: [seqLen, hiddenSize]
    int INTERNEL_TENSOR_HIDDEN_STATES = internelTensorIdx++;
    // idx: 1, shape: [batchSize * seqLen, hiddenSizePerAttentionHead]
    int INTERNEL_TENSOR_COS_EMB = internelTensorIdx++;
    // idx: 2, shape: [batchSize * seqLen, hiddenSizePerAttentionHead]
    int INTERNEL_TENSOR_SIN_EMB = internelTensorIdx++;
    // idx: [3, 3 + numHiddenLayers), shape: FA: [batchSize, seqLen, hiddenSize] PA: [seqLen, hiddenSize]
    int INTERNEL_TENSOR_LAYER_OUT_BASE = internelTensorIdx++;
    internelTensorIdx = internelTensorIdx + param_.numHiddenLayers - 1;
    // idx: 3 + numHiddenLayers, shape: FA: [batchSize, seqLen, hiddenSize] PA: [seqLen, hiddenSize]
    int INTERNEL_TENSOR_FINAL_NORM_OUT = internelTensorIdx++;

    // set size
    const int weightTensorSize = WEIGHT_COUNT_WORD_EMBEDDINGNODE + WEIGHT_COUNT_PER_LAYER * param_.numHiddenLayers +
        WEIGHT_COUNT_POST_NORM + WEIGHT_COUNT_LM_HEAD;
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

    auto &wordEmbeddingNode = graph_.nodes.at(nodeId++);
    atb_speed::common::WordEmbeddingParam wordEmbeddingParam;
    wordEmbeddingParam.unpadInputs = !param_.isFA;
    if (param_.isEmbeddingParallel) {
        wordEmbeddingParam.tensorParallelInfo = { param_.rank, param_.rankSize, param_.backend };
    };
    atb_speed::common::WordEmbedding(wordEmbeddingParam, &op);
    wordEmbeddingNode.operation.reset(op);
    wordEmbeddingNode.inTensors = { &graph_.weightTensors.at(0), // shape: [vocabSize + 1, hiddenSize]
        &graph_.inTensors.at(IN_TENSOR_INPUT_IDS) };
    wordEmbeddingNode.outTensors = { &graph_.internalTensors.at(INTERNEL_TENSOR_HIDDEN_STATES) };

    auto &peGatherNode = graph_.nodes.at(nodeId++);
    atb_speed::common::PositionalEmbeddingGather(&op);
    peGatherNode.operation.reset(op);
    peGatherNode.inTensors = {
        &graph_.inTensors.at(IN_TENSOR_POSITION_IDS),
        &graph_.inTensors.at(IN_TENSOR_COS_TABLE),
        &graph_.inTensors.at(IN_TENSOR_SIN_TABLE),
    };
    peGatherNode.outTensors = { &graph_.internalTensors.at(INTERNEL_TENSOR_COS_EMB),
        &graph_.internalTensors.at(INTERNEL_TENSOR_SIN_EMB) };

    atb::Tensor *firstInTensor = &graph_.internalTensors.at(INTERNEL_TENSOR_HIDDEN_STATES);
    ATB_LOG(INFO) << "Begin build layer";
    for (int layerId = 0; layerId < param_.numHiddenLayers; ++layerId) {
        auto &layerNode = graph_.nodes.at(nodeId++);
        atb_speed::baichuan2_7b::PAQuantLayerParam layerParam;
        layerParam.isFA = param_.isFA;
        ATB_LOG(INFO) << "layerParam.isFA:" << layerParam.isFA;
        layerParam.isPrefill = param_.isPrefill;
        ATB_LOG(INFO) << "layerParam.isPrefill:" << layerParam.isBF16;
        layerParam.isBF16 = param_.isBF16;
        ATB_LOG(INFO) << "layerParam.isBF16:" << layerParam.isBF16;
        layerParam.supportSwiGLU = param_.supportSwiGLU;
        ATB_LOG(INFO) << "layerParam.supportSwiGLU:" << layerParam.supportSwiGLU;
        ATB_LOG(INFO) << "param_.packQuantType[layerId]:" << param_.packQuantType[layerId];
        layerParam.packQuantType = param_.packQuantType[layerId];
        ATB_LOG(INFO) << "layerParam.packQuantType:" << layerParam.packQuantType;
        layerParam.linearQuantType = param_.linearQuantType[layerId];
        ATB_LOG(INFO) << "layerParam.linearQuantType:" << layerParam.linearQuantType;
        layerParam.rmsNormEps = param_.rmsNormEps;
        ATB_LOG(INFO) << "layerParam.rmsNormEps:" << layerParam.rmsNormEps;
        layerParam.numAttentionHeadsPerRank = param_.numAttentionHeadsPerRank;
        ATB_LOG(INFO) << "layerParam.numAttentionHeadsPerRank:" << layerParam.numAttentionHeadsPerRank;
        layerParam.hiddenSizePerAttentionHead = param_.hiddenSizePerAttentionHead;
        ATB_LOG(INFO) << "layerParam.hiddenSizePerAttentionHead:" << layerParam.hiddenSizePerAttentionHead;
        layerParam.numKeyValueHeadsPerRank = param_.numKeyValueHeadsPerRank;
        ATB_LOG(INFO) << "layerParam.numKeyValueHeadsPerRank:" << layerParam.numKeyValueHeadsPerRank;
        layerParam.rank = param_.rank;
        ATB_LOG(INFO) << "layerParam.rank:" << layerParam.rank;
        layerParam.rankSize = param_.rankSize;
        ATB_LOG(INFO) << "layerParam.rankSize:" << layerParam.rankSize;
        layerParam.backend = param_.backend;
        atb_speed::baichuan2_7b::PAQuantLayer(layerParam, &op);
        layerNode.operation.reset(op);
        layerNode.inTensors.resize(layerNode.operation->GetInputNum());
        size_t inTensorId = 0;
        layerNode.inTensors.at(inTensorId++) = firstInTensor;
        for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
            layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(layerId * WEIGHT_COUNT_PER_LAYER +
                weightTensorId + WEIGHT_COUNT_WORD_EMBEDDINGNODE);
        }
        layerNode.inTensors.at(inTensorId++) = &graph_.internalTensors.at(INTERNEL_TENSOR_COS_EMB);
        layerNode.inTensors.at(inTensorId++) = &graph_.internalTensors.at(INTERNEL_TENSOR_SIN_EMB);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ATTENTION_MASK);
        layerNode.inTensors.at(inTensorId++) = &graph_.kCacheTensors.at(layerId);
        layerNode.inTensors.at(inTensorId++) = &graph_.vCacheTensors.at(layerId);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SEQ_LEN);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PLACE_HOLDER);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_TOKEN_OFFSET);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_KV_CACHE_IDX);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_BLOCK_TABLES);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SLOTS);

        layerNode.outTensors = { &graph_.internalTensors.at(INTERNEL_TENSOR_LAYER_OUT_BASE + layerId) };
        firstInTensor = layerNode.outTensors.at(0);
    }

    auto &finalNormNode = graph_.nodes.at(nodeId++);
    atb::infer::RmsNormParam finalNormParam;
    finalNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    finalNormParam.normParam.epsilon = param_.rmsNormEps;
    CREATE_OPERATION(finalNormParam, &op);
    finalNormNode.operation.reset(op);
    const int finalLayerNormWeightTensorId =
        graph_.weightTensors.size() - WEIGHT_COUNT_POST_NORM - WEIGHT_COUNT_LM_HEAD;
    finalNormNode.inTensors = { firstInTensor, &graph_.weightTensors.at(finalLayerNormWeightTensorId) };
    finalNormNode.outTensors = { // shape: FA: [batchSize, seqLen, hiddenSize] PA: [seqLen, hiddenSize]
        &graph_.internalTensors.at(INTERNEL_TENSOR_FINAL_NORM_OUT)
    };

    auto &lmHeadNode = graph_.nodes.at(nodeId++);
    atb_speed::common::LmHeadParam lmHeadParam;
    lmHeadParam.unpadInputs = !param_.isFA;
    lmHeadParam.gatherAhead = param_.isPrefill;
    lmHeadParam.hiddenSizePerAttentionHead = param_.hiddenSizePerAttentionHead;
    lmHeadParam.linearParallelParam.fusionLinearParam.isBF16 = param_.isBF16;
    lmHeadParam.linearParallelParam.unpadInputs = !param_.isFA;
    if (param_.isLmHeadParallel) {
        lmHeadParam.linearParallelParam.parallelType = atb_speed::common::COLUMN_PARALLEL;
        lmHeadParam.linearParallelParam.tensorParallelInfo.rank = param_.rank;
        lmHeadParam.linearParallelParam.tensorParallelInfo.worldSize = param_.rankSize;
        lmHeadParam.linearParallelParam.tensorParallelInfo.backend = param_.backend;
    }
    LmHead(lmHeadParam, &op);
    lmHeadNode.operation.reset(op);
    const int finalLinearWeightTensorId = graph_.weightTensors.size() - WEIGHT_COUNT_LM_HEAD;
    lmHeadNode.inTensors = { &graph_.internalTensors.at(INTERNEL_TENSOR_FINAL_NORM_OUT),
        // shape: [vocabSizePerRank, hiddenSize]
        &graph_.weightTensors.at(finalLinearWeightTensorId),
        // LmHead未接入量化，量化权重使用placeholder代替
        &graph_.inTensors.at(IN_TENSOR_PLACE_HOLDER), &graph_.inTensors.at(IN_TENSOR_PLACE_HOLDER),
        &graph_.inTensors.at(IN_TENSOR_PLACE_HOLDER), &graph_.inTensors.at(IN_TENSOR_PLACE_HOLDER),
        &graph_.inTensors.at(IN_TENSOR_PLACE_HOLDER), &graph_.inTensors.at(IN_TENSOR_LOGTIS_INDICES) };
    // shpae: FA: [batchSize, seqLen, vocabSize] PA: [seqLen, vocabSize]
    lmHeadNode.outTensors = { &graph_.outTensors.at(0) };

    ATB_LOG(INFO) << "DecoderModel build graph success";
    return atb::NO_ERROR;
}

atb::Status PagedAttentionQuantModel::ParseParam(const std::string &param)
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

atb::Status PagedAttentionQuantModel::BindParamHostTensor(uint32_t nodeId)
{
    ATB_LOG(INFO) << "BindParamHostTensor";
    ATB_LOG(INFO) << "nodeId = " << nodeId;

    if (nodeId < OPERATION_COUNT_BEFORE_LAYER ||
        nodeId >= static_cast<uint32_t>(OPERATION_COUNT_BEFORE_LAYER + param_.numHiddenLayers)) {
        return atb::NO_ERROR;
    }

    auto &node = graph_.nodes.at(nodeId);
    const uint32_t tokenOffsetTensorId = LayerPATensorId::IN_TOKEN_OFFSET;
    const uint32_t seqLenTensorId = LayerPATensorId::IN_SEQ_LEN;
    node.variantPack.inTensors.at(tokenOffsetTensorId).hostData = tokenOffset_.data();
    node.variantPack.inTensors.at(seqLenTensorId).hostData = seqLen_.data();
    ATB_LOG(INFO) << "BindParamHostTensor end";
    return atb::NO_ERROR;
}
} // namespace baichuan2_7b
} // namespace atb_speed
