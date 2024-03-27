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
#include <vector>
#include <nlohmann/json.hpp>
#include "atb/atb_infer.h"
#include "atb_speed/log.h"
#include "layers/operations/word_embedding.h"
#include "layers/operations/positional_embedding.h"
#include "layers/operations/add_norm.h"
#include "layers/operations/lmhead.h"
#include "models/glm/v2_6b/layer/decoder_layer.h"
#include "models/glm/v2_6b/model/decoder_model.h"

namespace atb_speed {
namespace glm_v2_6b {

// Weight count
const int WEIGHT_COUNT_PER_LAYER = 50;
const int WEIGHT_COUNT_WORD_EMBEDDINGNODE = 1;
const int WEIGHT_COUNT_POST_NORM = 1;
const int WEIGHT_COUNT_LM_HEAD = 1;

// Operation count
const int OPERATION_COUNT_BEFORE_LAYER = 2;
const int OPERATION_COUNT_AFTER_LAYER = 2;  // RmsNorm + LmHead

void DecoderModel::Param::FromString(const std::string &param)
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
    worldSize = paramJson["worldSize"].get<int>();
    backend = paramJson["backend"].get<std::string>();
    for (auto item : paramJson["tokenOffset"]) {
        tokenOffset.push_back(item.get<int>());
    }
    for (auto item : paramJson["seqLen"]) {
        seqLen.push_back(item.get<int>());
    }
    for (auto item : paramJson["preScale"]) {
        preScale.push_back(item.get<float>());
    }
    for (auto item : paramJson["postScale"]) {
        postScale.push_back(item.get<float>());
    }
    for (auto item : paramJson["packQuantType"]) {
        packQuantType.push_back(item.get<std::vector<int>>());
    }
    for (auto item : paramJson["linearQuantType"]) {
        linearQuantType.push_back(item.get<std::vector<int>>());
    }
    ATB_LOG(INFO) << "DecoderModel param" << ", isFA:" << isFA << ", isPrefill:" << isPrefill
                  << ", isBF16:" << isBF16
                  << ", isEmbeddingParallel: " << isEmbeddingParallel << ", isLmHeadParallel: "
                  << isLmHeadParallel << ", supportSwiGLU: " << supportSwiGLU
                  << ", rmsNormEps:" << rmsNormEps << ", numAttentionHeadsPerRank:"
                  << numAttentionHeadsPerRank << ", hiddenSizePerAttentionHead:" << hiddenSizePerAttentionHead
                  << ", numHiddenLayers:" << numHiddenLayers
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

atb::Status DecoderModel::InferShape(
    const std::vector<atb::TensorDesc> &inTensorDescs,
    std::vector<atb::TensorDesc> &outTensorDescs
)
{
    if (outTensorDescs.size() != GetOutputNum()) {
        return atb::ERROR_INVALID_GRAPH;
    }

    const int64_t vocabSizePerRank = graph_.weightTensors.at(graph_.weightTensors.size() - 1).desc.shape.dims[0];
    // FA: [batchSize, seqLen, vocabSize] PA: [seqLen, vocabSisze]
    outTensorDescs.at(0).dtype = graph_.weightTensors.at(graph_.weightTensors.size() - 1).desc.dtype;
    outTensorDescs.at(0).format = graph_.weightTensors.at(0).desc.format;
    outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum + 1;
    outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
    if (param_.isFA) {  // unpadInputs = false
        outTensorDescs.at(0).shape.dims[1] = param_.isPrefill ? inTensorDescs.at(graph_.inTensors.size() - 1).shape.dims[0] : 1;
    } else {  // unpadInputs = true
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
    int INTERNEL_TENSOR_RESIDUAL_ADD_OUT = internelTensorIdx++;

    // set size
    const int weightTensorSize =
        WEIGHT_COUNT_WORD_EMBEDDINGNODE + WEIGHT_COUNT_PER_LAYER * param_.numHiddenLayers
        + WEIGHT_COUNT_POST_NORM + WEIGHT_COUNT_LM_HEAD;

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
        wordEmbeddingParam.tensorParallelInfo = {param_.rank, param_.worldSize, param_.backend};
    };
    atb_speed::common::WordEmbedding(wordEmbeddingParam, &op);
    wordEmbeddingNode.operation.reset(op);
    wordEmbeddingNode.inTensors = {
        &graph_.weightTensors.at(0),                    // shape: [vocabSize + 1, hiddenSize]
        &graph_.inTensors.at(IN_TENSOR_INPUT_IDS)
    };

    wordEmbeddingNode.outTensors = {&graph_.internalTensors.at(INTERNEL_TENSOR_HIDDEN_STATES)};

    auto &peGatherNode = graph_.nodes.at(nodeId++);
    atb_speed::common::PositionalEmbeddingGather(&op);
    peGatherNode.operation.reset(op);
    peGatherNode.inTensors = {
        &graph_.inTensors.at(IN_TENSOR_POSITION_IDS),
        &graph_.inTensors.at(IN_TENSOR_COS_TABLE),
        &graph_.inTensors.at(IN_TENSOR_SIN_TABLE),
    };
    peGatherNode.outTensors = {
        &graph_.internalTensors.at(INTERNEL_TENSOR_COS_EMB),
        &graph_.internalTensors.at(INTERNEL_TENSOR_SIN_EMB)
    };
    
    atb::Tensor *firstInTensor = &graph_.inTensors.at(INTERNEL_TENSOR_HIDDEN_STATES);
    atb::Tensor *secondInTensor = &graph_.internalTensors.at(INTERNEL_TENSOR_HIDDEN_STATES);

    for (int layerId = 0; layerId < param_.numHiddenLayers; ++layerId) {
        auto &layerNode = graph_.nodes.at(nodeId++);
        atb_speed::glm_v2_6b::DecoderLayerParam layerParam;
        layerParam.isFA = param_.isFA;
        layerParam.isPrefill = param_.isPrefill;
        layerParam.isBF16 = param_.isBF16;
        layerParam.supportSwiGLU = param_.supportSwiGLU;
        layerParam.packQuantType = param_.packQuantType[layerId];
        layerParam.linearQuantType = param_.linearQuantType[layerId];

        layerParam.rmsNormEps = param_.rmsNormEps;
        layerParam.layerId = layerId;
        layerParam.numAttentionHeadsPerRank = param_.numAttentionHeadsPerRank;
        layerParam.hiddenSizePerAttentionHead = param_.hiddenSizePerAttentionHead;
        layerParam.numKeyValueHeadsPerRank = param_.numKeyValueHeadsPerRank;
        layerParam.rank = param_.rank;
        layerParam.worldSize = param_.worldSize;
        layerParam.backend = param_.backend;
        if (param_.isFA) {
            layerParam.preScale = param_.preScale.at(layerId);
            layerParam.postScale = param_.postScale.at(layerId);
        }

        atb_speed::glm_v2_6b::DecoderLayer(layerParam, &op);

        layerNode.operation.reset(op);
        layerNode.inTensors.resize(layerNode.operation->GetInputNum());

        size_t inTensorId = 0;
        layerNode.inTensors.at(inTensorId++) = firstInTensor;
        layerNode.inTensors.at(inTensorId++) = secondInTensor;
        for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
            layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(
                layerId * WEIGHT_COUNT_PER_LAYER + weightTensorId + WEIGHT_COUNT_WORD_EMBEDDINGNODE);
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
        layerNode.outTensors = {
            &graph_.internalTensors.at(INTERNEL_TENSOR_RESIDUAL_ADD_OUT),
            &graph_.internalTensors.at(INTERNEL_TENSOR_HIDDEN_STATES),
        };

        firstInTensor = layerNode.outTensors.at(0);
        secondInTensor = layerNode.outTensors.at(1);
    }

    auto &addNormNode = graph_.nodes.at(nodeId++);
    atb::infer::RmsNormParam finalNormParam;
    // finalNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    // finalNormParam.normParam.epsilon = param_.rmsNormEps;
    // CREATE_OPERATION(finalNormParam, &op);
    // finalNormNode.operation.reset(op);
    finalNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_PRENORM;
    finalNormParam.preNormParam.epsilon = param_.rmsNormEps;

    atb_speed::common::AddNormParam<atb::infer::RmsNormParam> addNormParam;
    addNormParam.addNormType = atb_speed::common::AddNormType::FUSION_ADD_NORM;
    addNormParam.normParamType = finalNormParam;
    AddNorm(addNormParam, &op);
    addNormNode.operation.reset(op);
    const int finalLayerNormWeightTensorId =
        graph_.weightTensors.size() - WEIGHT_COUNT_POST_NORM - WEIGHT_COUNT_LM_HEAD;
    addNormNode.inTensors = {
        firstInTensor, secondInTensor,
        &graph_.weightTensors.at(finalLayerNormWeightTensorId),
        &graph_.inTensors.at(IN_TENSOR_PLACE_HOLDER),
        &graph_.inTensors.at(IN_TENSOR_PLACE_HOLDER),
        &graph_.inTensors.at(IN_TENSOR_PLACE_HOLDER),
        &graph_.inTensors.at(IN_TENSOR_PLACE_HOLDER),
        &graph_.inTensors.at(IN_TENSOR_PLACE_HOLDER),
    };
    addNormNode.outTensors = {
        // shape: FA: [batchSize, seqLen, hiddenSize] PA: [seqLen, hiddenSize]
        &graph_.internalTensors.at(INTERNEL_TENSOR_RESIDUAL_ADD_OUT),
        &graph_.internalTensors.at(INTERNEL_TENSOR_HIDDEN_STATES)
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
        lmHeadParam.linearParallelParam.tensorParallelInfo.worldSize = param_.worldSize;
        lmHeadParam.linearParallelParam.tensorParallelInfo.backend = param_.backend;
    }
    LmHead(lmHeadParam, &op);
    lmHeadNode.operation.reset(op);
    const int finalLinearWeightTensorId = graph_.weightTensors.size() - WEIGHT_COUNT_LM_HEAD;
    lmHeadNode.inTensors = {
        &graph_.internalTensors.at(INTERNEL_TENSOR_RESIDUAL_ADD_OUT),
        // shape: [vocabSizePerRank, hiddenSize]
        &graph_.weightTensors.at(finalLinearWeightTensorId),
        // LmHead未接入量化，量化权重使用placeholder代替
        &graph_.inTensors.at(IN_TENSOR_PLACE_HOLDER),
        &graph_.inTensors.at(IN_TENSOR_PLACE_HOLDER),
        &graph_.inTensors.at(IN_TENSOR_PLACE_HOLDER),
        &graph_.inTensors.at(IN_TENSOR_PLACE_HOLDER),
        &graph_.inTensors.at(IN_TENSOR_PLACE_HOLDER),
        &graph_.inTensors.at(IN_TENSOR_LOGTIS_INDICES)
    };
    // shpae: FA: [batchSize, seqLen, vocabSize] PA: [seqLen, vocabSize]
    lmHeadNode.outTensors = {&graph_.outTensors.at(0)};

    ATB_LOG(INFO) << "DecoderModel build graph success";
    ATB_LOG(FATAL) << "DecoderModel build graph success";

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

    if (nodeId < OPERATION_COUNT_BEFORE_LAYER || nodeId >= static_cast<uint32_t>(OPERATION_COUNT_BEFORE_LAYER + param_.numHiddenLayers)) {
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
} // namespace glm_v2_6b
} // namespace atb_speed