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
#include "atb/atb_infer.h"
#include "atb_speed/log.h"
#include "nlohmann/json.hpp"

#include "models/starcoder/layer/paged_attention_quant_layer.h"
#include "paged_attention_quant_model.h"
#include "atb_speed/utils/model_factory.h"
#include "layers/operations/word_embedding.h"
#include "layers/operations/lmhead.h"

namespace atb_speed {
namespace star_coder {

REGISTER_MODEL(star_coder, PAQuantModel);

const int WEIGHT_COUNT_PER_LAYER = 50;
const int WEIGHT_COUNT_WORD_EMBEDDINGNODE = 2;
const int WEIGHT_COUNT_POST_NORM = 2;
const int WEIGHT_COUNT_LM_HEAD = 1;

const int BEFORE_LAYER_WEIGHT_COUNT = 2;
const int OPERATION_COUNT_BEFORE_LAYER = 3;
const int OPERATION_COUNT_AFTER_LAYER = 2;
const int FINALNORMNODE_WEIGHT_COUNT = 2;
const int OUT_LM_HEAD_WEIGHT_COUNT = 1;
const int LAYER_NORM_AXIS_COUNT = 1;

enum InTensorId : int {
    IN_TENSOR_INPUT_IDS = 0,
    IN_TENSOR_POSITION_IDS,
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

void PAQuantModel::Param::FromString(const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);
    ATB_LOG(INFO) << "start parse";
    isFA = paramJson["isFA"].get<bool>();
    isBF16 = paramJson["isBF16"].get<bool>();
    isEmbeddingParallel = paramJson["isEmbeddingParallel"].get<bool>();
    isLmHeadParallel = paramJson["isLmHeadParallel"].get<bool>();
    headNum = paramJson["headNum"].get<int>();
    supportSwiGLU = paramJson["supportSwiGLU"].get<bool>();
    rank = paramJson["rank"].get<int>();
    worldSize = paramJson["worldSize"].get<int>();
    backend = paramJson["backend"].get<std::string>();
    layerNormEps = paramJson["layerNormEps"].get<double>();
    numAttentionHeadsPerRank = paramJson["numAttentionHeadsPerRank"].get<int>();
    hiddenSizePerAttentionHead = paramJson["hiddenSizePerAttentionHead"].get<int>();
    numHiddenLayers = paramJson["numHiddenLayers"].get<int>();
    numKeyValueHeadsPerRank = paramJson["numKeyValueHeadsPerRank"].get<int>();
    for (auto item : paramJson["packQuantType"]) {
        packQuantType.push_back(item.get<std::vector<int>>());
    }
    for (auto item : paramJson["linearQuantType"]) {
        linearQuantType.push_back(item.get<std::vector<int>>());
    }
    isPrefill = paramJson["isPrefill"].get<bool>();
    ATB_LOG(INFO) << "DecoderModel param" << ", isFA:" << isFA << ", isPrefill:" << isPrefill
                  << ", isBF16:" << isBF16
                  << ", isEmbeddingParallel: " << isEmbeddingParallel << ", isLmHeadParallel: "
                  << isLmHeadParallel << ", supportSwiGLU: " << supportSwiGLU
                  << ", layerNormEps:" << layerNormEps << ", numAttentionHeadsPerRank:"
                  << numAttentionHeadsPerRank << ", hiddenSizePerAttentionHead:" << hiddenSizePerAttentionHead
                  << ", numHiddenLayers:" << numHiddenLayers
                  << ", numKeyValueHeadsPerRank:" << numKeyValueHeadsPerRank
                  << ", rank:" << rank << ", worldSize:" << worldSize << ", backend:" << backend;
}

PAQuantModel::PAQuantModel(const std::string &param) : Model("PAQuantModel", param)
{
    ATB_LOG(INFO) << "start loading json config from python";
    param_.FromString(param);
    modelName_ += param_.isPrefill ? "_Prefill" : "_Decoder";
    ATB_LOG(INFO) << "check from string success";
}

PAQuantModel::~PAQuantModel() = default;

uint32_t PAQuantModel::GetInputNum() const
{
    return graph_.inTensors.size();
}

uint32_t PAQuantModel::GetOutputNum() const
{
    return graph_.outTensors.size();
}

atb::Status PAQuantModel::InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
    std::vector<atb::TensorDesc> &outTensorDescs)
{
    ATB_LOG(INFO) << "Enter PAModel InferShape";
    if (outTensorDescs.size() != GetOutputNum()) {
        return atb::ERROR_INVALID_GRAPH;
    }
    const int64_t vocabSizePerRank = graph_.weightTensors.at(graph_.weightTensors.size() - 1).desc.shape.dims[0];
    outTensorDescs.at(0).dtype = graph_.weightTensors.at(graph_.weightTensors.size() - 1).desc.dtype;
    outTensorDescs.at(0).format = graph_.weightTensors.at(0).desc.format;
    outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum + 1;
    outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
    if (param_.isFA) {  // unpadInputs = false
        outTensorDescs.at(0).shape.dims[1] = param_.isPrefill ? inTensorDescs.at(6).shape.dims[0] : 1;
    } else {  // unpadInputs = true
        if (param_.isPrefill) {
            outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(6).shape.dims[0];
        }
    }
    if (param_.isLmHeadParallel) {
        outTensorDescs.at(0).shape.dims[outTensorDescs.at(0).shape.dimNum - 1] = vocabSizePerRank * param_.worldSize;
    } else {
        outTensorDescs.at(0).shape.dims[outTensorDescs.at(0).shape.dimNum - 1] = vocabSizePerRank;
    }
    ATB_LOG(INFO) << "PAQuantModel InferShape Success";
    return atb::NO_ERROR;
}


int64_t PAQuantModel::BuildGraph()
{
    ATB_LOG(INFO) << "Enter PAQuantModel BuildGraph";
    // define internelTensor
    int internelTensorIdx = 0;
    // idx: 0, wte
    int INTERNEL_TENSOR_HIDDEN_STATES = internelTensorIdx++;
    // idx: 1, wpe
    int INTERNEL_TENSOR_POSITION_EMB = internelTensorIdx++;
    // idx  2, add
    int INTERNEL_TENSOR_ADD = internelTensorIdx++;
    // layer start
    int INTERNEL_LAYER_START_BASE = internelTensorIdx++;
    internelTensorIdx = internelTensorIdx + param_.numHiddenLayers - 1;
    // idx: 3 + numHiddenLayers, shape: FA: [batchSize, seqLen, hiddenSize] PA: [seqLen, hiddenSize]
    int INTERNEL_TENSOR_FINAL_NORM_OUT = internelTensorIdx++;

    const int weightTensorSize = BEFORE_LAYER_WEIGHT_COUNT +
                                 WEIGHT_COUNT_PER_LAYER * param_.numHiddenLayers +
                                 WEIGHT_COUNT_POST_NORM + WEIGHT_COUNT_LM_HEAD;

    graph_.weightTensors.resize(weightTensorSize);
    graph_.inTensors.resize(IN_TENSOR_MAX);
    graph_.outTensors.resize(1);
    
    graph_.kCacheTensors.resize(param_.numHiddenLayers);
    graph_.vCacheTensors.resize(param_.numHiddenLayers);

    const int nodeSize = param_.numHiddenLayers + OPERATION_COUNT_BEFORE_LAYER + OPERATION_COUNT_AFTER_LAYER;
    graph_.nodes.resize(nodeSize);
    graph_.internalTensors.resize(internelTensorIdx);

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
        &graph_.weightTensors.at(INTERNEL_TENSOR_HIDDEN_STATES),
        &graph_.inTensors.at(IN_TENSOR_INPUT_IDS)
    };
    wordEmbeddingNode.outTensors = {&graph_.internalTensors.at(INTERNEL_TENSOR_HIDDEN_STATES)};

    auto &posEmbeddingNode = graph_.nodes.at(nodeId++);
    atb_speed::common::WordEmbeddingParam posEmbeddingParam;
    posEmbeddingParam.unpadInputs = !param_.isFA;
    if (param_.isEmbeddingParallel) {
        posEmbeddingParam.tensorParallelInfo = {param_.rank, param_.worldSize, param_.backend};
    };
    atb_speed::common::WordEmbedding(posEmbeddingParam, &op);
    posEmbeddingNode.operation.reset(op);
    posEmbeddingNode.inTensors = {
        &graph_.weightTensors.at(INTERNEL_TENSOR_POSITION_EMB),
        &graph_.inTensors.at(IN_TENSOR_POSITION_IDS)
    };
    posEmbeddingNode.outTensors = {&graph_.internalTensors.at(INTERNEL_TENSOR_POSITION_EMB)};

    auto &addNode = graph_.nodes.at(nodeId++);
    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CREATE_OPERATION(addParam, &op);
    addNode.operation.reset(op);
    addNode.inTensors = {&graph_.internalTensors.at(INTERNEL_TENSOR_HIDDEN_STATES), &graph_.internalTensors.at(INTERNEL_TENSOR_POSITION_EMB)};
    addNode.outTensors = {&graph_.internalTensors.at(INTERNEL_TENSOR_ADD)};

    atb::Tensor *firstInTensor = &graph_.internalTensors.at(INTERNEL_TENSOR_ADD);
    for (int layerId = 0; layerId < param_.numHiddenLayers; ++layerId) {
        auto &layerNode = graph_.nodes.at(nodeId++);
        atb_speed::star_coder::PAQuantLayerParam layerParam;
        layerParam.isFA = param_.isFA;
        layerParam.isPrefill = param_.isPrefill;
        layerParam.isBF16 = param_.isBF16;
        layerParam.supportSwiGLU = param_.supportSwiGLU;
        layerParam.packQuantType = param_.packQuantType[layerId];
        layerParam.linearQuantType = param_.linearQuantType[layerId];
        layerParam.layerNormEps = param_.layerNormEps;
        layerParam.numAttentionHeadsPerRank = param_.numAttentionHeadsPerRank;
        layerParam.hiddenSizePerAttentionHead = param_.hiddenSizePerAttentionHead;
        layerParam.numKeyValueHeadsPerRank = param_.numKeyValueHeadsPerRank;
        layerParam.rank = param_.rank;
        layerParam.worldSize = param_.worldSize;
        layerParam.backend = param_.backend;
        atb_speed::star_coder::PAQuantLayer(layerParam, &op);
        layerNode.operation.reset(op);
        layerNode.inTensors.resize(layerNode.operation->GetInputNum());
        size_t inTensorId = 0;
        layerNode.inTensors.at(inTensorId++) = firstInTensor;
        ATB_LOG(INFO) << "check offset---------------------------------------";
        for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
            layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(
                layerId * WEIGHT_COUNT_PER_LAYER + weightTensorId + WEIGHT_COUNT_WORD_EMBEDDINGNODE);
        }
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ATTENTIONMASK);   // atten_mask
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_BLOCK_TABLES);    // block
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SLOTS);           // slots
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_INPUT_LENGTHS);   // seqLen
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_HOLDER);                 // holder
        layerNode.inTensors.at(inTensorId++) = &graph_.kCacheTensors.at(layerId);
        layerNode.inTensors.at(inTensorId++) = &graph_.vCacheTensors.at(layerId);
        layerNode.outTensors = {&graph_.internalTensors.at(INTERNEL_LAYER_START_BASE + layerId)};
        firstInTensor = layerNode.outTensors.at(0);
    }

    auto &finalNormNode = graph_.nodes.at(nodeId++);
    atb::infer::LayerNormParam finalNormParam;
    finalNormParam.layerType = atb::infer::LayerNormParam::LayerNormType::LAYER_NORM_NORM;
    finalNormParam.normParam.epsilon = param_.layerNormEps;
    finalNormParam.normParam.beginNormAxis = LAYER_NORM_AXIS_COUNT;
    finalNormParam.normParam.beginParamsAxis = LAYER_NORM_AXIS_COUNT;
    CREATE_OPERATION(finalNormParam, &op);
    finalNormNode.operation.reset(op);

    const int finalLayerNormWeightTensorId =
        graph_.weightTensors.size() - FINALNORMNODE_WEIGHT_COUNT - OUT_LM_HEAD_WEIGHT_COUNT;
    const int finalLayerNormBiasTensorId =
        graph_.weightTensors.size() - (FINALNORMNODE_WEIGHT_COUNT - 1) - OUT_LM_HEAD_WEIGHT_COUNT;
    finalNormNode.inTensors = {firstInTensor, &graph_.weightTensors.at(finalLayerNormWeightTensorId),
                               &graph_.weightTensors.at(finalLayerNormBiasTensorId)};
    finalNormNode.outTensors = {&graph_.internalTensors.at(INTERNEL_TENSOR_FINAL_NORM_OUT)};

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
        &graph_.internalTensors.at(INTERNEL_TENSOR_FINAL_NORM_OUT),
        // shape: [vocabSizePerRank, hiddenSize]
        &graph_.weightTensors.at(finalLinearWeightTensorId),
        // LmHead未接入量化，量化权重使用placeholder代替
        &graph_.inTensors.at(IN_HOLDER),
        &graph_.inTensors.at(IN_HOLDER),
        &graph_.inTensors.at(IN_HOLDER),
        &graph_.inTensors.at(IN_HOLDER),
        &graph_.inTensors.at(IN_HOLDER),
        &graph_.inTensors.at(IN_TENSOR_LOGTIS_INDICES)
    };
    lmHeadNode.outTensors = {&graph_.outTensors.at(0)};

    return atb::NO_ERROR;
}

atb::Status PAQuantModel::ParseParam(const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);

    seqLen_.clear();
    for (auto item : paramJson["seqLen"]) {
        seqLen_.push_back(item.get<int>());
    }
    
    ATB_LOG(INFO) << "PAModel ParseParam seqLen: " << seqLen_.capacity();
    return atb::NO_ERROR;
}

atb::Status PAQuantModel::BindParamHostTensor(uint32_t nodeId)
{
    ATB_LOG(INFO) << "BindParamHostTensor";
    if (nodeId < OPERATION_COUNT_BEFORE_LAYER || nodeId >= static_cast<uint32_t>(OPERATION_COUNT_BEFORE_LAYER + param_.numHiddenLayers)) {
        return atb::NO_ERROR;
    }

    auto &node = graph_.nodes.at(nodeId);
    const uint32_t seqLenTensorId = PAQuantLayerTensorId::IN_SEQ_LEN;
    node.variantPack.inTensors.at(seqLenTensorId).hostData = seqLen_.data();

    ATB_LOG(INFO) << "BindParamHostTensor end";
    return atb::NO_ERROR;
}
} // namespace star_coder
} // namespace atb_speed