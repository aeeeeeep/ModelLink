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
#include "vector"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtype-limits"
#include <nlohmann/json.hpp>
#pragma GCC diagnostic pop
#include "atb/atb_infer.h"
#include "atb_speed/log.h"
#include "layers/operations/lmhead.h"
#include "layers/operations/word_embedding.h"
#include "layers/operations/positional_embedding.h"
#include "layers/operations/add_norm.h"
#include "models/qwen/layer/paged_attention_w8a8_layer.h"
#include "models/qwen/model/paged_attention_w8a8_model.h"

namespace atb_speed {
namespace qwen_14b {

// Weight count
const int WEIGHT_COUNT_PER_LAYER = 43;
const int WEIGHT_COUNT_WORD_EMBEDDINGNODE = 1;
const int WEIGHT_COUNT_POST_NORM = 1;
const int WEIGHT_COUNT_LM_HEAD = 1;

// Operation count
const int OPERATION_COUNT_BEFORE_LAYER = 2;  // wte(wordEmbed) + gather(cos/sin embedding)
const int OPERATION_COUNT_AFTER_LAYER = 2;  // RmsNorm + LmHead

enum InTensorId : int {
    IN_TENSOR_INPUTIDS = 0,
    IN_TENSOR_POSITIONIDS,
    IN_TENSOR_COSTABLE,
    IN_TENSOR_SINTABLE,
    IN_TENSOR_ATTENTIONMASK,
    IN_TENSOR_BLOCK_TABLES,
    IN_TENSOR_SLOTS,
    IN_TENSOR_INPUT_LENGTHS,
    IN_TENSOR_LOGTIS_INDICES,
    IN_PLACEHOLDER,
    IN_TENSOR_MAX,
};

enum InternalTensorId : int {
    INTERNAL_HIDDENSTATES = 0,
    INTERNAL_COSEMBED,
    INTERNAL_SINEMBED,
    INTERNAL_TENSOR_MAX,
};

enum OutTensorId : int {
    OUT_TENSOR_HIDDENSTATES = 0,
    OUT_TENSOR_MAX,
};

void PAW8A8Model::Param::FromString(const std::string &param)
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
    supportLcoc = paramJson["supportLcoc"].get<bool>();
    rank = paramJson["rank"].get<int>();
    worldSize = paramJson["worldSize"].get<int>();
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
    ATB_LOG(INFO) << "PAW8A8Model param" << ", isFA:" << isFA << ", isPrefill:" << isPrefill
                  << ", isBF16:" << isBF16
                  << ", isEmbeddingParallel: " << isEmbeddingParallel << ", isLmHeadParallel: "
                  << isLmHeadParallel << ", supportSwiGLU: " << supportSwiGLU
                  << ", rmsNormEps:" << rmsNormEps << ", numAttentionHeadsPerRank:"
                  << numAttentionHeadsPerRank << ", hiddenSizePerAttentionHead:" << hiddenSizePerAttentionHead
                  << ", numHiddenLayers:" << numHiddenLayers
                  << ", numKeyValueHeadsPerRank:" << numKeyValueHeadsPerRank
                  << ", supportLcoc:" << supportLcoc << ", rank:" << rank << ", worldSize:" << worldSize
                  << ", backend:" << backend << ", tokenOffset:" << tokenOffset << ", seqLen:" << seqLen;
}

PAW8A8Model::PAW8A8Model(const std::string &param) : Model("PAW8A8Model", param)
{
    param_.FromString(param);
    modelName_ += param_.isPrefill ? "_Prefill" : "_Decoder";
}

PAW8A8Model::~PAW8A8Model() {}

uint32_t PAW8A8Model::GetInputNum() const { return graph_.inTensors.size(); }

uint32_t PAW8A8Model::GetOutputNum() const { return graph_.outTensors.size(); }

atb::Status PAW8A8Model::InferShape(
    const std::vector<atb::TensorDesc> &inTensorDescs,
    std::vector<atb::TensorDesc> &outTensorDescs
)
{
    ATB_LOG(INFO) << "Enter PAW8A8Model InferShape";
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
        outTensorDescs.at(0).shape.dims[1] = param_.isPrefill ? inTensorDescs.at(graph_.inTensors.size() - 2).shape.dims[0] : 1;
    } else {  // unpadInputs = true
        if (param_.isPrefill) {
            outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(graph_.inTensors.size() - 2).shape.dims[0];
        }
    }

    if (param_.isLmHeadParallel) {
        outTensorDescs.at(0).shape.dims[outTensorDescs.at(0).shape.dimNum - 1] = vocabSizePerRank * param_.worldSize;
    } else {
        outTensorDescs.at(0).shape.dims[outTensorDescs.at(0).shape.dimNum - 1] = vocabSizePerRank;
    }

    return atb::NO_ERROR;
}

int64_t PAW8A8Model::BuildGraph()
{
    // set size
    const int weightTensorSize = WEIGHT_COUNT_WORD_EMBEDDINGNODE + WEIGHT_COUNT_PER_LAYER * param_.numHiddenLayers +
                                 WEIGHT_COUNT_POST_NORM + WEIGHT_COUNT_LM_HEAD;
    graph_.weightTensors.resize(weightTensorSize);

    graph_.kCacheTensors.resize(param_.numHiddenLayers);
    graph_.vCacheTensors.resize(param_.numHiddenLayers);

    graph_.inTensors.resize(IN_TENSOR_MAX);
    graph_.outTensors.resize(OUT_TENSOR_MAX);

    const int nodeSize = param_.numHiddenLayers + OPERATION_COUNT_BEFORE_LAYER + OPERATION_COUNT_AFTER_LAYER;
    graph_.nodes.resize(nodeSize);

    const int internalTensorSize = 1 + 2 + param_.numHiddenLayers * 2 + 2;
    graph_.internalTensors.resize(internalTensorSize);

    ATB_LOG(INFO) << "weightTensors.size=" << graph_.weightTensors.size()
                  << ", inTensors.size=" << graph_.inTensors.size()
                  << ", outTensors.size=" << graph_.outTensors.size()
                  << ", internalTensor.size=" << graph_.internalTensors.size()
                  << ", nodes.size=" << graph_.nodes.size();

    ATB_LOG(INFO) << "PAW8A8Model build graph begin";
    int nodeId = 0;

    atb::Operation *op = nullptr;

    // wte
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
        &graph_.inTensors.at(IN_TENSOR_INPUTIDS)
    };
    wordEmbeddingNode.outTensors = {&graph_.internalTensors.at(INTERNAL_HIDDENSTATES)};
    ATB_LOG(INFO) << "[+] wordEmbeddingNode";

    // gather
    auto &peGatherNode = graph_.nodes.at(nodeId++);
    atb_speed::common::PositionalEmbeddingGather(&op);
    peGatherNode.operation.reset(op);
    peGatherNode.inTensors = {
        &graph_.inTensors.at(IN_TENSOR_POSITIONIDS),
        &graph_.inTensors.at(IN_TENSOR_COSTABLE),
        &graph_.inTensors.at(IN_TENSOR_SINTABLE),
    };
    peGatherNode.outTensors = {
        &graph_.internalTensors.at(INTERNAL_COSEMBED),
        &graph_.internalTensors.at(INTERNAL_SINEMBED)
    };
    ATB_LOG(INFO) << "[+] peGatherNode";

    atb::Tensor *firstInTensor = &graph_.inTensors.at(IN_PLACEHOLDER);
    atb::Tensor *secondInTensor = &graph_.internalTensors.at(INTERNAL_HIDDENSTATES);
    // layers
    for (int layerId = 0; layerId < param_.numHiddenLayers; ++layerId) {
        auto &layerNode = graph_.nodes.at(nodeId++);
        atb_speed::qwen_14b::PAW8A8LayerParam layerParam;
        layerParam.isFA = param_.isFA;
        layerParam.isPrefill = param_.isPrefill;
        layerParam.isBF16 = param_.isBF16;
        layerParam.supportSwiGLU = param_.supportSwiGLU;
        layerParam.packQuantType = param_.packQuantType[layerId];
        layerParam.linearQuantType = param_.linearQuantType[layerId];
        layerParam.supportLcoc = param_.supportLcoc;
        layerParam.rmsNormEps = param_.rmsNormEps;
        layerParam.layerId = layerId;
        layerParam.numAttentionHeadsPerRank = param_.numAttentionHeadsPerRank;
        layerParam.hiddenSizePerAttentionHead = param_.hiddenSizePerAttentionHead;
        layerParam.numKeyValueHeadsPerRank = param_.numKeyValueHeadsPerRank;
        layerParam.rank = param_.rank;
        layerParam.worldSize = param_.worldSize;
        layerParam.backend = param_.backend;
        atb_speed::qwen_14b::PAW8A8Layer(layerParam, &op);
        layerNode.operation.reset(op);
        layerNode.inTensors.resize(layerNode.operation->GetInputNum());

        size_t inTensorId = 0;
        layerNode.inTensors.at(inTensorId++) = firstInTensor;
        layerNode.inTensors.at(inTensorId++) = secondInTensor;
        for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
            layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(
                layerId * WEIGHT_COUNT_PER_LAYER + weightTensorId + WEIGHT_COUNT_WORD_EMBEDDINGNODE);
        }
        layerNode.inTensors.at(inTensorId++) = &graph_.internalTensors.at(INTERNAL_COSEMBED);
        layerNode.inTensors.at(inTensorId++) = &graph_.internalTensors.at(INTERNAL_SINEMBED);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ATTENTIONMASK);
        layerNode.inTensors.at(inTensorId++) = &graph_.kCacheTensors.at(layerId);
        layerNode.inTensors.at(inTensorId++) = &graph_.vCacheTensors.at(layerId);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_BLOCK_TABLES);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SLOTS);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_INPUT_LENGTHS);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_PLACEHOLDER);

        layerNode.outTensors = {
            &graph_.internalTensors.at(INTERNAL_TENSOR_MAX + layerId),
            &graph_.internalTensors.at(INTERNAL_TENSOR_MAX + param_.numHiddenLayers + layerId)
        };
        ATB_LOG(INFO) << "[+] layerNode_" << layerId;
        firstInTensor = layerNode.outTensors.at(0);
        secondInTensor = layerNode.outTensors.at(1);
    }

    auto &addNormNode = graph_.nodes.at(nodeId++);
    atb::infer::RmsNormParam finalNormParam;
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
        &graph_.inTensors.at(IN_PLACEHOLDER),
        &graph_.inTensors.at(IN_PLACEHOLDER),
        &graph_.inTensors.at(IN_PLACEHOLDER),
        &graph_.inTensors.at(IN_PLACEHOLDER),
        &graph_.inTensors.at(IN_PLACEHOLDER),
    };
    addNormNode.outTensors = {
        &graph_.internalTensors.at(internalTensorSize - 2),
        &graph_.internalTensors.at(internalTensorSize - 1)
    };
    ATB_LOG(INFO) << "[+] finalNormNode";

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
        &graph_.internalTensors.at(internalTensorSize - 2),
        // shape: [vocabSizePerRank, hiddenSize]
        &graph_.weightTensors.at(finalLinearWeightTensorId),
        // LmHead未接入量化，量化权重使用placeholder代替
        &graph_.inTensors.at(IN_PLACEHOLDER),
        &graph_.inTensors.at(IN_PLACEHOLDER),
        &graph_.inTensors.at(IN_PLACEHOLDER),
        &graph_.inTensors.at(IN_PLACEHOLDER),
        &graph_.inTensors.at(IN_PLACEHOLDER),
        &graph_.inTensors.at(IN_TENSOR_LOGTIS_INDICES)
    };
    // shpae: FA: [batchSize, seqLen, vocabSize] PA: [seqLen, vocabSize]
    lmHeadNode.outTensors = {&graph_.outTensors.at(0)};
    ATB_LOG(INFO) << "[+] lmHeadNode";

    ATB_LOG(INFO) << "PAW8A8Model build graph success";
    return atb::NO_ERROR;
}

atb::Status PAW8A8Model::ParseParam(const std::string &param)
{
    ATB_LOG(INFO) << "ParseParam start.";
    nlohmann::json paramJson = nlohmann::json::parse(param);

    seqLen_.clear();
    for (auto item : paramJson["seqLen"]) {
        seqLen_.push_back(item.get<int>());
        ATB_LOG(INFO) << "Prefill" << paramJson["isPrefill"] << "seqLen value: " << item;
    }
    ATB_LOG(INFO) << "ParseParam end.";
    return atb::NO_ERROR;
}

atb::Status PAW8A8Model::BindParamHostTensor(uint32_t nodeId)
{
    ATB_LOG(INFO) << "BindParamHostTensor";
    ATB_LOG(INFO) << "nodeId = " << nodeId;

    if (nodeId < OPERATION_COUNT_BEFORE_LAYER || nodeId >= static_cast<uint32_t>(OPERATION_COUNT_BEFORE_LAYER + param_.numHiddenLayers)) {
        return atb::NO_ERROR;
    }

    auto &node = graph_.nodes.at(nodeId);
    const uint32_t seqLenTensorId = 52;
    node.variantPack.inTensors.at(seqLenTensorId).hostData = seqLen_.data();
    ATB_LOG(INFO) << "BindParamHostTensor end";
    return atb::NO_ERROR;
}
} // namespace qwen_14b
} // namespace atb_speed