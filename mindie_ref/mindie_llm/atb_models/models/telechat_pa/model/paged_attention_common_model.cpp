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


#include <nlohmann/json.hpp>
#include <atb/atb_infer.h>
#include "models/telechat_pa/layer/paged_attention_common_layer.h"
#include "models/telechat/layer/quant_layer.h"
#include "telechat/layer/embedding_layer.h"
#include "paged_attention_common_model.h"

namespace atb_speed {
namespace telechat {

const int QUANT_WEIGHT_COUNT_PER_LAYER = 22;
const int FLOAT_WEIGHT_COUNT_PER_LAYER = 10;
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
    IN_TENSOR_BLOCK_TABLES,
    IN_TENSOR_SLOTS,
    IN_TENSOR_INPUT_LENGTHS,
    IN_TENSOR_PAST_KEY,
    IN_TENSOR_PAST_VALUE,
    IN_TOKENOFFSET,
    IN_SEQLEN,
    IN_TENSOR_LOGTIS_INDICES,
    IN_HOLDER,
    IN_TENSOR_MAX,
};

int64_t CommonPAModel::Param::FromString(const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);
    rmsNormEps = paramJson["rmsNormEps"].get<float>();
    headNum = paramJson["headNum"].get<int>();
    dk = paramJson["dk"].get<int>();
    layerNum = paramJson["layerNum"].get<int>();
    rank = paramJson["rank"].get<int>();
    rankSize = paramJson["rankSize"].get<int>();
    if (paramJson.contains("transposedWeight")) {
        transposedWeight = paramJson["transposedWeight"].get<bool>();
    }
    if (paramJson.contains("isPrefill")) {
        isPrefill = paramJson["isPrefill"].get<bool>();
    }
    if (paramJson.contains("isBF16")) {
        isBF16 = paramJson["isBF16"].get<bool>();
    }
    if (paramJson.contains("isQuant")) {
        isQuant = paramJson["isQuant"].get<bool>();
        if (isQuant) {
            for (auto item : paramJson["float_query_layers"]) {
                float_query_layers.push_back(item.get<int>());
            }
            for (auto item : paramJson["float_kv_layers"]) {
                float_kv_layers.push_back(item.get<int>());
            }
            for (auto item : paramJson["float_down_layers"]) {
                float_down_layers.push_back(item.get<int>());
            }
            for (auto item : paramJson["inputScale_qkv"]) {
                inputScale_qkv.push_back(item.get<float>());
            }
            for (auto item : paramJson["inputOffset_qkv"]) {
                inputOffset_qkv.push_back(item.get<int>());
            }
            for (auto item : paramJson["inputScale_dense"]) {
                inputScale_dense.push_back(item.get<float>());
            }
            for (auto item : paramJson["inputOffset_dense"]) {
                inputOffset_dense.push_back(item.get<int>());
            }
            for (auto item : paramJson["inputScale_gate_up"]) {
                inputScale_gate_up.push_back(item.get<float>());
            }
            for (auto item : paramJson["inputOffset_gate_up"]) {
                inputOffset_gate_up.push_back(item.get<int>());
            }
            for (auto item : paramJson["inputScale_down_proj"]) {
                inputScale_down_proj.push_back(item.get<float>());
            }
            for (auto item : paramJson["inputOffset_down_proj"]) {
                inputOffset_down_proj.push_back(item.get<int>());
            }
        }
    };
    if (headNum == 0) {
        ATB_LOG(ERROR) << "param.headNum is 0, please input a correct value";
        return atb::ERROR_INVALID_PARAM;
    };
    if (dk == 0) {
        ATB_LOG(ERROR) << "param.dk is 0, please input a correct value";
        return atb::ERROR_INVALID_PARAM;
    };
    ATB_LOG(INFO) << "Llama CommonPAModel param rmsNormEps:" << rmsNormEps << ", headNum:" << headNum << ", dk:" << dk
                  << ", layerNum:" << layerNum << ", transposedWeight:" << transposedWeight << ", rank:" << rank
                  << ", rankSize:" << rankSize << ", isBF16:" << isBF16 << ", is Quant:" << isQuant;
    
    return atb::NO_ERROR;
}

CommonPAModel::CommonPAModel(const std::string &param) : Model("TelechatCommonPAModel", param)
{
    param_.FromString(param);
    modelName_ += param_.isPrefill ? "_Prefill" : "_Decoder";
}

CommonPAModel::~CommonPAModel() {}
uint32_t CommonPAModel::GetInputNum() const
{return graph_.inTensors.size(); }

uint32_t CommonPAModel::GetOutputNum() const
{return graph_.outTensors.size(); }

atb::Status CommonPAModel::InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
                                      std::vector<atb::TensorDesc> &outTensorDescs)
{
    if (outTensorDescs.size() != GetOutputNum()) {
        return atb::ERROR_INVALID_GRAPH;
    }

    outTensorDescs.at(0) = graph_.weightTensors.at(0).desc;
    outTensorDescs.at(0).shape.dimNum = 3;
    outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(IN_TENSOR_INPUT_IDS).shape.dims[0];
    outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(IN_TENSOR_INPUT_IDS).shape.dims[1];
    outTensorDescs.at(0).shape.dims[2] = graph_.weightTensors.at(graph_.weightTensors.size() - 1).desc.shape.dims[1];

    if (param_.isPrefill) {
        outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(IN_TENSOR_LOGTIS_INDICES).shape.dims[0];
    }

    return atb::NO_ERROR;
}

int64_t CommonPAModel::BuildGraph()
{
    ATB_LOG(INFO) << "Enter Telechat FloatFAModel BuildGraph";
    const int QuantweightTensorSize = WORD_EMBEDDING_WEIGHT_COUNT + QUANT_WEIGHT_COUNT_PER_LAYER * param_.layerNum +
                                     FINAL_RMSNORM_WEIGHT_COUNT + OUT_LM_HEAD_WEIGHT_COUNT;

    const int FloatweightTensorSize = WORD_EMBEDDING_WEIGHT_COUNT + FLOAT_WEIGHT_COUNT_PER_LAYER * param_.layerNum +
                                     FINAL_RMSNORM_WEIGHT_COUNT + OUT_LM_HEAD_WEIGHT_COUNT;

    if (param_.isQuant)
    {
        graph_.weightTensors.resize(QuantweightTensorSize);
        ATB_LOG(INFO) << "model type is Quant, Weight tensor size is: " << QuantweightTensorSize;
    }
    else {
        graph_.weightTensors.resize(FloatweightTensorSize);
        ATB_LOG(INFO) << "model type is Quant, Weight tensor size is: " << FloatweightTensorSize;
    }

    const int inTensorSize = IN_TENSOR_MAX + param_.layerNum;
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
        PaCommonLayerParam layerParam;
        if (!param_.isQuant)
        {
            // 浮点
            ATB_LOG(INFO) << "Enter Float Layer, layerId: " << layerId;
            layerParam.rmsNormEps = param_.rmsNormEps;
            layerParam.headNum = param_.headNum;
            layerParam.dk = param_.dk;
            layerParam.transposedWeight = param_.transposedWeight;
            layerParam.model = "telechat_7B";
            layerParam.isPrefill = param_.isPrefill;
            layerParam.rank = param_.rank;
            layerParam.rankSize = param_.rankSize;
            layerParam.isBF16 = param_.isBF16;
            layerParam.isQuant = false;
  
            PaCommonLayer(layerParam, &op);
            layerNode.operation.reset(op);
            layerNode.inTensors.resize(layerNode.operation->GetInputNum());
    
            size_t inTensorId = 0;
            layerNode.inTensors.at(inTensorId++) = firstInTensor;
            for (size_t weightTensorId = 0; weightTensorId < FLOAT_WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
                layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(
                    WORD_EMBEDDING_WEIGHT_COUNT + layerId * FLOAT_WEIGHT_COUNT_PER_LAYER + weightTensorId);
            }
            layerNode.inTensors.at(inTensorId++) = cosEmbedTensor;
            layerNode.inTensors.at(inTensorId++) = sinEmbedTensor;
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ATTENTIONMASK);  // attentionMaskTensor
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PAST_KEY);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PAST_VALUE);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TOKENOFFSET);           // tokenoffset
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_SEQLEN);                // seqlen
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_HOLDER);                // holder
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_MAX + layerId);  // layerid
    
            layerNode.outTensors = { &graph_.internalTensors.at(INTERNAL_TENSOR_COUNT_BEFORE_LAYER + layerId) };
            firstInTensor = layerNode.outTensors.at(0);
        }
        else {
            // 量化
            ATB_LOG(INFO) << "Enter Pure Quant Layer, layerId: " << layerId;
            layerParam.rmsNormEps = param_.rmsNormEps;
            layerParam.headNum = param_.headNum;
            layerParam.dk = param_.dk;
            layerParam.transposedWeight = param_.transposedWeight;
            layerParam.model = "telechat_7B";
            layerParam.isPrefill = param_.isPrefill;
            layerParam.rank = param_.rank;
            layerParam.rankSize = param_.rankSize;
            layerParam.isBF16 = param_.isBF16;
            layerParam.isQuant = true;

            if (std::find(param_.float_query_layers.begin(), param_.float_query_layers.end(), layerId) !=
                param_.float_query_layers.end()) {
                layerParam.isFloatQueryLayer = true;
                ATB_LOG(INFO) << "layerParam.isFloatQueryLayer" << layerParam.isFloatQueryLayer;
            }
            if (std::find(param_.float_kv_layers.begin(), param_.float_kv_layers.end(), layerId) !=
                param_.float_kv_layers.end()) {
                layerParam.isFloatKVLayer = true;
                ATB_LOG(INFO) << "layerParam.isFloatKVLayer" << layerParam.isFloatKVLayer;
            }
            if (std::find(param_.float_down_layers.begin(), param_.float_down_layers.end(), layerId) !=
                param_.float_down_layers.end()) {
                layerParam.isFloatDownLayer = true;
                ATB_LOG(INFO) << "layerParam.isFloatDownLayer" << layerParam.isFloatDownLayer;
            }
            layerParam.inputScale_qkv = param_.inputScale_qkv[layerId];
            layerParam.inputOffset_qkv = param_.inputOffset_qkv[layerId];
            layerParam.inputScale_dense = param_.inputScale_dense[layerId];
            layerParam.inputOffset_dense = param_.inputOffset_dense[layerId];
            layerParam.inputScale_gate_up = param_.inputScale_gate_up[layerId];
            layerParam.inputOffset_gate_up = param_.inputOffset_gate_up[layerId];
            layerParam.inputScale_down_proj = param_.inputScale_down_proj[layerId];
            layerParam.inputOffset_down_proj = param_.inputOffset_down_proj[layerId];
    
            PaCommonLayer(layerParam, &op);
            layerNode.operation.reset(op);
            layerNode.inTensors.resize(layerNode.operation->GetInputNum());
    
            size_t inTensorId = 0;
            layerNode.inTensors.at(inTensorId++) = firstInTensor;
            for (size_t weightTensorId = 0; weightTensorId < QUANT_WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
                layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(
                    WORD_EMBEDDING_WEIGHT_COUNT + layerId * QUANT_WEIGHT_COUNT_PER_LAYER + weightTensorId);
            }
            layerNode.inTensors.at(inTensorId++) = cosEmbedTensor;
            layerNode.inTensors.at(inTensorId++) = sinEmbedTensor;
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ATTENTIONMASK);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PAST_KEY);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PAST_VALUE);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TOKENOFFSET);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_SEQLEN);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_HOLDER);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_MAX + layerId);
    
            layerNode.outTensors = { &graph_.internalTensors.at(INTERNAL_TENSOR_COUNT_BEFORE_LAYER + layerId) };
            firstInTensor = layerNode.outTensors.at(0);
        }
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
    return atb::NO_ERROR;
}

atb::Status CommonPAModel::ParseParam(const std::string &param)
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

atb::Status CommonPAModel::BindParamHostTensor(uint32_t nodeId)
{
    if (nodeId < OPERATION_COUNT_BEFORE_LAYER || nodeId >= static_cast<uint32_t>(OPERATION_COUNT_BEFORE_LAYER + param_.layerNum)) {
        return atb::NO_ERROR;
    }

    auto &node = graph_.nodes.at(nodeId);
    const uint32_t float_tokenOffsetTensorId = 16;
    const uint32_t float_seqLenTensorId = 17;

    const uint32_t quant_tokenOffsetTensorId = 28;
    const uint32_t quant_seqLenTensorId = 29;
    if (!param_.isQuant) {
        node.variantPack.inTensors.at(float_tokenOffsetTensorId).hostData = tokenOffset_.data();
        node.variantPack.inTensors.at(float_seqLenTensorId).hostData = seqLen_.data();
    }
    else {
        node.variantPack.inTensors.at(quant_tokenOffsetTensorId).hostData = tokenOffset_.data();
        node.variantPack.inTensors.at(quant_seqLenTensorId).hostData = seqLen_.data();
    }


    return atb::NO_ERROR;
}
}  // namespace telechat
}  // namespace atb_speed