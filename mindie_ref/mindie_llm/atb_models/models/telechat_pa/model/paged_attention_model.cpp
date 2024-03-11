/* quant_layer
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
#include "models/telechat_pa/layer/paged_attention_layer.h"
#include "telechat/layer/embedding_layer.h"
#include "paged_attention_model.h"
#include "atb_speed/utils/model_factory.h"

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
    IN_TENSOR_POSITIONID,
    IN_TENSOR_COSEMBED,
    IN_TENSOR_SINEMBED,
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

REGISTER_MODEL(telechat, PAModel);

int64_t PAModel::Param::FromString(const std::string &param)
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
    };
    if (paramJson.contains("isPrefill")) {
        isPrefill = paramJson["isPrefill"].get<bool>();
    };
    if (headNum == 0) {
        ATB_LOG(ERROR) << "param.headNum is 0, please input a correct value";
        return atb::ERROR_INVALID_PARAM;
    };
    if (dk == 0) {
        ATB_LOG(ERROR) << "param.dk is 0, please input a correct value";
        return atb::ERROR_INVALID_PARAM;
    };
    ATB_LOG(INFO) << "Llama PAModel param rmsNormEps:" << rmsNormEps << ", headNum:" << headNum << ", dk:" << dk
                  << ", layerNum:" << layerNum << ", transposedWeight:" << transposedWeight << ", rank:" << rank
                  << ", rankSize:" << rankSize;
    
    return atb::NO_ERROR;
}

PAModel::PAModel(const std::string &param) : Model("TelechatPAModel", param)
{
    param_.FromString(param);
    modelName_ += param_.isPrefill ? "_Prefill" : "_Decoder";
}

PAModel::~PAModel() {}
uint32_t PAModel::GetInputNum() const
{return graph_.inTensors.size(); }

uint32_t PAModel::GetOutputNum() const
{return graph_.outTensors.size(); }

atb::Status PAModel::InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
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

int64_t PAModel::BuildGraph()
{
    ATB_LOG(INFO) << "Enter Telechat FloatFAModel BuildGraph";
    const int weightTensorSize = WORD_EMBEDDING_WEIGHT_COUNT + WEIGHT_COUNT_PER_LAYER * param_.layerNum +
                                 FINAL_RMSNORM_WEIGHT_COUNT + OUT_LM_HEAD_WEIGHT_COUNT;

    graph_.weightTensors.resize(weightTensorSize);

    graph_.kCacheTensors.resize(param_.layerNum);
    graph_.vCacheTensors.resize(param_.layerNum);

    const int inTensorSize = IN_TENSOR_MAX + param_.layerNum;
    graph_.inTensors.resize(inTensorSize);

    const int outTensorSize = OUT_TENSOR_NUM;
    graph_.outTensors.resize(outTensorSize);

    const int nodeSize = OPERATION_COUNT_BEFORE_LAYER + param_.layerNum + OPERATION_COUNT_AFTER_LAYER;
    ATB_LOG(INFO) << "TeleChat_7b_PAModel nodeSize is " << nodeSize;
    graph_.nodes.resize(nodeSize);

    const int internalTensorSize = graph_.nodes.size() + 1;
    graph_.internalTensors.resize(internalTensorSize);

    int nodeId = 0;
    auto &wordEmbeddingNode = graph_.nodes.at(nodeId++);
    atb::infer::GatherParam wordEmbeddingParam;
    atb::Operation *op = nullptr;
    atb::CreateOperation(wordEmbeddingParam, &op);
    wordEmbeddingNode.operation.reset(op);
    wordEmbeddingNode.inTensors = {&graph_.weightTensors.at(0), &graph_.inTensors.at(IN_TENSOR_INPUT_IDS)};
    wordEmbeddingNode.outTensors = {&graph_.internalTensors.at(0)};
    
    atb::Tensor *firstInTensor = &graph_.internalTensors.at(0);

    for (int layerId = 0; layerId < param_.layerNum; ++layerId) {
        auto &layerNode = graph_.nodes.at(nodeId++);
        PALayerParam layerParam;

        ATB_LOG(INFO) << "Enter Float Layer, layerId: " << layerId;
        layerParam.rmsNormEps = param_.rmsNormEps;
        layerParam.headNum = param_.headNum;
        layerParam.dk = param_.dk;
        layerParam.transposedWeight = param_.transposedWeight;
        layerParam.model = "telechat_7B";
        layerParam.isPrefill = param_.isPrefill;
        layerParam.rank = param_.rank;
        layerParam.rankSize = param_.rankSize;

        PALayer(layerParam, &op);
        layerNode.operation.reset(op);
        layerNode.inTensors.resize(layerNode.operation->GetInputNum());

        size_t inTensorId = 0;
        layerNode.inTensors.at(inTensorId++) = firstInTensor;
        for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
            layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(
                WORD_EMBEDDING_WEIGHT_COUNT + layerId * WEIGHT_COUNT_PER_LAYER + weightTensorId);
        }
        
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_COSEMBED);      // cosEmbed
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SINEMBED);      // sinEmbed
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ATTENTIONMASK);  // attentionMaskTensor
        layerNode.inTensors.at(inTensorId++) = &graph_.kCacheTensors.at(layerId);
        layerNode.inTensors.at(inTensorId++) = &graph_.vCacheTensors.at(layerId);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_BLOCK_TABLES);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SLOTS);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_INPUT_LENGTHS);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_HOLDER);

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
    atb::infer::LinearParam linearParam;
    linearParam.hasBias = false;
    CREATE_OPERATION(linearParam, &op);
    lmHeadNode.operation.reset(op);
    const int finalLmHeadWeightTensorId = graph_.weightTensors.size() - OUT_LM_HEAD_WEIGHT_COUNT;
    
    if (param_.isPrefill) {
        lmHeadNode.inTensors = {&graph_.internalTensors.at(finalRmsNormOutTensorId),
                                &graph_.weightTensors.at(finalLmHeadWeightTensorId),
                                &graph_.inTensors.at(IN_TENSOR_LOGTIS_INDICES)};
    } else {
        lmHeadNode.inTensors = { &graph_.internalTensors.at(finalRmsNormOutTensorId),
                                &graph_.weightTensors.at(finalLmHeadWeightTensorId) };
    }
    lmHeadNode.outTensors = { &graph_.outTensors.at(0) };
    return atb::NO_ERROR;
}

atb::Status  PAModel::ParseParam(const std::string &param)
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

atb::Status  PAModel::BindParamHostTensor(uint32_t nodeId)
{
    if (nodeId < OPERATION_COUNT_BEFORE_LAYER || nodeId >= static_cast<uint32_t>(OPERATION_COUNT_BEFORE_LAYER + param_.layerNum)) {
        return atb::NO_ERROR;
    }

    auto &node = graph_.nodes.at(nodeId);
    const uint32_t seqLenTensorId = 19;
    node.variantPack.inTensors.at(seqLenTensorId).hostData = seqLen_.data();

    return atb::NO_ERROR;
}
}  // namespace telechat
}  // namespace atb_speed