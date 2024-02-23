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

#include "flash_attention_model.h"
#include <atb/atb_infer.h>
#include <nlohmann/json.hpp>
#include "atb_speed/log.h"
#include "models/chatglm/6b/layer/flash_attention_layer.h"
#include "layers/parallel_layer_v2.h"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wsign-compare"

namespace atb_speed {
namespace chatglm_6b {
const int WEIGHT_COUNT_BEFORE_LAYER = 1;
const int WEIGHT_COUNT_AFTER_LAYER = 3;
const int WEIGHT_COUNT_QUANT_LAYER = 16;
const int WEIGHT_COUNT_SPARSE_LAYER = 28;
const int WEIGHT_COUNT_FLOAT_LAYER = 12;
const int FINALNORMNODE_WEIGHT_COUNT = 2;
const int OPERATION_COUNT_BEFORE_LAYER = 1;
const int OPERATION_COUNT_AFTER_LAYER = 3;
const int INTERMEDIATETENSOR_COUNT_BEFORE_LAYER = 1;
const int INTERMEDIATETENSOR_COUNT_AFTER_LAYER = 2;
const int DIM1 = 1;
const int DIM2 = 2;
const int DIM3 = 3;

enum InTensorId : int {
    IN_TENSOR_HIDDENSTATES = 0,
    IN_TENSOR_COS,
    IN_TENSOR_SIN,
    IN_SEQLEN,
    IN_ATTENTION_MASK,
    IN_TOKEN_OFFSET,
    IN_TENSOR_PASTKEY,
    IN_TENSOR_PASTVALUE,
    IN_TENSOR_BETA,
    IN_PLACE_HOLDER,
    IN_TENSOR_MAX,
};

enum OutTensorId : int {
    OUT_TENSOR_HIDDENSTATES = 0,
    OUT_TENSOR_MAX,
};

void ChatGlmCommonModelFa::Param::FromString(const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);
    rmsNormEps = paramJson["rmsNormEps"].get<float>();
    numHeadsPerPartition = paramJson["numHeadsPerPartition"].get<int>();
    hiddenSizePerHead = paramJson["hiddenSizePerHead"].get<int>();
    transKey = paramJson["transKey"].get<bool>();
    layerNum = paramJson["layerNum"].get<int>();
    residualAddScale = paramJson["residualAddScale"].get<float>();
    quantmodel = paramJson["quantmodel"].get<bool>();
    isSparse = paramJson["isSparse"].get<bool>();
    correctNodeId = paramJson["correctNodeId"].get<int>();

    for (auto item : paramJson["qkvInputScale"]) {
        qkvInputScale.push_back(item.get<float>());
    }
    for (auto item : paramJson["qkvInputOffset"]) {
        qkvInputOffset.push_back(item.get<int>());
    }
    for (auto item : paramJson["denseInputScale"]) {
        denseInputScale.push_back(item.get<float>());
    }
    for (auto item : paramJson["denseInputOffset"]) {
        denseInputOffset.push_back(item.get<int>());
    }
    for (auto item : paramJson["selfLnInputScale"]) {
        selfLnInputScale.push_back(item.get<float>());
    }
    for (auto item : paramJson["selfLnInputOffset"]) {
        selfLnInputOffset.push_back(item.get<int>());
    }
    for (auto item : paramJson["ffnOutInputScale"]) {
        ffnOutInputScale.push_back(item.get<float>());
    }
    for (auto item : paramJson["ffnOutInputOffset"]) {
        ffnOutInputOffset.push_back(item.get<int>());
    }
    for (auto item : paramJson["preScale"]) {
        preScale.push_back(item.get<float>());
    }
    for (auto item : paramJson["postScale"]) {
        postScale.push_back(item.get<float>());
    }
    for (auto item : paramJson["offsetX"]) {
        offsetX.push_back(item.get<int>());
    }
    for (auto item : paramJson["compressInfo"]) {
        std::vector<int64_t> tmp = {};
        for (auto i: item) {
            tmp.push_back(i);
        }
        compressInfo.push_back(tmp);
    }
    rank = paramJson["rank"].get<int>();
    rankSize = paramJson["rankSize"].get<int>();
    backend = paramJson["backend"].get<std::string>();
    isEncoder = paramJson["isEncoder"].get<bool>();
    ATB_LOG(INFO) << "ChatGlmQuantMixDecoderParallelModelFa param rmsNormEps:" << rmsNormEps
                  << ", numHeadsPerPartition:" << numHeadsPerPartition << ", hiddenSizePerHead:" << hiddenSizePerHead
                  << ", transKey:" << transKey << ", layerNum:" << layerNum << ", residualAddScale:" << residualAddScale
                  << ", quantmodel" << quantmodel << ", correctNodeId" << correctNodeId << ", isEncoder" << isEncoder;
}

ChatGlmCommonModelFa::ChatGlmCommonModelFa(const std::string &param)
    : Model("ChatGlmCommonModelFa", param)
{
    param_.FromString(param);
}

ChatGlmCommonModelFa::~ChatGlmCommonModelFa() {}

uint32_t ChatGlmCommonModelFa::GetInputNum() const
{
    return graph_.inTensors.size();
}

uint32_t ChatGlmCommonModelFa::GetOutputNum() const
{
    return graph_.outTensors.size();
}

atb::Status ChatGlmCommonModelFa::InferShape(
    const std::vector<atb::TensorDesc> &inTensorDescs, std::vector<atb::TensorDesc> &outTensorDescs)
{
    if (outTensorDescs.size() != GetOutputNum()) {
        return atb::ERROR_INVALID_GRAPH;
    }

    const int64_t outDim = graph_.weightTensors.at(graph_.weightTensors.size() - 1).desc.shape.dims[0];
    outTensorDescs.at(0) = graph_.weightTensors.at(0).desc;
    outTensorDescs.at(0).shape.dimNum = DIM3;
    outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
    outTensorDescs.at(0).shape.dims[1] = DIM1;
    outTensorDescs.at(0).shape.dims[2] = outDim * param_.rankSize; // 对输出的1，2，3轴infershape
    return atb::NO_ERROR;
}

int64_t ChatGlmCommonModelFa::BuildGraph()
{
    int weightTensorSize = 0;
    if (param_.quantmodel && param_.isSparse) {
        weightTensorSize =
        WEIGHT_COUNT_FLOAT_LAYER + WEIGHT_COUNT_SPARSE_LAYER * (param_.layerNum - 1)
        + WEIGHT_COUNT_AFTER_LAYER + WEIGHT_COUNT_BEFORE_LAYER;
    } else if (param_.quantmodel) {
        weightTensorSize =
        WEIGHT_COUNT_FLOAT_LAYER + WEIGHT_COUNT_QUANT_LAYER * (param_.layerNum - 1)
        + WEIGHT_COUNT_AFTER_LAYER + WEIGHT_COUNT_BEFORE_LAYER;
    } else {
        weightTensorSize = WEIGHT_COUNT_FLOAT_LAYER * param_.layerNum
        + WEIGHT_COUNT_AFTER_LAYER + WEIGHT_COUNT_BEFORE_LAYER;
    }

    ATB_LOG(INFO) << "weight Tensor Size: " << weightTensorSize;
    graph_.weightTensors.resize(weightTensorSize);

    graph_.inTensors.resize(IN_TENSOR_MAX + param_.layerNum);
    graph_.outTensors.resize(OUT_TENSOR_MAX);

    const int nodeSize = param_.layerNum + OPERATION_COUNT_BEFORE_LAYER + OPERATION_COUNT_AFTER_LAYER;
    graph_.nodes.resize(nodeSize);

    graph_.internalTensors.resize(graph_.nodes.size() - 1);

    atb::Operation *op = nullptr;
    int nodeId = 0;

    // before layers
    auto &wordEmbeddingNode = graph_.nodes.at(nodeId++);
    atb::infer::GatherParam wordEmbeddingParam;
    CREATE_OPERATION(wordEmbeddingParam, &op);
    wordEmbeddingNode.operation.reset(op);
    wordEmbeddingNode.inTensors = {&graph_.weightTensors.at(0), &graph_.inTensors.at(0)};
    wordEmbeddingNode.outTensors = {&graph_.internalTensors.at(0)};

    // layers
    atb::Tensor *firstInTensor = &graph_.internalTensors.at(0);
    size_t weight_id = WEIGHT_COUNT_BEFORE_LAYER;

    for (int layerId = 0; layerId < param_.layerNum; ++layerId) {
        if ((param_.quantmodel && (layerId == param_.correctNodeId)) || (!param_.quantmodel)) {
            auto &layerNode = graph_.nodes.at(nodeId++);
            CommonLayerParamFa opParam;
            opParam.rmsNormEps = param_.rmsNormEps;
            opParam.numHeadsPerPartition = param_.numHeadsPerPartition;
            opParam.hiddenSizePerHead = param_.hiddenSizePerHead;
            opParam.transKey = param_.transKey;
            opParam.layerId = layerId;
            opParam.residualAddScale = param_.residualAddScale;
            opParam.preScale = param_.preScale.at(layerId);
            opParam.postScale = param_.postScale.at(layerId);
            opParam.rank = param_.rank;
            opParam.rankSize = param_.rankSize;
            opParam.backend = param_.backend;
            opParam.isEncoder = param_.isEncoder;
            opParam.quantmodel = false;
            CommonLayerFa(opParam, &op);
            layerNode.operation.reset(op);
            layerNode.inTensors.resize(layerNode.operation->GetInputNum());
            layerNode.outTensors.resize(layerNode.operation->GetOutputNum());
            size_t inTensorId = 0;
            layerNode.inTensors.at(inTensorId++) = firstInTensor;
            for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_FLOAT_LAYER; ++weightTensorId) {
                layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(weight_id++);
            }
            size_t holder_num = WEIGHT_COUNT_SPARSE_LAYER - WEIGHT_COUNT_FLOAT_LAYER;
            for (size_t weightTensorId = 0; weightTensorId < holder_num; ++weightTensorId) {
                layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_PLACE_HOLDER); // 补全浮点相比量化缺少的权重个数
            }
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_COS);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SIN);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_SEQLEN);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_ATTENTION_MASK);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PASTKEY);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PASTVALUE);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TOKEN_OFFSET);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_BETA);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_PLACE_HOLDER);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_MAX + layerId);
            layerNode.outTensors = {&graph_.internalTensors.at(layerId + INTERMEDIATETENSOR_COUNT_BEFORE_LAYER)};
            firstInTensor = layerNode.outTensors.at(0);
        } else {
            auto &layerNode = graph_.nodes.at(nodeId++);
            CommonLayerParamFa opParam;
            opParam.rmsNormEps = param_.rmsNormEps;
            opParam.numHeadsPerPartition = param_.numHeadsPerPartition;
            opParam.hiddenSizePerHead = param_.hiddenSizePerHead;
            opParam.transKey = param_.transKey;
            opParam.layerId = layerId;
            opParam.residualAddScale = param_.residualAddScale;
            opParam.preScale = param_.preScale.at(layerId);
            opParam.postScale = param_.postScale.at(layerId);
            opParam.qkvInputScale = param_.qkvInputScale[layerId];
            opParam.qkvInputOffset = param_.qkvInputOffset[layerId];
            opParam.denseInputScale = param_.denseInputScale[layerId];
            opParam.denseInputOffset = param_.denseInputOffset[layerId];
            opParam.selfLnInputScale = param_.selfLnInputScale[layerId];
            opParam.selfLnInputOffset = param_.selfLnInputOffset[layerId];
            opParam.ffnOutInputScale = param_.ffnOutInputScale[layerId];
            opParam.ffnOutInputOffset = param_.ffnOutInputOffset[layerId];
            opParam.quantmodel = true;
            opParam.rank = param_.rank;
            opParam.rankSize = param_.rankSize;
            opParam.backend = param_.backend;
            opParam.isEncoder = param_.isEncoder;
            opParam.isSparse = param_.isSparse;
            CommonLayerFa(opParam, &op);
            layerNode.operation.reset(op);
            layerNode.inTensors.resize(layerNode.operation->GetInputNum());
            layerNode.outTensors.resize(layerNode.operation->GetOutputNum());
            size_t inTensorId = 0;
            layerNode.inTensors.at(inTensorId++) = firstInTensor;
            if (param_.isSparse) {
                for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_SPARSE_LAYER; ++weightTensorId) {
                    layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(weight_id++);
                }
            } else {
                for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_QUANT_LAYER; ++weightTensorId) {
                    layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(weight_id++);
                }
                size_t holder_num = WEIGHT_COUNT_SPARSE_LAYER - WEIGHT_COUNT_QUANT_LAYER;
                for (size_t weightTensorId = 0; weightTensorId < holder_num; ++weightTensorId) {
                    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_PLACE_HOLDER); // 补全量化相比稀疏缺少的权重个数
                }
            }
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_COS);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SIN);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_SEQLEN);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_ATTENTION_MASK);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PASTKEY);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PASTVALUE);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TOKEN_OFFSET);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_BETA);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_PLACE_HOLDER);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_MAX + layerId);

            layerNode.outTensors = {&graph_.internalTensors.at(layerId + INTERMEDIATETENSOR_COUNT_BEFORE_LAYER)};
            firstInTensor = layerNode.outTensors.at(0);
        }
    }

    // after layers
    int internalTensorId = graph_.internalTensors.size() - INTERMEDIATETENSOR_COUNT_AFTER_LAYER;
    int weightTensorId = graph_.weightTensors.size() - WEIGHT_COUNT_AFTER_LAYER;

    auto &finalNormNode = graph_.nodes.at(nodeId++);
    atb::infer::LayerNormParam finalNormParam;
    finalNormParam.layerType = atb::infer::LayerNormParam::LayerNormType::LAYER_NORM_NORM;
    finalNormParam.normParam.beginNormAxis = 2;
    finalNormParam.normParam.beginParamsAxis = 1;
    finalNormParam.normParam.epsilon = param_.rmsNormEps;
    CREATE_OPERATION(finalNormParam, &op);
    finalNormNode.operation.reset(op);
    finalNormNode.inTensors = {firstInTensor, &graph_.weightTensors.at(weightTensorId++), &graph_.weightTensors.at(weightTensorId++)};
    finalNormNode.outTensors = {&graph_.internalTensors.at(internalTensorId)};

    auto &sliceNode = graph_.nodes.at(nodeId++);
    atb::infer::SliceParam sliceParam;
    sliceParam.offsets = {0, -1, 0};
    sliceParam.size = {-1, 1, -1};
    CREATE_OPERATION(sliceParam, &op);
    sliceNode.operation.reset(op);
    sliceNode.inTensors = {&graph_.internalTensors.at(internalTensorId++)};
    sliceNode.outTensors = {&graph_.internalTensors.at(internalTensorId)};

    auto &lmNode = graph_.nodes.at(nodeId++);
    atb_speed::common::ParallelParamV2 lmParam = {false, false, true, param_.quantmodel, param_.isSparse};
    lmParam.isAllGatherTranspose = true;
    lmParam.commParam.rank = param_.rank;
    lmParam.commParam.rankSize = param_.rankSize;
    lmParam.commParam.backend = param_.backend;
    atb_speed::common::ColumnParallelLinearV2(lmParam, &op);
    lmNode.operation.reset(op);
    lmNode.inTensors = {&graph_.internalTensors.at(internalTensorId++),
        &graph_.weightTensors.at(weightTensorId),
        &graph_.inTensors.at(IN_PLACE_HOLDER),
        &graph_.inTensors.at(IN_PLACE_HOLDER),
        &graph_.inTensors.at(IN_PLACE_HOLDER),
        &graph_.inTensors.at(IN_PLACE_HOLDER),
        &graph_.inTensors.at(IN_PLACE_HOLDER)
    };
    lmNode.outTensors = {&graph_.outTensors.at(0)};

    return atb::NO_ERROR;
}

atb::Status ChatGlmCommonModelFa::ParseParam(const std::string &param)
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
 
    ATB_LOG(INFO) << "ChatGlmCommonModelFa ParseParam tokenOffset:"
                  << tokenOffset_ << ", seqLen:" << seqLen_;
 
    return atb::NO_ERROR;
}
 
atb::Status ChatGlmCommonModelFa::BindParamHostTensor(uint32_t nodeId)
{
    if (nodeId < OPERATION_COUNT_BEFORE_LAYER || nodeId >= OPERATION_COUNT_BEFORE_LAYER + param_.layerNum) {
        return atb::NO_ERROR;
    }
    const uint32_t index_bind = 4;
    const uint32_t InttokenOffsetTensorId = 35;
    const uint32_t IntseqLenTensorId = 31;

    auto &node = graph_.nodes.at(nodeId);

    node.variantPack.inTensors.at(InttokenOffsetTensorId).hostData = tokenOffset_.data();
    node.variantPack.inTensors.at(IntseqLenTensorId).hostData = seqLen_.data();
 
    if (nodeId > 1 && nodeId <= 28) { // node从第2个到第28个为量化layer
        std::vector<int32_t> nodeOffsetXIds = {16, 19, 22, 25};
        std::vector<int32_t> nodeCompressInfoIds = {17, 20, 23, 26};

        int32_t i = 0;
        int32_t j = (nodeId-2)*index_bind+i; // 第2个node开始
        for (i=0; i < nodeOffsetXIds.size(); i++, j++) {
            node.variantPack.inTensors.at(nodeOffsetXIds[i]).hostData = &param_.offsetX.at(j);
        }

        i = 0;
        j = (nodeId-2)*index_bind+i; // 第2个node开始
        for (i=0; i < nodeOffsetXIds.size(); i++, j++) {
            node.variantPack.inTensors.at(nodeCompressInfoIds[i]).hostData = param_.compressInfo.at(j).data();
        }
    }

    return atb::NO_ERROR;
}
} // namespace chatglm_6b
} // namespace atb_speed
