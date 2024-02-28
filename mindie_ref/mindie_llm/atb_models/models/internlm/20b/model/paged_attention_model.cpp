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
#include "paged_attention_model.h"

#include "atb/atb_infer.h"
#include "nlohmann/json.hpp"

#include "models/internlm/20b/layer/paged_attention_layer.h"

namespace atb_speed {
namespace internlm_20b {
enum InTensorId : int {
    IN_TENSOR_INPUT_IDS = 0,
    IN_TENSOR_COSEMBED,
    IN_TENSOR_SINEMBED,
    IN_TENSOR_ATTENTION_MASK,
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

const int WEIGHT_COUNT_PER_LAYER = 9;
const int WORDEMBEDDINGNODE_WEIGHT_COUNT = 1;
const int FINALNORMNODE_WEIGHT_COUNT = 1;
const int OUT_LM_HEAD_WEIGHT_COUNT = 1;
const int OPERATION_COUNT_BEFORE_LAYER = 1;
const int INTERMEDIATETENSOR_COUNT_BEFORE_LAYER = 1;
const int OPERATION_COUNT_AFTER_LAYER = 2;
const int OUT_TENSOR_HIDDENSTATES_ID = 0;
const int OUT_TENSOR_MAX_ID = 1;
const int IN_TENSOR_INPUTIDS_ID = 0;
const int WORDEMBEDDINGNODE_WEIGHT_ID = 0;
const int FIRST_INTERNAL_TENSORS = 0;
const int LAYER_FIRST_OUT_TENSORS = 0;
const int FA_ROPE_LAYER_IN_TOKENOFFSET_ID = 15;
const int FA_ROPE_LAYER_IN_SEQLEN_ID = 16;
const int OUT_TENSOR_HIDDENSTATES_ID_DIM_NUM = 3;

void PagedAttentionModel::Param::FromString(const std::string &param)
{
   nlohmann::json paramJson = nlohmann::json::parse(param);
   rmsNormEps = paramJson["rmsNormEps"].get<double>();
   headNum = paramJson["headNum"].get<int>();
   dk = paramJson["dk"].get<int>();
   layerNum = paramJson["layerNum"].get<int>();
   if (paramJson.contains("rank")) {
       rank = paramJson["rank"].get<int>();
   }
   if (paramJson.contains("rankSize")) {
       rankSize = paramJson["rankSize"].get<int>();
   }
   if (paramJson.contains("isPrefill")) {
       isPrefill = paramJson["isPrefill"].get<bool>();
   }
   if (paramJson.contains("backend")) {
       backend = paramJson["backend"];
   }
}

PagedAttentionModel::PagedAttentionModel(const std::string &param) : Model("InternLM_20B_PagedAttentionModel", param)
{
   param_.FromString(param);
   modelName_ += param_.isPrefill ? "_Prefill" : "_Decoder";
}

PagedAttentionModel::~PagedAttentionModel() = default;

uint32_t PagedAttentionModel::GetInputNum() const
{
   return graph_.inTensors.size();
}

uint32_t PagedAttentionModel::GetOutputNum() const
{
   return graph_.outTensors.size();
}

atb::Status PagedAttentionModel::InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
                                               std::vector<atb::TensorDesc> &outTensorDescs)
{
   if (outTensorDescs.size() != GetOutputNum()) {
       return atb::ERROR_INVALID_GRAPH;
   }
   const int64_t outDim = graph_.weightTensors.at(graph_.weightTensors.size() - 1).desc.shape.dims[0];
   outTensorDescs.at(OUT_TENSOR_HIDDENSTATES_ID) = graph_.weightTensors.at(WORDEMBEDDINGNODE_WEIGHT_ID).desc;
   outTensorDescs.at(OUT_TENSOR_HIDDENSTATES_ID).shape.dimNum = OUT_TENSOR_HIDDENSTATES_ID_DIM_NUM;

   size_t outTensorShapeDimIndex = 0;
   size_t inTensorShapeDimIndex = 0;

   outTensorDescs.at(OUT_TENSOR_HIDDENSTATES_ID).shape.dims[outTensorShapeDimIndex++] =
       inTensorDescs.at(IN_TENSOR_INPUTIDS_ID).shape.dims[inTensorShapeDimIndex++];
   outTensorDescs.at(OUT_TENSOR_HIDDENSTATES_ID).shape.dims[outTensorShapeDimIndex++] =
       inTensorDescs.at(IN_TENSOR_INPUTIDS_ID).shape.dims[inTensorShapeDimIndex++];
   outTensorDescs.at(OUT_TENSOR_HIDDENSTATES_ID).shape.dims[outTensorShapeDimIndex++] = outDim;

   return atb::NO_ERROR;
}

int64_t PagedAttentionModel::BuildGraph()
{
   const int weightTensorSize = WORDEMBEDDINGNODE_WEIGHT_COUNT + WEIGHT_COUNT_PER_LAYER * param_.layerNum +
                                FINALNORMNODE_WEIGHT_COUNT + OUT_LM_HEAD_WEIGHT_COUNT;
   graph_.weightTensors.resize(weightTensorSize);

   graph_.kCacheTensors.resize(param_.layerNum);
   graph_.vCacheTensors.resize(param_.layerNum);

   graph_.inTensors.resize(IN_TENSOR_MAX);
   graph_.outTensors.resize(OUT_TENSOR_MAX);

   const int nodeSize = param_.layerNum + OPERATION_COUNT_BEFORE_LAYER + OPERATION_COUNT_AFTER_LAYER;
   graph_.nodes.resize(nodeSize);

   const int internalTensorSize = graph_.nodes.size() - 1;
   graph_.internalTensors.resize(internalTensorSize);

   int nodeId = 0;
   auto &wordEmbeddingNode = graph_.nodes.at(nodeId++);
   atb::infer::GatherParam wordEmbeddingParam;
   atb::Operation *op = nullptr;
   CREATE_OPERATION(wordEmbeddingParam, &op);
   wordEmbeddingNode.operation.reset(op);
   wordEmbeddingNode.inTensors = { &graph_.weightTensors.at(WORDEMBEDDINGNODE_WEIGHT_ID),
                                  &graph_.inTensors.at(IN_TENSOR_INPUTIDS) };
   wordEmbeddingNode.outTensors = { &graph_.internalTensors.at(FIRST_INTERNAL_TENSORS) };

   atb::Tensor *firstInTensor = &graph_.internalTensors.at(FIRST_INTERNAL_TENSORS);

   for (int layerId = 0; layerId < param_.layerNum; ++layerId) {
       auto &layerNode = graph_.nodes.at(nodeId++);

       atb_speed::internlm_20b::PagedAttentionLayerParam opParam;
       opParam.rmsNormEps = param_.rmsNormEps;
       opParam.headNum = param_.headNum;
       opParam.dk = param_.dk;
       opParam.rank = param_.rank;
       opParam.rankSize = param_.rankSize;
       opParam.isPrefill = param_.isPrefill;
       opParam.backend = param_.backend;
       atb_speed::internlm_20b::PagedAttentionLayer(opParam, &op);
       layerNode.operation.reset(op);
       layerNode.inTensors.resize(layerNode.operation->GetInputNum());

       size_t inTensorId = 0;
       layerNode.inTensors.at(inTensorId++) = firstInTensor;
       for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
           layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(layerId * WEIGHT_COUNT_PER_LAYER +
                                                                           weightTensorId + WORDEMBEDDINGNODE_WEIGHT_COUNT);
       }
       layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_COSEMBED);      // cosEmbed
       layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SINEMBED);      // sinEmbed
       layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ATTENTIONMASK); // attentionMaskTensor
       layerNode.inTensors.at(inTensorId++) = &graph_.kCacheTensors.at(layerId);
       layerNode.inTensors.at(inTensorId++) = &graph_.vCacheTensors.at(layerId);
       layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_BLOCK_TABLES);
       layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SLOTS);
       layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_INPUT_LENGTHS);
       layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_HOLDER);

       layerNode.outTensors = { &graph_.internalTensors.at(INTERMEDIATETENSOR_COUNT_BEFORE_LAYER + layerId) };

       firstInTensor = layerNode.outTensors.at(LAYER_FIRST_OUT_TENSORS);
   }

   auto &finalNormNode = graph_.nodes.at(nodeId++);
   atb::infer::RmsNormParam finalNormParam;
   finalNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
   finalNormParam.normParam.epsilon = param_.rmsNormEps;
   CREATE_OPERATION(finalNormParam, &op);
   finalNormNode.operation.reset(op);
   const int finalLayerNormWeightTensorId =
       graph_.weightTensors.size() - FINALNORMNODE_WEIGHT_COUNT - OUT_LM_HEAD_WEIGHT_COUNT;
   const int finalLayerNormOutTensorId = internalTensorSize - 1;
   finalNormNode.inTensors = { firstInTensor, &graph_.weightTensors.at(finalLayerNormWeightTensorId) };
   finalNormNode.outTensors = { &graph_.internalTensors.at(finalLayerNormOutTensorId) };

   auto &outLinearNode = graph_.nodes.at(nodeId++);
   atb::infer::LinearParam outLinearParm;
   outLinearParm.hasBias = false;
   CREATE_OPERATION(outLinearParm, &op);
   outLinearNode.operation.reset(op);
   const int finalLinearWeightTensorId = graph_.weightTensors.size() - OUT_LM_HEAD_WEIGHT_COUNT;
   if (param_.isPrefill) {
       outLinearNode.inTensors = {&graph_.internalTensors.at(finalLayerNormOutTensorId),
                               &graph_.weightTensors.at(finalLinearWeightTensorId),
                               &graph_.inTensors.at(IN_TENSOR_LOGTIS_INDICES)};
   } else {
       outLinearNode.inTensors = {&graph_.internalTensors.at(finalLayerNormOutTensorId),
                               &graph_.weightTensors.at(finalLinearWeightTensorId)};
   }
//   outLinearNode.inTensors = { &graph_.internalTensors.at(finalLayerNormOutTensorId),
//                              &graph_.weightTensors.at(finalLinearWeightTensorId) };
   outLinearNode.outTensors = { &graph_.outTensors.at(OUT_TENSOR_HIDDENSTATES_ID) };
   return atb::NO_ERROR;
}

atb::Status PagedAttentionModel::ParseParam(const std::string &param)
{
   nlohmann::json paramJson = nlohmann::json::parse(param);
   tokenOffset_.clear();
   seqLen_.clear();
   for (const auto &item : paramJson["seqLen"]) {
       seqLen_.push_back(item.get<int>());
   }

   return atb::NO_ERROR;
}

atb::Status PagedAttentionModel::BindParamHostTensor(uint32_t nodeId)
{
   if (nodeId < OPERATION_COUNT_BEFORE_LAYER || nodeId >= OPERATION_COUNT_BEFORE_LAYER + param_.layerNum) {
       return atb::NO_ERROR;
   }
   auto &node = graph_.nodes.at(nodeId);

   const uint32_t seqLenTensorId = FA_ROPE_LAYER_IN_SEQLEN_ID;

   node.variantPack.inTensors.at(seqLenTensorId).hostData = seqLen_.data();
   ATB_LOG(INFO) << "BindParamHostTensor end";
   return atb::NO_ERROR;
}
} // namespace internlm_20b
} // namespace atb_speed