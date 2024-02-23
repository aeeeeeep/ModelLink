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

#include "atb/atb_infer.h"
#include "nlohmann/json.hpp"

#include "layers/parallel_layer_v2.h"
#include "models/vlmo/2b/layer/encoder_layer.h"
#include "models/vlmo/2b/layer/encoder_vl_layer.h"

namespace atb_speed {
namespace vlmo {
const int WEIGHT_COUNT_PER_LAYER = 23;
const int WEIGHT_COUNT_PER_VL_LAYER = 17;

const int FINALNORMNODE_WEIGHT_COUNT = 2;
const int OUT_LM_HEAD_WEIGHT_COUNT = 2;
const int INTERMEDIATETENSOR_COUNT_BEFORE_VL_LAYER = 1;
const int OPERATION_COUNT_AFTER_LAYER = 0;
const int OPERATION_COUNT_BEFORE_LAYER = 0;

enum InTensorId : int {
    IN_TENSOR_X = 0,
    IN_TENSOR_MASK,
    IN_TENSOR_PAST_KEY,
    IN_TENSOR_PAST_VALUE,
    IN_TENSOR_TOKENOFFSET,
    IN_TENSOR_SEQLEN,
    IN_HOLDER,
    IN_TENSOR_MAX, // 7
};

enum OutTensorId : int {

    OUT_LAYEROUT = 0,
    OUT_TENSOR_MAX,
};

void FlashAttentionModel::Param::FromString(const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);
    layerNormEps = paramJson["rmsNormEps"].get<double>();
    headNum = paramJson["headNum"].get<int>();
    dk = paramJson["dk"].get<int>();
    layerNum = paramJson["layerNum"].get<int>();
    if (paramJson.contains("rank")) {
        rank = paramJson["rank"].get<int>();
    }
    if (paramJson.contains("rankSize")) {
        rankSize = paramJson["rankSize"].get<int>();
    }
    if (paramJson.contains("backend")) {
        backend = paramJson["backend"];
    }
    if (paramJson.contains("maxTextLen")) {
        maxTextLen = paramJson["maxTextLen"];
    }
    if (paramJson.contains("vlLayerIndex")) {
        vlLayerIndex = paramJson["vlLayerIndex"];
    }
}

FlashAttentionModel::FlashAttentionModel(const std::string &param) : Model("FlashAttentionModel", param)
{
    param_.FromString(param);
}

FlashAttentionModel::~FlashAttentionModel() = default;

uint32_t FlashAttentionModel::GetInputNum() const { return graph_.inTensors.size(); }

uint32_t FlashAttentionModel::GetOutputNum() const { return graph_.outTensors.size(); }

atb::Status FlashAttentionModel::InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
                                                std::vector<atb::TensorDesc> &outTensorDescs)
{
    if (outTensorDescs.size() != GetOutputNum()) {
        return atb::ERROR_INVALID_GRAPH;
    }
    outTensorDescs.at(0) = inTensorDescs.at(0);
    
    // outTensorDescs.at(0) = graph_.inTensors.at(0).desc;
    // outTensorDescs.at(0).shape.dimNum = 3;
    // outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
    // outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1];
    // outTensorDescs.at(0).shape.dims[2] = inTensorDescs.at(0).shape.dims[2];

    return atb::NO_ERROR;
}

int64_t FlashAttentionModel::BuildGraph()
{
    const int weightTensorSize = WEIGHT_COUNT_PER_LAYER * param_.vlLayerIndex +
                                 WEIGHT_COUNT_PER_VL_LAYER * (param_.layerNum - param_.vlLayerIndex) ;
    graph_.weightTensors.resize(weightTensorSize);

    graph_.inTensors.resize(IN_TENSOR_MAX + param_.layerNum);
    graph_.outTensors.resize(OUT_TENSOR_MAX);

    const int nodeSize = OPERATION_COUNT_BEFORE_LAYER + param_.layerNum  + OPERATION_COUNT_AFTER_LAYER;
    graph_.nodes.resize(nodeSize);

    const int internalTensorSize = (param_.layerNum - 1) * OUT_TENSOR_MAX;
    graph_.internalTensors.resize(internalTensorSize);

    int nodeId = 0;

    atb::Operation *op = nullptr;
    // auto &splitNode = graph_.nodes.at(nodeId++);
    // atb::infer::SplitParam splitParam;
    // splitParam.splitDim = 0;
    // splitParam.splitNum = param_.layerNum;

    
    // CREATE_OPERATION(splitParam, &op);
    // splitNode.operation.reset(op);
    // splitNode.inTensors = {&graph_.inTensors.at(IN_TENSOR_RELATIVE_POSITION_BIAS_LIST)};
    // for(int i =0 ;i< param_.layerNum;i++){
    //     splitNode.outTensors.at(i) = &graph_.internalTensors.at(i);
    // }
    
    
    ATB_LOG(INFO) << __func__ << " called, layerNum: " << param_.layerNum;
    atb::Tensor *firstInTensor = &graph_.inTensors.at(0);
    int layerId = 0;
    for (; layerId < param_.vlLayerIndex; ++layerId) {//0-9层共10层
        ATB_LOG(INFO) << __func__ << " layerId " << layerId << " create node";
        auto &layerNode = graph_.nodes.at(nodeId++);

        atb_speed::vlmo::EncoderLayerParam opParam;
        opParam.layerNormEps = param_.layerNormEps;
        opParam.headNum = param_.headNum;
        opParam.dk = param_.dk;
        opParam.rank = param_.rank;
        opParam.rankSize = param_.rankSize;
        opParam.backend = param_.backend;
        opParam.maxTextLen = param_.maxTextLen;
        atb_speed::vlmo::EncoderLayer(opParam, &op);
        layerNode.operation.reset(op);
        layerNode.inTensors.resize(layerNode.operation->GetInputNum()); 

        size_t inTensorId = 0;
        layerNode.inTensors.at(inTensorId++) = firstInTensor;
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_MASK); // attentionMaskTensor
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PAST_KEY);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PAST_VALUE);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_TOKENOFFSET);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SEQLEN); // seqLen
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_HOLDER);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_MAX + layerId);
        
        for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
            ATB_LOG(INFO) << __func__ << " layerId " << layerId << " weightID"<<weightTensorId << " -> in weight ID" << (layerId * WEIGHT_COUNT_PER_LAYER + weightTensorId );
            layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(
                layerId * WEIGHT_COUNT_PER_LAYER + weightTensorId );
        }
        

        layerNode.outTensors.resize(layerNode.operation->GetOutputNum());
        for (int i=0;i< OUT_TENSOR_MAX;i++ ){
            layerNode.outTensors.at(i) = &graph_.internalTensors.at((layerId * 1) + i );
        }
        firstInTensor = layerNode.outTensors.at(0);
    }
    for (; layerId < param_.layerNum; ++layerId) {//10-11 两层
        ATB_LOG(INFO) << __func__ << " layerId " << layerId << " create node";
        auto &layerNode = graph_.nodes.at(nodeId++);

        atb_speed::vlmo::EncoderVllayerParam opParam;
        opParam.layerNormEps = param_.layerNormEps;
        opParam.headNum = param_.headNum;
        opParam.dk = param_.dk;
        opParam.rank = param_.rank;
        opParam.rankSize = param_.rankSize;
        opParam.backend = param_.backend;
        opParam.maxTextLen = param_.maxTextLen;
        atb_speed::vlmo::EncoderVlLayer(opParam, &op);
        layerNode.operation.reset(op);
        layerNode.inTensors.resize(layerNode.operation->GetInputNum()); // .at 需要resize，直接赋值不需要

        size_t inTensorId = 0;
        layerNode.inTensors.at(inTensorId++) = firstInTensor;
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_MASK); // attentionMaskTensor
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PAST_KEY);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PAST_VALUE);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_TOKENOFFSET);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SEQLEN); // seqLen
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_HOLDER);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_MAX + layerId);
        
        for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_VL_LAYER; ++weightTensorId) {
            ATB_LOG(INFO) << __func__ << " layerId " << layerId << " weightID"<<weightTensorId << " -> in weight ID" << ( (WEIGHT_COUNT_PER_LAYER * param_.vlLayerIndex) + 
                ( layerId - param_.vlLayerIndex ) * WEIGHT_COUNT_PER_VL_LAYER + weightTensorId);
            layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at( 
                (WEIGHT_COUNT_PER_LAYER * param_.vlLayerIndex) + // 23 * 10  + (10 - 10) * 17 + i
                ( layerId - param_.vlLayerIndex ) * WEIGHT_COUNT_PER_VL_LAYER + weightTensorId);
        }

        layerNode.outTensors.resize(layerNode.operation->GetOutputNum());
        if(layerId + 1 == param_.layerNum){//已经到结尾，需要退出
            for (int i=0;i< OUT_TENSOR_MAX;i++ ){
                layerNode.outTensors.at(i) = &graph_.outTensors.at(i);
            }
            // layerNode.outTensors = {&graph_.outTensors.at(0)};
        }else{
            for (int i=0;i< OUT_TENSOR_MAX;i++ ){
                layerNode.outTensors.at(i) = &graph_.internalTensors.at((layerId * OUT_TENSOR_MAX) + i );
            }
            firstInTensor = layerNode.outTensors.at(0);
            // layerNode.outTensors = {&graph_.internalTensors.at(layerId)};
        }

        // firstInTensor = layerNode.outTensors.at(0);
    }
    
    // auto &finalNormNode = graph_.nodes.at(nodeId++);
    // atb::infer::RmsNormParam finalNormParam;
    // finalNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    // finalNormParam.normParam.epsilon = param_.layerNormEps;
    // CREATE_OPERATION(finalNormParam, &op);
    // finalNormNode.operation.reset(op);
    // const int finalLayerNormWeightTensorId =
    //     graph_.weightTensors.size() - FINALNORMNODE_WEIGHT_COUNT - OUT_LM_HEAD_WEIGHT_COUNT;
    // const int finalLayerNormOutTensorId = internalTensorSize - 2;
    // finalNormNode.inTensors = {firstInTensor, &graph_.weightTensors.at(finalLayerNormWeightTensorId)};
    // finalNormNode.outTensors = {&graph_.internalTensors.at(finalLayerNormOutTensorId)};

    // const int hiddenSize = param_.headNum * param_.dk;
    // auto &qPassSliceNode = graph_.nodes.at(nodeId++);
    // atb::infer::SliceParam slicePassParam;
    // slicePassParam.offsets = {0, 0, hiddenSize * param_.rank};
    // slicePassParam.size = {-1, -1, hiddenSize};
    // CREATE_OPERATION(slicePassParam, &op);
    // qPassSliceNode.operation.reset(op);
    // const int qPassSliceNodeOutTensorId = internalTensorSize - 1;
    // qPassSliceNode.inTensors = {&graph_.internalTensors.at(finalLayerNormOutTensorId)};
    // qPassSliceNode.outTensors = {&graph_.internalTensors.at(qPassSliceNodeOutTensorId)};

    // auto &outLinearNode = graph_.nodes.at(nodeId++);
    // atb_speed::common::ParallelParamV2 outLinearParm;
    // outLinearParm.commParam.rank = param_.rank;
    // outLinearParm.commParam.rankSize = param_.rankSize;
    // outLinearParm.isBias = false;
    // outLinearParm.commParam.backend = param_.backend;
    // atb_speed::common::RowParallelLinearV2(outLinearParm, &op);
    // outLinearNode.operation.reset(op);
    // const int finalLinearWeightTensorId = graph_.weightTensors.size() - OUT_LM_HEAD_WEIGHT_COUNT;
    // outLinearNode.inTensors = {&graph_.internalTensors.at(qPassSliceNodeOutTensorId),
    //                            &graph_.weightTensors.at(finalLinearWeightTensorId),
    //                            &graph_.internalTensors.at(IN_HOLDER),
    //                            &graph_.internalTensors.at(IN_HOLDER),
    //                            &graph_.internalTensors.at(IN_HOLDER),
    //                            &graph_.internalTensors.at(IN_HOLDER),
    //                            &graph_.internalTensors.at(IN_HOLDER)};
    // outLinearNode.outTensors = {&graph_.outTensors.at(0)};
    return atb::NO_ERROR;
}

atb::Status FlashAttentionModel::ParseParam(const std::string &param)
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
    return atb::NO_ERROR;
}

atb::Status FlashAttentionModel::BindParamHostTensor(uint32_t nodeId)
{
    if (nodeId >= param_.layerNum) {
        return atb::NO_ERROR;
    }

    
    auto &node = graph_.nodes.at(nodeId);

    const uint32_t tokenOffsetTensorId = 4;
    const uint32_t seqLenTensorId = 5;

    

    node.variantPack.inTensors.at(tokenOffsetTensorId).hostData = tokenOffset_.data();
    node.variantPack.inTensors.at(seqLenTensorId).hostData = seqLen_.data();
    ATB_LOG(INFO) << "BindParamHostTensor end";
    return atb::NO_ERROR;
}
} // namespace vlmo
} // namespace atb_speed