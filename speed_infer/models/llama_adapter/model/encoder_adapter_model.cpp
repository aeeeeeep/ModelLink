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
#include "models/llama_adapter/layer/layer.h"
#include "adapter_model.h"

namespace atb_speed {
namespace llama_adapter {
const int WEIGHT_COUNT_PER_LAYER = 15;

enum InTensorId : int {
    IN_TENSOR_HIDDENSTATES = 0,
    IN_TENSOR_FREQCIS,
    IN_TENSOR_ATTENTIONMASK,
    IN_TENSOR_ADAPTER,
};

enum OutTensorId : int {
    OUT_TENSOR_HIDDENSTATES = 0,
    OUT_TENSOR_MAX,
};

void EncoderAdapterModel::Param::FromString(const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);
    rmsNormEps = paramJson["rmsNormEps"].get<double>();
    headNum = paramJson["headNum"].get<int>();
    dk = paramJson["dk"].get<int>();
    layerNum = paramJson["layerNum"].get<int>();
    ATB_LOG(INFO) << "Llama7BEncoderAdapterModel param rmsNormEps:" << rmsNormEps << ", headNum:" << headNum
                  << ", dk:" << dk << ", layerNum:" << layerNum;
}

EncoderAdapterModel::EncoderAdapterModel(const std::string &param) : Model("EncoderAdapterModel", param)
{
    param_.FromString(param);
}

EncoderAdapterModel::~EncoderAdapterModel() = default;

uint32_t EncoderAdapterModel::GetInputNum() const { return graph_.inTensors.size(); }

uint32_t EncoderAdapterModel::GetOutputNum() const { return graph_.outTensors.size(); }

atb::Status EncoderAdapterModel::InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
                                            std::vector<atb::TensorDesc> &outTensorDescs)
{
    ATB_LOG(INFO) << "Enter EncoderAdapterModel InferShape";
    if (outTensorDescs.size() != GetOutputNum()) {
        return atb::ERROR_INVALID_GRAPH;
    }

    const atb::TensorDesc &keyTensorDesc = inTensorDescs.at(0);

    outTensorDescs.at(0) = inTensorDescs.at(0);
    outTensorDescs.at(1) = keyTensorDesc;
    outTensorDescs.at(1).shape.dimNum += 1;
    outTensorDescs.at(1).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
    outTensorDescs.at(1).shape.dims[1] = inTensorDescs.at(0).shape.dims[1];
    outTensorDescs.at(1).shape.dims[2] = param_.headNum;
    outTensorDescs.at(1).shape.dims[3] = param_.dk;

    ATB_LOG(INFO) << "EncoderAdapterModel InferShape Looping";
    for (size_t i = 2; i < outTensorDescs.size(); ++i) {
        outTensorDescs.at(i) = outTensorDescs.at(1);
    }

    return atb::NO_ERROR;
}

int64_t EncoderAdapterModel::BuildGraph()
{
    ATB_LOG(INFO) << "Enter EncoderAdapterModel BuildGraph";
    const int weightTensorSize = WEIGHT_COUNT_PER_LAYER * param_.layerNum;
    graph_.weightTensors.resize(weightTensorSize);

    graph_.inTensors.resize(IN_TENSOR_ADAPTER + param_.layerNum - 1);
    graph_.outTensors.resize(OUT_TENSOR_MAX + 2 * param_.layerNum);

    const int nodeSize = param_.layerNum;
    ATB_LOG(INFO) << "Llama_Adapter_EncoderAdapterModel nodeSize is " << nodeSize;
    graph_.nodes.resize(nodeSize);

    graph_.internalTensors.resize(graph_.nodes.size());

    int nodeId = 0;

    atb::Operation *op = nullptr;
    atb::Tensor *firstInTensor = &graph_.inTensors.at(0);

    for (int layerId = 0; layerId < param_.layerNum; ++layerId) {
        auto &layerNode = graph_.nodes.at(nodeId++);

        atb_speed::llama_adapter::LayerParam modelParam;
        modelParam.rmsNormEps = param_.rmsNormEps;
        modelParam.headNum = param_.headNum;
        modelParam.dk = param_.dk;
        modelParam.model = "llama_adapter";

        if (modelParam.headNum == 0) {
            ATB_LOG(INFO) << "headNum can not be zero,but here modelParam.headNum is " << modelParam.headNum;
            return atb::ERROR_INVALID_PARAM;
        }

        if (layerId == 0) {
            atb_speed::llama_adapter::EncoderLayer(modelParam, &op);
            layerNode.operation.reset(op);
            layerNode.inTensors.resize(layerNode.operation->GetInputNum());
            size_t inTensorId = 0;
            layerNode.inTensors.at(inTensorId++) = firstInTensor; // hidden states
            for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER - 1; ++weightTensorId) {
                if (weightTensorId < 5) {
                    layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(weightTensorId);
                } else {
                    layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(weightTensorId + 1);
                }
            }
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_FREQCIS);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ATTENTIONMASK);
            
            layerNode.outTensors = {&graph_.internalTensors.at(layerId),
                                    &graph_.outTensors.at(OUT_TENSOR_MAX + layerId),
                                    &graph_.outTensors.at(OUT_TENSOR_MAX + layerId + param_.layerNum)};
        } else {
            atb_speed::llama_adapter::EncoderAdapterLayer(modelParam, &op);
            layerNode.operation.reset(op);
            layerNode.inTensors.resize(layerNode.operation->GetInputNum());
            size_t inTensorId = 0;
            layerNode.inTensors.at(inTensorId++) = firstInTensor; // hidden states
            for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
                layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(
                    layerId * WEIGHT_COUNT_PER_LAYER + weightTensorId);
            }
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_FREQCIS);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ADAPTER + layerId - 1);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ATTENTIONMASK);

            layerNode.outTensors = {&graph_.internalTensors.at(layerId),
                                    &graph_.outTensors.at(OUT_TENSOR_MAX + layerId),
                                    &graph_.outTensors.at(OUT_TENSOR_MAX + layerId + param_.layerNum)};
        }

        firstInTensor = layerNode.outTensors.at(0);

        if (layerId == param_.layerNum - 1) {
            layerNode.outTensors.at(0) = {&graph_.outTensors.at(0)};
        }
    }
    return atb::NO_ERROR;
}
} // namespace llama_adapter
} // namespace atb_speed