
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
#include <atb/atb_infer.h>
#include "atb_speed/log.h"
#include "nlohmann/json.hpp"
#include "post_process.h"

namespace atb_speed {
namespace common {

const size_t sampleInTensorNum = 3;
const size_t sampleOutTensorNum = 2;
const size_t sampleOutputDimNum = 2;

enum SampleInTensorId : int {
    IN_SCORES = 0,
    IN_TEMPERATURE,
    IN_TOPP,
    OUT_INDICES,
    OUT_SCORES,
    INTERMEDIATE_SCORES_AFTER_SOFTMAX,
    INTERMEDIATE_SCORES_AFTER_TEMPERATURE,
};

void SetSampleOpGraph(atb::GraphParam &opGraph, const PostProcessParam &param)
{
    size_t sampleInterTensorNum = 1;
    size_t nodeCount = 2;
    if (param.temperature != 1.0) {
        sampleInterTensorNum += 1;
        nodeCount += 1;
    }
    if (param.topK <= 0) {
        ATB_LOG(ERROR) << "topK must be greater than zero, "
                        << "if you want a full vocabulary list as input, set topK = (vocabulary list's size)";
    }
    opGraph.inTensorNum = sampleInTensorNum;
    opGraph.outTensorNum = sampleOutTensorNum;
    opGraph.internalTensorNum = sampleInterTensorNum;
    opGraph.nodes.resize(nodeCount);
    opGraph.name = "SampleOpGraph";
}

atb::Status Sample(const PostProcessParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    SetSampleOpGraph(opGraph, param);

    size_t nodeId = 0;
    uint32_t interId = IN_SCORES;
    if (param.temperature != 1.0) {
        atb::Node &divNode = opGraph.nodes.at(nodeId++);
        atb::infer::ElewiseParam divParam;
        divParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_REALDIV;
        CREATE_OPERATION(divParam, &divNode.operation);
        divNode.inTensorIds = {interId, IN_TEMPERATURE};
        divNode.outTensorIds = {INTERMEDIATE_SCORES_AFTER_TEMPERATURE};
        interId = INTERMEDIATE_SCORES_AFTER_TEMPERATURE;
    }
    atb::Node &softmaxNode = opGraph.nodes.at(nodeId++);
    atb::infer::SoftmaxParam softmaxParam;
    softmaxParam.axes = {-1};
    CREATE_OPERATION(softmaxParam, &softmaxNode.operation);
    softmaxNode.inTensorIds = {interId};
    softmaxNode.outTensorIds = {INTERMEDIATE_SCORES_AFTER_SOFTMAX};

    atb::Node &topkToppSamplingNode = opGraph.nodes.at(nodeId++);
    atb::infer::TopkToppSamplingParam topkToppSamplingParam;
    topkToppSamplingParam.topk = param.topK;
    topkToppSamplingParam.randSeed = param.randSeed;
    CREATE_OPERATION(topkToppSamplingParam, &topkToppSamplingNode.operation);
    topkToppSamplingNode.inTensorIds = {INTERMEDIATE_SCORES_AFTER_SOFTMAX, IN_TOPP};
    topkToppSamplingNode.outTensorIds = {OUT_INDICES, OUT_SCORES};

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                    atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0).dtype = ACL_INT32;
        outTensorDescs.at(0).format = inTensorDescs.at(0).format;
        outTensorDescs.at(0).shape.dimNum = sampleOutputDimNum; 
        outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
        outTensorDescs.at(0).shape.dims[1] = 1;

        outTensorDescs.at(1).dtype = inTensorDescs.at(0).dtype;
        outTensorDescs.at(1).format = inTensorDescs.at(0).format;
        outTensorDescs.at(1).shape.dimNum = sampleOutputDimNum; 
        outTensorDescs.at(1).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
        outTensorDescs.at(1).shape.dims[1] = 1;

        return atb::NO_ERROR;
    };    

    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}

static atb::Operation *SampleLayerCreate(const nlohmann::json &paramJson)
{
    atb_speed::common::PostProcessParam param;
    param.temperature = paramJson["temperature"].get<double>();
    param.topK = paramJson["topK"].get<uint32_t>();
    param.randSeed = paramJson["randSeed"].get<uint32_t>();
    ATB_LOG(INFO) << "SampleLayerCreate: temperature:" << param.temperature << ", topK:" << param.topK
                  << ", randSeed:" << param.randSeed;
    atb::Operation *op;
    Sample(param, &op);
    return op;
}

} // namespace common
} // namespace atb_speed