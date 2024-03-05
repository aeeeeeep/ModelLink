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
#include "topktopp.h"
#include <iostream>
using namespace std;
namespace atb_speed {
namespace layers {
enum TopktoppTensorId {
    probsTensor,
    pTensor,
    temeraturetensor,
    probsSampledTensor,
    indicesSampledTensor,
    probsmultinomialedTensor,
    indicesSortedTensor,
    probsSortedNormedFilteredTensor,
    probssoftmaxslicededTensor,
    probscumsumedTensor,
    probsgreatedTensor,
    probsSortededTensor,
    probssoftmaxdedTensor,
    probsdivededTensor,
    probsSortedNormedFilteredsoftmaxedTensor,

};

static const uint64_t IN_TENSOR_COUNT = 3;
static const uint64_t OUT_TENSOR_COUNT = 2;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 10;
static const uint64_t NODE_COUNT = 12;

atb::Status Topktopp(const TopktoppParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);
    opGraph.name = "topktopp";

    size_t nodeId = 0;
    auto &realdivNode = opGraph.nodes.at(nodeId++);
    auto &sortNode = opGraph.nodes.at(nodeId++);
    auto &softmaxNode = opGraph.nodes.at(nodeId++);
    auto &cumsumNode = opGraph.nodes.at(nodeId++);
    auto &greaterNode = opGraph.nodes.at(nodeId++);
    auto &fillMaskNode = opGraph.nodes.at(nodeId++);
    auto &slice2Node = opGraph.nodes.at(nodeId++);
    auto &setvalueNode = opGraph.nodes.at(nodeId++);
    auto &softmax2Node = opGraph.nodes.at(nodeId++);
    auto &multinomialNode = opGraph.nodes.at(nodeId++);
    auto &gatherprobsNode = opGraph.nodes.at(nodeId++);
    auto &gatherindicesNode = opGraph.nodes.at(nodeId++);


    atb::infer::ElewiseParam realdivParam;
    realdivParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_REALDIV;
    CreateOperation(realdivParam, &realdivNode.operation);
    realdivNode.inTensorIds = {probsTensor,temeraturetensor};
    realdivNode.outTensorIds = {probsdivededTensor};

    atb::infer::SortParam sortParam;
    sortParam.num = {param.topk};
    CreateOperation(sortParam, &sortNode.operation);
    sortNode.inTensorIds = {probsdivededTensor};
    sortNode.outTensorIds = {probsSortededTensor,indicesSortedTensor};

    atb::infer::SoftmaxParam softmaxParam;
    softmaxParam.axes = {-1};
    atb::CreateOperation(softmaxParam, &softmaxNode.operation);
    softmaxNode.inTensorIds = { probsSortededTensor };
    softmaxNode.outTensorIds = { probssoftmaxdedTensor };

    atb::infer::CumsumParam cumsumParam;
    cumsumParam.axes = {param.axes};
    cumsumParam.exclusive = false;
    cumsumParam.reverse = false;
    atb::CreateOperation(cumsumParam, &cumsumNode.operation);
    cumsumNode.inTensorIds = { probssoftmaxdedTensor };
    cumsumNode.outTensorIds = { probscumsumedTensor };

    atb::infer::ElewiseParam greaterParam;
    greaterParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_GREATER;
    CreateOperation(greaterParam, &greaterNode.operation);
    greaterNode.inTensorIds = {probscumsumedTensor, pTensor};
    greaterNode.outTensorIds = {probsgreatedTensor};

    atb::infer::FillParam fillParam;
    fillParam.value = {param.filter_value};
    CreateOperation(fillParam, &fillMaskNode.operation);
    fillMaskNode.inTensorIds = {probsSortededTensor, probsgreatedTensor};
    fillMaskNode.outTensorIds = {probsSortedNormedFilteredTensor};
    
    atb::infer::SliceParam slice2Param;
    slice2Param.offsets = {0, 0};
    slice2Param.size = {-1, param.min_tokens_to_keep};
    CreateOperation(slice2Param, &slice2Node.operation);
    slice2Node.inTensorIds = {probsSortededTensor};
    slice2Node.outTensorIds = {probssoftmaxslicededTensor};

    atb::infer::SetValueParam setvalueParam;
    setvalueParam.starts = {0,0};
    setvalueParam.ends = {param.row,param.min_tokens_to_keep};
    setvalueParam.strides = {1,1};
    CreateOperation(setvalueParam, &setvalueNode.operation);
    setvalueNode.inTensorIds = {probsSortedNormedFilteredTensor, probssoftmaxslicededTensor};

    atb::infer::SoftmaxParam softmax2Param;
    softmax2Param.axes = {-1};
    atb::CreateOperation(softmax2Param, &softmax2Node.operation);
    softmax2Node.inTensorIds = { probsSortedNormedFilteredTensor };
    softmax2Node.outTensorIds = { probsSortedNormedFilteredsoftmaxedTensor };

    atb::infer::MultinomialParam multinomialParam;
    multinomialParam.numSamples = 1;
    multinomialParam.randSeed = param.randseed;
    atb::CreateOperation(multinomialParam, &multinomialNode.operation);
    multinomialNode.inTensorIds = { probsSortedNormedFilteredsoftmaxedTensor };
    multinomialNode.outTensorIds = { probsmultinomialedTensor };

    atb::infer::GatherParam gatherprobsParam;
    gatherprobsParam.axis = 1;
    gatherprobsParam.batchDims = 1;
    CreateOperation(gatherprobsParam, &gatherprobsNode.operation);
    gatherprobsNode.inTensorIds = {probsSortededTensor, probsmultinomialedTensor};
    gatherprobsNode.outTensorIds = {probsSampledTensor};

    atb::infer::GatherParam gatherindicesParam;
    gatherindicesParam.axis = 1;
    gatherindicesParam.batchDims = 1;
    CreateOperation(gatherindicesParam, &gatherindicesNode.operation);
    gatherindicesNode.inTensorIds = {indicesSortedTensor, probsmultinomialedTensor};
    gatherindicesNode.outTensorIds = {indicesSampledTensor};


    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        outTensorDescs.at(0).shape.dimNum = 1;
        outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
        // outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[0];

        outTensorDescs.at(1) = inTensorDescs.at(0);
        outTensorDescs.at(1).dtype=ACL_INT32;
        outTensorDescs.at(1).shape.dimNum = 1;
        outTensorDescs.at(1).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
        // outTensorDescs.at(1).shape.dims[1] = inTensorDescs.at(0).shape.dims[0];

        return atb::NO_ERROR;
    };

    atb::CreateOperation(opGraph, operation);
    return atb::NO_ERROR;
}
} 
} 