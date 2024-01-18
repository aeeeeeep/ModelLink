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

#include "embedding_layer.h"

namespace atb_speed {
namespace telechat {
enum EmbeddingLayerTensorId {
    IN_EMBEDDING_WEIGHT = 0,
    IN_INPUT_IDS,
    IN_COSTABLE,
    IN_SINTABLE,
    IN_POSITION_IDS,
    OUT_HIDDEN_STATES,
    OUT_COS_EMBED,
    OUT_SIN_EMBED,
};

static const uint64_t IN_TENSOR_COUNT = 5;
static const uint64_t OUT_TENSOR_COUNT = 3;
static const uint64_t INTERNAL_TENSOR_COUNT = 0;
static const uint64_t NODE_COUNT = 3;

atb::Status EmbeddingLayer(const EmbeddingLayerParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERNAL_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    atb::Node &wordEmbeddingNode = opGraph.nodes.at(nodeId++);
    atb::Node &cosEmbeddingNode = opGraph.nodes.at(nodeId++);
    atb::Node &sinEmbeddingNode = opGraph.nodes.at(nodeId++);

    atb::infer::GatherParam gatherParam;
    gatherParam.axis = param.axis;
    CreateOperation(gatherParam, &wordEmbeddingNode.operation);
    wordEmbeddingNode.inTensorIds = {IN_EMBEDDING_WEIGHT, IN_INPUT_IDS};
    wordEmbeddingNode.outTensorIds = {OUT_HIDDEN_STATES};

    CreateOperation(gatherParam, &cosEmbeddingNode.operation);
    cosEmbeddingNode.inTensorIds = {IN_COSTABLE, IN_POSITION_IDS};
    cosEmbeddingNode.outTensorIds = {OUT_COS_EMBED};

    CreateOperation(gatherParam, &sinEmbeddingNode.operation);
    sinEmbeddingNode.inTensorIds = {IN_SINTABLE, IN_POSITION_IDS};
    sinEmbeddingNode.outTensorIds = {OUT_SIN_EMBED};

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        outTensorDescs.at(0).shape.dimNum = 3;
        outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(1).shape.dims[0];
        outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(1).shape.dims[1];
        outTensorDescs.at(0).shape.dims[2] = inTensorDescs.at(0).shape.dims[1];

        outTensorDescs.at(1) = inTensorDescs.at(2);
        outTensorDescs.at(1).shape.dimNum = 3;
        outTensorDescs.at(1).shape.dims[0] = inTensorDescs.at(4).shape.dims[0];
        outTensorDescs.at(1).shape.dims[1] = inTensorDescs.at(4).shape.dims[1];
        outTensorDescs.at(1).shape.dims[2] = inTensorDescs.at(2).shape.dims[1];

        outTensorDescs.at(2) = inTensorDescs.at(2);
        outTensorDescs.at(2).shape.dimNum = 3;
        outTensorDescs.at(2).shape.dims[0] = inTensorDescs.at(4).shape.dims[0];
        outTensorDescs.at(2).shape.dims[1] = inTensorDescs.at(4).shape.dims[1];
        outTensorDescs.at(2).shape.dims[2] = inTensorDescs.at(2).shape.dims[1];
        return atb::NO_ERROR;
    };

    atb::CreateOperation(opGraph, operation);
    return atb::NO_ERROR;
}
}   // namespace telechat
}   // namespace atb_speed