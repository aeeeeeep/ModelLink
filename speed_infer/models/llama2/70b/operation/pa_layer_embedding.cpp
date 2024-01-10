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
#include "pa_layer_embedding.h"

namespace atb_speed {
namespace llama2_70b {
enum PALayerEmbeddingTensorId : int {
    IN_EMBEDDING_WEIGHTS = 0,
    IN_INPUT_IDS,
    IN_COS_TABLE,
    IN_SIN_TABLE,
    IN_POSITION_IDS,
    OUT_HIDDEN_STATES,
    OUT_COS_EMBED,
    OUT_SIN_EMBED
};

static const uint64_t IN_TENSOR_COUNT = 5;
static const uint64_t OUT_TENSOR_COUNT = 3;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 0;
static const uint64_t NODE_COUNT = 3;

atb::Status PALayerEmbedding(const PALayerEmbeddingParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);
    opGraph.name = "PALayerEmbedding";

    size_t nodeId = 0;
    auto &inputIdEmbeddingNode = opGraph.nodes.at(nodeId++);
    auto &cosEmbeddingNode = opGraph.nodes.at(nodeId++);
    auto &sinEmbeddingNode = opGraph.nodes.at(nodeId++);

    atb::infer::GatherParam inputembedinggatherparam;
    inputembedinggatherparam.axis = param.axis;
    CreateOperation(inputembedinggatherparam, &inputIdEmbeddingNode.operation);
    inputIdEmbeddingNode.inTensorIds = {IN_EMBEDDING_WEIGHTS, IN_INPUT_IDS};
    inputIdEmbeddingNode.outTensorIds = {OUT_HIDDEN_STATES};

    atb::infer::GatherParam cosEmbeddingGatherParam;
    cosEmbeddingGatherParam.axis = param.axis;
    CreateOperation(cosEmbeddingGatherParam, &cosEmbeddingNode.operation);
    cosEmbeddingNode.inTensorIds = {IN_COS_TABLE, IN_POSITION_IDS};
    cosEmbeddingNode.outTensorIds = {OUT_COS_EMBED};

    atb::infer::GatherParam sinEmbeddingGatherParam;
    sinEmbeddingGatherParam.axis = param.axis;
    CreateOperation(sinEmbeddingGatherParam, &sinEmbeddingNode.operation);
    sinEmbeddingNode.inTensorIds = {IN_SIN_TABLE, IN_POSITION_IDS};
    sinEmbeddingNode.outTensorIds = {OUT_SIN_EMBED};

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        outTensorDescs.at(0).shape.dimNum = 2; // dimNum is 2
        outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(1).shape.dims[0];
        outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1];

        outTensorDescs.at(1) = inTensorDescs.at(2); // dimNum is 2
        outTensorDescs.at(1).shape.dimNum = 2; // dimNum is 2
        outTensorDescs.at(1).shape.dims[0] = inTensorDescs.at(4).shape.dims[0]; // dimNum is 4
        outTensorDescs.at(1).shape.dims[1] = inTensorDescs.at(2).shape.dims[1]; // dimNum is 2

        outTensorDescs.at(2) = outTensorDescs.at(1); // dimNum is 2
        return atb::NO_ERROR;
    };

    atb::CreateOperation(opGraph, operation);
    return atb::NO_ERROR;
}
} // namespace llama2_70b
} // namespace atb_speed
