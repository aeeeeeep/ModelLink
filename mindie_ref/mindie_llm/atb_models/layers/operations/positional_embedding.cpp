/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
#include "layers/operations/positional_embedding.h"

namespace atb_speed {
namespace common {

enum PositionalEmbeddingTensorIdx : uint32_t {
    IN_POSITION_IDS = 0,
    IN_COS_TABLE,
    IN_SIN_TABLE,
    OUT_COS_EMBEDDING,
    OUT_SIN_EMBEDDING,
};

static const uint64_t IN_TENSOR_COUNT = 3;
static const uint64_t OUT_TENSOR_COUNT = 2;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 0;
static const uint64_t NODE_COUNT = 2;

atb::Status PositionalEmbedding(const PositionalEmbeddingParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);
    opGraph.name = "PositionalEmbedding";

    size_t nodeId = 0;

    auto &cosEmbeddingNode = opGraph.nodes.at(nodeId++);
    atb::infer::GatherParam cosEmbeddingGatherParam;
    CREATE_OPERATION(cosEmbeddingGatherParam, &cosEmbeddingNode.operation);
    cosEmbeddingNode.inTensorIds = {
        PositionalEmbeddingTensorIdx::IN_COS_TABLE, PositionalEmbeddingTensorIdx::IN_POSITION_IDS
    };
    cosEmbeddingNode.outTensorIds = {PositionalEmbeddingTensorIdx::OUT_COS_EMBEDDING};

    auto &sinEmbeddingNode = opGraph.nodes.at(nodeId++);
    atb::infer::GatherParam sinEmbeddingGatherParam;
    CREATE_OPERATION(sinEmbeddingGatherParam, &sinEmbeddingNode.operation);
    sinEmbeddingNode.inTensorIds = {
        PositionalEmbeddingTensorIdx::IN_SIN_TABLE, PositionalEmbeddingTensorIdx::IN_POSITION_IDS
    };
    sinEmbeddingNode.outTensorIds = {PositionalEmbeddingTensorIdx::OUT_SIN_EMBEDDING};

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(1);
        outTensorDescs.at(0).shape.dimNum = 2;
        outTensorDescs.at(0).shape.dims[0] = 1;
        for (uint64_t i = 0; i < inTensorDescs.at(0).shape.dimNum; i++) {
            outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[i];
        }

        outTensorDescs.at(1) = outTensorDescs.at(0);
        return atb::NO_ERROR;
    };

    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}
}  // namespace common
}  // namespace atb_speed