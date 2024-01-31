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
#include "layers/operations/embedding.h"

namespace atb_speed {
namespace common {

enum LayerEmbeddingTensorIdx : uint32_t {
    IN_EMBEDDING_WEIGHTS = 0,
    IN_INPUT_IDS,
    IN_COS_TABLE,
    IN_SIN_TABLE,
    IN_POSITION_IDS,
    OUT_HIDDEN_STATES,
    OUT_COS_EMBED,
    OUT_SIN_EMBED,
    INTERMEDIATE_GATHER,
    INTERMEDIATE_ALLGATHER_OUT_ID,
};

static const uint64_t IN_TENSOR_COUNT = 5;
static const uint64_t OUT_TENSOR_COUNT = 3;
static const uint64_t INTERMEDIATE_TENSOR_NO_ALL_GATHER_COUNT = 0;
static const uint64_t INTERMEDIATE_TENSOR_ALL_GATHER_COUNT = 2;
static const uint64_t NODE_NO_ALL_GATHER_COUNT = 3;
static const uint64_t NODE_ALL_GATHER_COUNT = 5;

atb::Status Embedding(const EmbeddingParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    // 若权重按列切分，则需使用all gather方式收集完整的hidden states
    // 相比不使用all gather会多两个internalTensor和两个node
    opGraph.internalTensorNum \
        = param.worldSize > 1 ? INTERMEDIATE_TENSOR_ALL_GATHER_COUNT : INTERMEDIATE_TENSOR_NO_ALL_GATHER_COUNT;
    opGraph.nodes.resize(param.worldSize > 1 ? NODE_ALL_GATHER_COUNT : NODE_NO_ALL_GATHER_COUNT);
    opGraph.name = "ParallelEmbedding";

    size_t nodeId = 0;
    auto &inputIdEmbeddingNode = opGraph.nodes.at(nodeId++);
    atb::infer::GatherParam inputembedinggatherparam;
    inputembedinggatherparam.axis = param.axis;
    CREATE_OPERATION(inputembedinggatherparam, &inputIdEmbeddingNode.operation);
    inputIdEmbeddingNode.inTensorIds = {
        LayerEmbeddingTensorIdx::IN_EMBEDDING_WEIGHTS, LayerEmbeddingTensorIdx::IN_INPUT_IDS
    };
    inputIdEmbeddingNode.outTensorIds = {
        param.worldSize > 1 ? LayerEmbeddingTensorIdx::INTERMEDIATE_GATHER : LayerEmbeddingTensorIdx::OUT_HIDDEN_STATES
    };

    if (param.worldSize > 1) {
        auto &allGatherNode = opGraph.nodes[nodeId++];
        atb::infer::AllGatherParam allGatherParam;
        allGatherParam.rank = param.rank;
        allGatherParam.rankSize = param.worldSize;
        allGatherParam.rankRoot = param.rankRoot;
        allGatherParam.backend = param.backend;
        CREATE_OPERATION(allGatherParam, &allGatherNode.operation);
        allGatherNode.inTensorIds = {LayerEmbeddingTensorIdx::INTERMEDIATE_GATHER};
        allGatherNode.outTensorIds = {LayerEmbeddingTensorIdx::INTERMEDIATE_ALLGATHER_OUT_ID};
    
        auto &transposeNode = opGraph.nodes[nodeId++];
        atb::infer::TransposeParam transposeParam;
        if (param.unpadInputs) {
            transposeParam.perm = {1, 0, 2};
        } else {
            transposeParam.perm = {1, 2, 0, 3};
        }
        CREATE_OPERATION(transposeParam, &transposeNode.operation);
        transposeNode.inTensorIds = {LayerEmbeddingTensorIdx::INTERMEDIATE_ALLGATHER_OUT_ID};
        transposeNode.outTensorIds = {LayerEmbeddingTensorIdx::OUT_HIDDEN_STATES};
    }

    auto &cosEmbeddingNode = opGraph.nodes.at(nodeId++);
    atb::infer::GatherParam cosEmbeddingGatherParam;
    cosEmbeddingGatherParam.axis = param.axis;
    CREATE_OPERATION(cosEmbeddingGatherParam, &cosEmbeddingNode.operation);
    cosEmbeddingNode.inTensorIds = {LayerEmbeddingTensorIdx::IN_COS_TABLE, LayerEmbeddingTensorIdx::IN_POSITION_IDS};
    cosEmbeddingNode.outTensorIds = {LayerEmbeddingTensorIdx::OUT_COS_EMBED};

    auto &sinEmbeddingNode = opGraph.nodes.at(nodeId++);
    atb::infer::GatherParam sinEmbeddingGatherParam;
    sinEmbeddingGatherParam.axis = param.axis;
    CREATE_OPERATION(sinEmbeddingGatherParam, &sinEmbeddingNode.operation);
    sinEmbeddingNode.inTensorIds = {LayerEmbeddingTensorIdx::IN_SIN_TABLE, LayerEmbeddingTensorIdx::IN_POSITION_IDS};
    sinEmbeddingNode.outTensorIds = {LayerEmbeddingTensorIdx::OUT_SIN_EMBED};

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
        outTensorDescs.at(0).format = inTensorDescs.at(0).format;
        if (param.unpadInputs) {
            outTensorDescs.at(0).shape.dimNum = 2;
            outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(1).shape.dims[0];
            outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1] * param.worldSize;
        } else {
            outTensorDescs.at(0).shape.dimNum = 3;
            outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(1).shape.dims[0];
            outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(1).shape.dims[1];
            outTensorDescs.at(0).shape.dims[2] = inTensorDescs.at(0).shape.dims[1] * param.worldSize;
        }

        outTensorDescs.at(1) = inTensorDescs.at(2);
        outTensorDescs.at(1).shape.dimNum = 2;
        if (param.unpadInputs) {
            outTensorDescs.at(1).shape.dims[0] = inTensorDescs.at(4).shape.dims[0];
        } else {
            outTensorDescs.at(1).shape.dims[0] = inTensorDescs.at(4).shape.dims[0] * inTensorDescs.at(4).shape.dims[1];
            outTensorDescs.at(1).shape.dims[1] = inTensorDescs.at(2).shape.dims[1];
        }
        outTensorDescs.at(1).shape.dims[1] = inTensorDescs.at(2).shape.dims[1];

        outTensorDescs.at(2) = outTensorDescs.at(1);

        return atb::NO_ERROR;
    };

    return CREATE_OPERATION(opGraph, operation);
}
}  // namespace common
}  // namespace atb_speed