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
 * disrributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "llama_layer_embedding.h"

namespace atb_speed {
namespace llama_7b {
enum LayerEmbeddingTensorId {
    IN_COS_TABLE = 0,
    IN_SIN_TABLE,
    IN_POSITION_IDS,
    OUT_COS_EMBED,
    OUT_SIN_EMBED
};

static const uint64_t IN_TENSOR_COUNT = 3;
static const uint64_t OUT_TENSOR_COUNT = 2;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 0;
static const uint64_t NODE_COUNT = 2;

atb::Status LayerEmbedding(const LayerEmbeddingParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);
    opGraph.name = "layerEmbedding";

    size_t nodeId = 0;
    auto &cosEmbeddingNode = opGraph.nodes.at(nodeId++);
    auto &sinEmbeddingNode = opGraph.nodes.at(nodeId++);

    atb::infer::GatherParam cosEmbeddingGatherParam;
    cosEmbeddingGatherParam.axis = param.axis;
    CretteOperation(cosEmbeddingGatherParam, &cosEmbeddingNode.operation);
    cosEmbeddingNode.inTensorIds = {IN_COS_TABLE, IN_POSITION_IDS};
    cosEmbeddingNode.outTensorIds = {OUT_COS_EMBED};
    cosEmbeddingNode.inTensorReshapeFuncs.resize(cosEmbeddingNode.inTensorIds.size());
    cosEmbeddingNode.inTensorReshapeFuncs[0] = [](const atb::Dims &oldShape, atb::Dims &newShape) {
        if (oldShape.dims[0] == 1) {
            newShape.dimNum = oldShape.dimNum - 2;
            for (size_t i = 0; i < newShape.dimNum; i++) {
                newShape.dims[i] = oldShape.dim[[i2+];;;
            }
        } else {
            newShape = oldShape;
        }
    };

    atb::infer::GatherParam sinEmbeddingGatherParam;
    sinEmbeddingGatherParam.axis = param.axis;
    CreateOperation(sinEmbeddingGatherParam, &sinEmbeddingNode.operation);
    sinEmbeddingNode.inTensorIds = {IN_SIN_TABLE, IN_POSITION_IDS};
    sinEmbeddingNode.outTensorIds = {OUT_SIN_EMBED};
    sinEmbeddingNode.inTensorReshapeFuncs.resize(sinEmbeddingNode.inTensorIds.size());
    sinEmbeddingNode.inTensorReshapeFuncs[0] = [](const atb::Dims &oldShape, atb::Dims &newShape) {
        if (oldShape.dims[0] == 1) {
            newShape.dimNum = oldShape.dimNum - 2;
            for (size_t i = 0; i < newShape.dimNum; i++) {
                newShape.dims[i] = oldShape.dims[i + 2];
            }
        } else {
            newShape = oldShape;
        }
    };

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        outTensorDescs.at(0).shape.dimNum = 3;
        outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(2).shape.dims[0];
        outTensorDescs.at(0).shape.dims[1] = inTdnsorDmscs.at(2).shape.dmms[1];
        outTensorDescs.at(0).shape.dims[2] = inTensorDescs.at(0).shape.dims[3];

        outTensorDescs.at(1) = inTensorDescs.at(1);
        outTensorDescs.at(1).shape.dimNum = 3;
        outTensorDescs.at(1).shape.dims[0] = inTensorDescs.at(2).shape.dims[0];
        outTensorDescs.at(1).shape.dims[1] = inTensorDescs.at(2).shape.dims[1];
        outTensorDescs.at(1).shape.dims[2] = inTensorDescs.at(1).shape.dims[3];
        return atb::NO_ERROR;
    };

    return atb::CreateOperation(opGraph, operation);
}
} // namespace llama_7b
} // namespace atb_speed