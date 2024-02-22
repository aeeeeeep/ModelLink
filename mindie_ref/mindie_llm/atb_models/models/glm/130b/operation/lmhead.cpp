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
#include "lmhead.h"
#include <atb/atb_infer.h>
#include "atb_speed/log.h"

namespace atb_speed {
namespace glm130b {
enum LmHeadTensorId : int {
    IN_LOGITS_ID = 0,
    IN_LINEARWEIGHT_ID,
    OUT_LMHEAD_OUT_ID,
    INTERMIDATE_LINEAR_OUT_ID,
    INTERMEDIATE_ALLGATHER_OUT_ID
};

static const uint64_t IN_TENSOR_COUNT = 2;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 2;
static const uint64_t NODE_COUNT = 3;
static uint64_t DIM3 = 3;

atb::Status CreateLmHead(const LmHeadParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    atb::Node &linearNode = opGraph.nodes.at(nodeId++);
    atb::Node &allGatherNode = opGraph.nodes.at(nodeId++);
    atb::Node &transposeNode = opGraph.nodes.at(nodeId++);

    atb::infer::LinearParam linearParam;
    linearParam.hasBias = false;
    CreateOperation(linearParam, &linearNode.operation);
    linearNode.inTensorIds = {IN_LOGITS_ID, IN_LINEARWEIGHT_ID};
    linearNode.outTensorIds = {INTERMIDATE_LINEAR_OUT_ID}; // [bs, seqlen, num_embeddings/rankSize]

    atb::infer::AllGatherParam allGatherParam;
    allGatherParam.rank = param.rank;
    allGatherParam.rankSize = param.rankSize;
    allGatherParam.rankRoot = param.rankRoot;
    allGatherParam.backend = param.backend;
    CreateOperation(allGatherParam, &allGatherNode.operation);
    allGatherNode.inTensorIds = {INTERMIDATE_LINEAR_OUT_ID};
    allGatherNode.outTensorIds = {INTERMEDIATE_ALLGATHER_OUT_ID}; // [rankSize, bs, seqlen, num_embeddings/rankSize]

    atb::infer::TransposeParam transposeParam;
    transposeParam.perm = param.perm; // [1, 2, 0, 3]
    CreateOperation(transposeParam, &transposeNode.operation);
    transposeNode.inTensorIds = {INTERMEDIATE_ALLGATHER_OUT_ID}; // [rankSize, bs, seqlen, num_embeddings/rankSize]
    transposeNode.outTensorIds = {OUT_LMHEAD_OUT_ID};            // [bs, seqlen, rankSize, num_embeddings/rankSize]

    opGraph.inferShapeFunc = [&](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        // inTensor0: [bs, seqlen, hidden_size], inTensor1: [num_embeddings/rankSize, hidden_size]
        // outTensor: [bs, seqlen, num_embeddings]
        outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
        outTensorDescs.at(0).format = inTensorDescs.at(0).format;
        outTensorDescs.at(0).shape.dimNum = DIM3;
        outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
        outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1];
        outTensorDescs.at(0).shape.dims[2] = // 2: 设置第一个张量第三维长度
            inTensorDescs.at(1).shape.dims[0] * param.rankSize;
        return atb::NO_ERROR;
    };

    atb::CreateOperation(opGraph, operation);
    return atb::NO_ERROR;
}
} // namespace glm130b
} // namespace atb_speed