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
namespace gptneox_20b {
enum LayerTensorId : int {
    IN_EMBEDDING_WEIGHTS = 0,
    IN_INPUT_IDS,
    IN_COS_TABLE,
    IN_SIN_TABLE,
    IN_POSITION_IDS,
    OUT_HIDDEN_STATES,
    OUT_COS_EMBED,
    OUT_SIN_EMBED,
};

static const uint64_t IN_TENSOR_COUNT = 5;
static const uint64_t OUT_TENSOR_COUNT = 3;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 0;
static const uint64_t NODE_COUNT = 3;
static const uint64_t OUT_TENSOR_DIM_NUM = 3;

atb::Status EmbeddingLayer(const EmbeddingLayerParam &param, atb::Operation **operation)
{
    ATB_LOG(INFO) << __func__ << " called, axis: " << param.axis;
    atb::GraphParam opGraph;
    opGraph.name = "EmbeddingLayer";
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    atb::Node &inputIdEmbeddingNode = opGraph.nodes.at(nodeId++);
    atb::Node &cosEmbeddingNode = opGraph.nodes.at(nodeId++);
    atb::Node &sinEmbeddingNode = opGraph.nodes.at(nodeId++);

    // params
    atb::infer::GatherParam gatherParam;
    gatherParam.axis = param.axis;

    CREATE_OPERATION(gatherParam, &inputIdEmbeddingNode.operation);
    inputIdEmbeddingNode.inTensorIds = { IN_EMBEDDING_WEIGHTS, IN_INPUT_IDS };
    inputIdEmbeddingNode.outTensorIds = { OUT_HIDDEN_STATES };

    CREATE_OPERATION(gatherParam, &cosEmbeddingNode.operation);
    cosEmbeddingNode.inTensorIds = { IN_COS_TABLE, IN_POSITION_IDS };
    cosEmbeddingNode.outTensorIds = { OUT_COS_EMBED };

    CREATE_OPERATION(gatherParam, &sinEmbeddingNode.operation);
    sinEmbeddingNode.inTensorIds = { IN_SIN_TABLE, IN_POSITION_IDS };
    sinEmbeddingNode.outTensorIds = { OUT_SIN_EMBED };

    opGraph.inferShapeFunc = [](const atb::SVector<atb::TensorDesc> &inTensorDescs,
        atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        outTensorDescs.at(0).shape.dimNum = OUT_TENSOR_DIM_NUM;
        // [batch_size, seq_len, hidden_size]
        outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(1).shape.dims[0];
        outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(1).shape.dims[1];
        outTensorDescs.at(0).shape.dims[2] = inTensorDescs.at(0).shape.dims[1];

        outTensorDescs.at(1) = inTensorDescs.at(2);
        // [batch_size, seq_len, rotary_dims]
        outTensorDescs.at(1).shape.dimNum = OUT_TENSOR_DIM_NUM;
        outTensorDescs.at(1).shape.dims[0] = inTensorDescs.at(4).shape.dims[0];
        outTensorDescs.at(1).shape.dims[1] = inTensorDescs.at(4).shape.dims[1];
        outTensorDescs.at(1).shape.dims[2] = inTensorDescs.at(2).shape.dims[1];

        outTensorDescs.at(2) = outTensorDescs.at(1); // [batch_size, seq_len, rotary_dims]
        return atb::NO_ERROR;
    };

    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}

atb::Operation *CreateFlashAttentionKvCacheLayer(const nlohmann::json &paramJson)
{
    LayerParam param;
    param.layerNormEps = paramJson["layerNormEps"].get<float>();
    param.headNum = paramJson["headNum"].get<int>();
    param.dk = paramJson["dk"].get<int>();
    param.model = paramJson["model"].get<std::string>();
    param.qScale = paramJson["qScale"].get<float>();
    if (paramJson.contains("rotaryPct")) {
        param.rotaryPct = paramJson["rotaryPct"].get<float>();
    }
    if (paramJson.contains("isPrefill")) {
        param.isPrefill = paramJson["isPrefill"].get<bool>();
    }
    if (paramJson.contains("rank")) {
        param.rank = paramJson["rank"].get<int>();
    }
    if (paramJson.contains("rankSize")) {
        param.rankSize = paramJson["rankSize"].get<int>();
    }
    if (paramJson.contains("qkScale")) {
        param.qkScale = paramJson["qkScale"].get<int>();
    }

    ATB_LOG(INFO) << __func__ << " layerNormEps:" << param.layerNormEps << ", headNum:" << param.headNum << ", dk:" <<
        param.dk << ", model:" << param.model;
    atb::Operation *op;
    FlashAttentionKvCacheLayer(param, &op);
    return op;
}
} // namespace gptneox_20b
} // atb_speed