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
#include "position_embedding_fusion.h"
#include "atb/types.h"

namespace atb_speed {
namespace gptneox_20b {
enum PositionEmbeddingFusionTensorId : int {
    IN_MIXEDQKV = 0, // [batch, seqLen, hiddenSize], half
    IN_COS_EMBED,    // [maxseqLen, rotaryNum], half
    IN_SIN_EMBED,    // [maxseqLen, rotaryNum], half
    IN_SEQLEN,
    OUT_Q_EMBED,
    OUT_K_EMBED,
    OUT_VALUE, // [batch, seqlen, headNum, headDim], half

    INTERNAL_Q_SPLIT,
    INTERNAL_K_SPLIT,
    INTERNAL_Q_ROT,
    INTERNAL_Q_PASS,
    INTERNAL_K_ROT,
    INTERNAL_K_PASS,

    INTERNAL_Q_ROPE,
    INTERNAL_K_ROPE,
};
static const int64_t IN_TENSOR_COUNT = 4;
static const int64_t OUT_TENSOR_COUNT = 3;
static const int64_t INTERNAL_TENSOR_COUNT = 8;
static const int64_t NODE_COUNT = 8;

static const int64_t OUT_TENSOR_DIM_NUM = 4;
static const int64_t QKV_SPLIT_DIM = 3;
static const int64_t QKV_SPLIT_NUM = 3;
static const int64_t Q_ROT_PASS_CAT_DIM = 2;

void mergeBatchSeqAndHeadFunc(const atb::Dims &oldShape, atb::Dims &newShape)
{
    newShape.dimNum = oldShape.dimNum - 2;
    newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1]; // 第一维 bs * sq;
    newShape.dims[1] = oldShape.dims[2] * oldShape.dims[3]; // 第二维 hn * rotaryNum;
}

void mergeBatchSeqForSinCosFunc(const atb::Dims &oldShape, atb::Dims &newShape)
{
    newShape.dimNum = oldShape.dimNum - 1;
    newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1]; // 第一维 bs * sq
    newShape.dims[1] = oldShape.dims[2];                    // 第二维 rotaryNum
}

void mergeBatchSeqForQKFunc(const atb::Dims &oldShape, atb::Dims &newShape)
{
    newShape.dimNum = oldShape.dimNum - 1;
    newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1]; // 第一维 bs * sq
    newShape.dims[1] = oldShape.dims[2];                    // 第二维 hn
    newShape.dims[2] = oldShape.dims[3];                    // 第三维 passNum
}

atb::Status PositionEmbeddingFusionOperation(const PositionEmbeddingFusionParam &param, atb::Operation **operation)
{
    ATB_LOG(INFO) << __func__ << " called, headNum: " << param.headNum << ", dk:" << param.dk << ", rotaryPct:" <<
        param.rotaryPct;

    atb::GraphParam opGraph;
    opGraph.name = "positionEmbeddingFusionOperation";
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERNAL_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);

    int64_t nodeId = 0;
    auto &splitQkvNode = opGraph.nodes[nodeId++];
    auto &qRotSliceNode = opGraph.nodes[nodeId++];
    auto &qPassSliceNode = opGraph.nodes[nodeId++];
    auto &kRotSliceNode = opGraph.nodes[nodeId++];
    auto &kPassSliceNode = opGraph.nodes[nodeId++];

    // do rope
    auto &ropeNode = opGraph.nodes[nodeId++];

    // do cat and output
    auto &qCatNode = opGraph.nodes[nodeId++];
    auto &kCatNode = opGraph.nodes[nodeId++];

    int64_t rotaryNum = param.dk * param.rotaryPct;
    int64_t passNum = param.dk - rotaryNum;
    ATB_LOG(INFO) << __func__ << "Rotary num is " << rotaryNum << " Pass num is " << passNum;
    atb::SVector<int64_t> sliceOffsetRot = { 0, 0, 0, 0 };
    atb::SVector<int64_t> sliceSizeRot = { -1, -1, -1, rotaryNum };
    atb::SVector<int64_t> sliceOffsetPass = { 0, 0, 0, rotaryNum };
    atb::SVector<int64_t> sliceSizePass = { -1, -1, -1, passNum };

    // split mixedQKV to q k v
    // [bs, sq, hn * 3 * hs] --> [bs, sq, hn, 3*hs] --> 3 of [bs, sq, hn, hs]
    atb::infer::SplitParam splitParam;
    splitParam.splitDim = QKV_SPLIT_DIM;
    splitParam.splitNum = QKV_SPLIT_NUM;
    CREATE_OPERATION(splitParam, &splitQkvNode.operation);
    splitQkvNode.inTensorIds = { IN_MIXEDQKV };
    splitQkvNode.outTensorIds = { INTERNAL_Q_SPLIT, INTERNAL_K_SPLIT, OUT_VALUE };
    splitQkvNode.inTensorReshapeFuncs.resize(splitQkvNode.inTensorIds.size());
    splitQkvNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = OUT_TENSOR_DIM_NUM;
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = oldShape.dims[1];
        newShape.dims[2] = param.headNum;
        newShape.dims[3] = oldShape.dims[2] / param.headNum;
    };

    // [bs, sq, hn, hs] --> [bs, sq, hn, 0:rotaryNum]
    atb::infer::SliceParam sliceRotParam;
    sliceRotParam.offsets = sliceOffsetRot;
    sliceRotParam.size = sliceSizeRot;
    CREATE_OPERATION(sliceRotParam, &qRotSliceNode.operation);
    qRotSliceNode.inTensorIds = { INTERNAL_Q_SPLIT };
    qRotSliceNode.outTensorIds = { INTERNAL_Q_ROT };

    // [bs, sq, hn, hs] --> [bs, sq, hn, rotaryNum:(rotaryNum + passNum)]
    atb::infer::SliceParam slicePassParam;
    slicePassParam.offsets = sliceOffsetPass;
    slicePassParam.size = sliceSizePass;
    CREATE_OPERATION(slicePassParam, &qPassSliceNode.operation);
    qPassSliceNode.inTensorIds = { INTERNAL_Q_SPLIT };
    qPassSliceNode.outTensorIds = { INTERNAL_Q_PASS };

    CREATE_OPERATION(sliceRotParam, &kRotSliceNode.operation);
    kRotSliceNode.inTensorIds = { INTERNAL_K_SPLIT };
    kRotSliceNode.outTensorIds = { INTERNAL_K_ROT };

    CREATE_OPERATION(slicePassParam, &kPassSliceNode.operation);
    kPassSliceNode.inTensorIds = { INTERNAL_K_SPLIT };
    kPassSliceNode.outTensorIds = { INTERNAL_K_PASS };

    // [bs, sq, hn, rotaryNum] --> [bs * sq, hn, rotaryNum]
    atb::infer::RopeParam ropeParam;
    ropeParam.rotaryCoeff = 2;
    CREATE_OPERATION(ropeParam, &ropeNode.operation);
    ropeNode.inTensorIds = { INTERNAL_Q_ROT, INTERNAL_K_ROT, IN_COS_EMBED, IN_SIN_EMBED, IN_SEQLEN };
    ropeNode.outTensorIds = { INTERNAL_Q_ROPE, INTERNAL_K_ROPE };
    ropeNode.inTensorReshapeFuncs = { &mergeBatchSeqAndHeadFunc, &mergeBatchSeqAndHeadFunc, &mergeBatchSeqForSinCosFunc,
        &mergeBatchSeqForSinCosFunc };

    // [bs * sq, hn * rotaryNum] -->  [bs * sq, hn, (rotaryNum + passNum)]
    atb::ReshapeFunc splitHeadsFunc = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = oldShape.dimNum + 1;
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = param.headNum;
        newShape.dims[2] = oldShape.dims[1] / param.headNum;
    };
    atb::infer::ConcatParam concatParam;
    concatParam.concatDim = Q_ROT_PASS_CAT_DIM;
    CREATE_OPERATION(concatParam, &qCatNode.operation);
    qCatNode.inTensorIds = { INTERNAL_Q_ROPE, INTERNAL_Q_PASS };
    qCatNode.outTensorIds = { OUT_Q_EMBED };
    qCatNode.inTensorReshapeFuncs = { splitHeadsFunc, &mergeBatchSeqForQKFunc };

    CREATE_OPERATION(concatParam, &kCatNode.operation);
    kCatNode.inTensorIds = { INTERNAL_K_ROPE, INTERNAL_K_PASS };
    kCatNode.outTensorIds = { OUT_K_EMBED };
    kCatNode.inTensorReshapeFuncs = { splitHeadsFunc, &mergeBatchSeqForQKFunc };

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
        atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        outTensorDescs.at(0).shape.dimNum = OUT_TENSOR_DIM_NUM;
        outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
        outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1];
        outTensorDescs.at(0).shape.dims[2] = param.headNum;
        outTensorDescs.at(0).shape.dims[3] = inTensorDescs.at(0).shape.dims[2] / param.headNum / 3;
        outTensorDescs.at(1) = outTensorDescs.at(0);
        outTensorDescs.at(2) = outTensorDescs.at(0);

        return atb::NO_ERROR;
    };
    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}

atb::Operation *CreatePositionEmbeddingFusionOperation(const nlohmann::json &paramJson)
{
    PositionEmbeddingFusionParam param;
    param.headNum = paramJson["headNum"].get<int>();
    param.dk = paramJson["dk"].get<int>();
    if (paramJson.contains("rotaryPct")) {
        param.rotaryPct = paramJson["rotaryPct"].get<float>();
    }
    ATB_LOG(INFO) << "GptNeoxPositionEmbeddingParam headNum:" << param.headNum << ", dk:" << param.dk <<
        ", rotaryPct:" << param.rotaryPct;
    atb::Operation *op;
    PositionEmbeddingFusionOperation(param, &op);
    return op;
}
} // namespace gptneox_20b
} // namespace atb_speed
