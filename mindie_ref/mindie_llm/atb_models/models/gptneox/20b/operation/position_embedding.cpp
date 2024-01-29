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
#include "position_embedding.h"
#include "atb/types.h"

namespace atb_speed {
namespace gptneox_20b {
enum PositionEmbeddingTensorId : int {
    IN_MIXEDQKV = 0, // [batch, seqLen, hiddenSize], half
    IN_POSITION_ID,  // [batch, seqLen], int64
    IN_COS_EMBED,    // [1, 1, maxseqLen, headDim], half
    IN_SIN_EMBED,    // [1, 1, maxseqLen, headDim], half
    OUT_Q_EMBED,
    OUT_K_EMBED,
    OUT_VALUE, // [batch, seqlen, headNum, headDim], half

    INTERNAL_Q_SPLIT,
    INTERNAL_K_SPLIT,
    INTERNAL_Q_ROT,
    INTERNAL_Q_PASS,
    INTERNAL_K_ROT,
    INTERNAL_K_PASS,

    INTERNAL_Q_ROT_LEFT,
    INTERNAL_Q_ROT_RIGHT,
    INTERNAL_Q_ROT_RIGHT_NEG,
    INTERNAL_Q_ROT_CAT,
    INTERNAL_Q_ROT_COS_MUL,
    INTERNAL_Q_ROT_SIN_MUL,
    INTERNAL_Q_ROT_ADD,

    INTERNAL_K_ROT_LEFT,
    INTERNAL_K_ROT_RIGHT,
    INTERNAL_K_ROT_RIGHT_NEG,
    INTERNAL_K_ROT_CAT,
    INTERNAL_K_ROT_COS_MUL,
    INTERNAL_K_ROT_SIN_MUL,
    INTERNAL_K_ROT_ADD,
};
static const int64_t IN_TENSOR_COUNT = 4;
static const int64_t OUT_TENSOR_COUNT = 3;
static const int64_t INTERNAL_TENSOR_COUNT = 20;
static const int64_t NODE_COUNT = 19;
static const int64_t OUT_TENSOR_DIM_NUM = 4;
static const int64_t UNSQUEEZE_COS_SIN_DIM_SIZE = 1;
static const int64_t QKV_SPLIT_DIM = 3;
static const int64_t QKV_SPLIT_NUM = 3;
static const int64_t Q_SPLIT_HALF_DIM = 3;
static const int64_t Q_SPLIT_HALF_NUM = 2;
static const int64_t Q_ROT_NEG_CAT_DIM = 3;
static const int64_t Q_ROT_PASS_CAT_DIM = 3;

void unsqueezeCosSinView(const atb::Dims &oldShape, atb::Dims &newShape)
{
    newShape.dimNum = OUT_TENSOR_DIM_NUM;
    newShape.dims[0] = oldShape.dims[0];
    newShape.dims[1] = oldShape.dims[1];
    newShape.dims[2] = UNSQUEEZE_COS_SIN_DIM_SIZE;
    newShape.dims[3] = oldShape.dims[2];
}

atb::Status PositionEmbedding(const PositionEmbeddingParam &param, atb::Operation **operation)
{
    ATB_LOG(INFO) << __func__ << " called, headNum: " << param.headNum << ", dk:" << param.dk << ", rotaryPct:" <<
        param.rotaryPct;

    atb::GraphParam opGraph;
    opGraph.name = "positionEmbeddingOperation";
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
    // do rotary embedding for q
    auto &qSplitHalfNode = opGraph.nodes[nodeId++];
    auto &qNegNode = opGraph.nodes[nodeId++];
    auto &qCatNegNode = opGraph.nodes[nodeId++];
    auto &qCosMulNode = opGraph.nodes[nodeId++];
    auto &qSinMulNode = opGraph.nodes[nodeId++];
    auto &qAddNode = opGraph.nodes[nodeId++];
    // do rotary embedding for k
    auto &kSliceNode = opGraph.nodes[nodeId++];
    auto &kNegNode = opGraph.nodes[nodeId++];
    auto &kCatNegNode = opGraph.nodes[nodeId++];
    auto &kCosMulNode = opGraph.nodes[nodeId++];
    auto &kSinMulNode = opGraph.nodes[nodeId++];
    auto &kAddNode = opGraph.nodes[nodeId++];
    // do cat and output
    auto &qCatNode = opGraph.nodes[nodeId++];
    auto &kCatNode = opGraph.nodes[nodeId++];

    int64_t rotaryNum = param.dk * param.rotaryPct;
    int64_t passNum = param.dk - rotaryNum;
    ATB_LOG(INFO) << __func__ << "Rotary num is " << rotaryNum << " pass num is " << passNum;
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

    // [bs, sq, hn, rotaryNum] --> 2 of [bs, sq, hn, rotaryNum/2]
    atb::infer::SplitParam splitHalfParam;
    splitHalfParam.splitDim = Q_SPLIT_HALF_DIM;
    splitHalfParam.splitNum = Q_SPLIT_HALF_NUM;
    CREATE_OPERATION(splitHalfParam, &qSplitHalfNode.operation);
    qSplitHalfNode.inTensorIds = { INTERNAL_Q_ROT };
    qSplitHalfNode.outTensorIds = { INTERNAL_Q_ROT_LEFT, INTERNAL_Q_ROT_RIGHT };

    atb::infer::ElewiseParam mulsParam;
    mulsParam.elewiseType = atb::infer::ElewiseParam::ELEWISE_MULS;
    mulsParam.mulsParam.varAttr = -1;
    CREATE_OPERATION(mulsParam, &qNegNode.operation);
    qNegNode.inTensorIds = { INTERNAL_Q_ROT_RIGHT };
    qNegNode.outTensorIds = { INTERNAL_Q_ROT_RIGHT_NEG };

    // [bs, sq, hn, rotaryNum/2] -> [bs, sq, hn, rotaryNum]
    atb::infer::ConcatParam catParam;
    catParam.concatDim = Q_ROT_NEG_CAT_DIM;
    CREATE_OPERATION(catParam, &qCatNegNode.operation);
    qCatNegNode.inTensorIds = { INTERNAL_Q_ROT_RIGHT_NEG, INTERNAL_Q_ROT_LEFT };
    qCatNegNode.outTensorIds = { INTERNAL_Q_ROT_CAT };

    atb::infer::ElewiseParam mulParam;
    mulParam.elewiseType = atb::infer::ElewiseParam::ELEWISE_MUL;
    CREATE_OPERATION(mulParam, &qCosMulNode.operation);
    qCosMulNode.inTensorIds = { INTERNAL_Q_ROT, IN_COS_EMBED };
    qCosMulNode.outTensorIds = { INTERNAL_Q_ROT_COS_MUL };
    qCosMulNode.inTensorReshapeFuncs.resize(qCosMulNode.inTensorIds.size());
    qCosMulNode.inTensorReshapeFuncs[1] = &unsqueezeCosSinView; // [bs, sq, 1, rotaryNum]

    CREATE_OPERATION(mulParam, &qSinMulNode.operation);
    qSinMulNode.inTensorIds = { INTERNAL_Q_ROT_CAT, IN_SIN_EMBED };
    qSinMulNode.outTensorIds = { INTERNAL_Q_ROT_SIN_MUL };
    qSinMulNode.inTensorReshapeFuncs.resize(qSinMulNode.inTensorIds.size());
    qSinMulNode.inTensorReshapeFuncs[1] = &unsqueezeCosSinView; // [bs, sq, 1, rotaryNum]

    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ELEWISE_ADD;
    CREATE_OPERATION(addParam, &qAddNode.operation);
    qAddNode.inTensorIds = { INTERNAL_Q_ROT_COS_MUL, INTERNAL_Q_ROT_SIN_MUL };
    qAddNode.outTensorIds = { INTERNAL_Q_ROT_ADD };

    // do k rotary
    CREATE_OPERATION(splitHalfParam, &kSliceNode.operation);
    kSliceNode.inTensorIds = { INTERNAL_K_ROT };
    kSliceNode.outTensorIds = { INTERNAL_K_ROT_LEFT, INTERNAL_K_ROT_RIGHT };

    CREATE_OPERATION(mulsParam, &kNegNode.operation);
    kNegNode.inTensorIds = { INTERNAL_K_ROT_RIGHT };
    kNegNode.outTensorIds = { INTERNAL_K_ROT_RIGHT_NEG };

    CREATE_OPERATION(catParam, &kCatNegNode.operation);
    kCatNegNode.inTensorIds = { INTERNAL_K_ROT_RIGHT_NEG, INTERNAL_K_ROT_LEFT };
    kCatNegNode.outTensorIds = { INTERNAL_K_ROT_CAT };

    CREATE_OPERATION(mulParam, &kCosMulNode.operation);
    kCosMulNode.inTensorIds = { INTERNAL_K_ROT, IN_COS_EMBED };
    kCosMulNode.outTensorIds = { INTERNAL_K_ROT_COS_MUL };
    kCosMulNode.inTensorReshapeFuncs.resize(kCosMulNode.inTensorIds.size());
    kCosMulNode.inTensorReshapeFuncs[1] = &unsqueezeCosSinView; // [bs, sq, 1, rotaryNum]

    CREATE_OPERATION(mulParam, &kSinMulNode.operation);
    kSinMulNode.inTensorIds = { INTERNAL_K_ROT_CAT, IN_SIN_EMBED };
    kSinMulNode.outTensorIds = { INTERNAL_K_ROT_SIN_MUL };
    kSinMulNode.inTensorReshapeFuncs.resize(kSinMulNode.inTensorIds.size());
    kSinMulNode.inTensorReshapeFuncs[1] = &unsqueezeCosSinView; // [bs, sq, 1, rotaryNum]

    CREATE_OPERATION(addParam, &kAddNode.operation);
    kAddNode.inTensorIds = { INTERNAL_K_ROT_COS_MUL, INTERNAL_K_ROT_SIN_MUL };
    kAddNode.outTensorIds = { INTERNAL_K_ROT_ADD };

    atb::infer::ConcatParam concatParam;
    concatParam.concatDim = Q_ROT_PASS_CAT_DIM;
    CREATE_OPERATION(concatParam, &qCatNode.operation);
    qCatNode.inTensorIds = { INTERNAL_Q_ROT_ADD, INTERNAL_Q_PASS };
    qCatNode.outTensorIds = { OUT_Q_EMBED };

    CREATE_OPERATION(concatParam, &kCatNode.operation);
    kCatNode.inTensorIds = { INTERNAL_K_ROT_ADD, INTERNAL_K_PASS };
    kCatNode.outTensorIds = { OUT_K_EMBED };

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
        atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        outTensorDescs.at(0).shape.dimNum = OUT_TENSOR_DIM_NUM;
        outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
        outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1];
        outTensorDescs.at(0).shape.dims[2] = param.headNum;
        outTensorDescs.at(0).shape.dims[3] = inTensorDescs.at(0).shape.dims[2] / param.headNum / 3;
        // [bs, sq, hn, hs]
        outTensorDescs.at(1) = outTensorDescs.at(0);
        outTensorDescs.at(2) = outTensorDescs.at(0);
        return atb::NO_ERROR;
    };
    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}

atb::Operation *CreatePositionEmbedding(const nlohmann::json &paramJson)
{
    PositionEmbeddingParam param;
    param.headNum = paramJson["headNum"].get<int>();
    param.dk = paramJson["dk"].get<int>();
    if (paramJson.contains("rotaryPct")) {
        param.rotaryPct = paramJson["rotaryPct"].get<float>();
    }
    ATB_LOG(INFO) << "GptNeoxPositionEmbeddingParam headNum:" << param.headNum << ", dk:" << param.dk <<
        ", rotaryPct:" << param.rotaryPct;
    atb::Operation *op;
    PositionEmbedding(param, &op);
    return op;
}
} // namespace gptneox_20b
} // namespace atb_speed
