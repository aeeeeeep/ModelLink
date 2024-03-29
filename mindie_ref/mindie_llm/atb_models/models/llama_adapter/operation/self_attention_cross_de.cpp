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
#include "self_attention_cross.h"

#include <cmath>

#include "atb/atb_infer.h"

#include "atb_speed/log.h"

namespace atb_speed {
namespace llama_adapter {
static const uint64_t IN_TENSOR_COUNT = 6;
static const uint64_t OUT_TENSOR_COUNT = 3;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 10;
static const uint64_t NODE_COUNT = 13;
enum SelfAttentionCrossDeTensorId : int {
    IN_XQ = 0, // [bs, sq, hidden_states]
    IN_XK,
    IN_XV,
    IN_PASTK,
    IN_PASTV,
    IN_ATTENTION_MASK,
    OUT_CONTEXT_OUT,
    OUT_PRESENT_KEY,
    OUT_PRESENT_VALUE,
    INTERNAL_PQ,
    INTERNAL_PK,
    INTERNAL_PV,
    INTERNAL_BMM_QK_OUT,
    INTERNAL_Q_SCALED_OUT,
    INTERNAL_ATTENTION_SCORES,
    INTERNAL_ATTENTION_SCORES_F32,
    INTERNAL_ATTENTION_PROBS_F32,
    INTERNAL_ATTENTION_PROBS,
    INTERNAL_BMM_V_OUT, // [bs, sq, hidden_states]
};

void BmmReshapeDeFunc(const atb::Dims &oldShape, atb::Dims &newShape)
{
    newShape.dimNum = oldShape.dimNum - 1;
    newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
    newShape.dims[1] = oldShape.dims[2];
    newShape.dims[2] = oldShape.dims[3];
}

atb::Status SelfAttentionCrossDe(const SelfAttentionCrossParam &param, atb::Operation **operation)
{
    ATB_LOG(INFO) << __func__ << " dk: " << param.dk << ", headNum: " << param.headNum;
    atb::GraphParam opGraph;
    opGraph.name = "SelfAttentionCrossDe";
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);

    int64_t nodeId = 0;
    auto &catKeyNode = opGraph.nodes.at(nodeId++);
    auto &catValueNode = opGraph.nodes.at(nodeId++);
    auto &permuteQNode = opGraph.nodes.at(nodeId++);
    auto &permuteVNode = opGraph.nodes.at(nodeId++);
    auto &permuteKNode = opGraph.nodes.at(nodeId++);
    auto &bmmQKNode = opGraph.nodes.at(nodeId++);
    auto &mulsQNode = opGraph.nodes.at(nodeId++);
    auto &addMaskNode = opGraph.nodes.at(nodeId++);
    auto &castInNode = opGraph.nodes.at(nodeId++);
    auto &softMaxNode = opGraph.nodes.at(nodeId++);
    auto &castOutNode = opGraph.nodes.at(nodeId++);
    auto &bmmVNode = opGraph.nodes.at(nodeId++);
    auto &transposeOutputNode = opGraph.nodes.at(nodeId++);

    atb::infer::ConcatParam concatParam = {1};
    CREATE_OPERATION(concatParam, &catKeyNode.operation);
    catKeyNode.inTensorIds = {IN_PASTK, IN_XK};
    catKeyNode.outTensorIds = {OUT_PRESENT_KEY};

    CREATE_OPERATION(concatParam, &catValueNode.operation);
    catValueNode.inTensorIds = {IN_PASTV, IN_XV};
    catValueNode.outTensorIds = {OUT_PRESENT_VALUE};

    atb::infer::TransposeParam oriTranspose2Param = { { 0, 2, 1, 3 } };
    CREATE_OPERATION(oriTranspose2Param, &permuteQNode.operation);
    permuteQNode.inTensorIds = { IN_XQ };
    permuteQNode.outTensorIds = { INTERNAL_PQ };

    CREATE_OPERATION(oriTranspose2Param, &permuteVNode.operation);
    permuteVNode.inTensorIds = { OUT_PRESENT_VALUE };
    permuteVNode.outTensorIds = { INTERNAL_PV };

    atb::infer::TransposeParam oriTranspose3Param = { { 0, 2, 3, 1 } };
    CREATE_OPERATION(oriTranspose3Param, &permuteKNode.operation);
    permuteKNode.inTensorIds = { OUT_PRESENT_KEY };
    permuteKNode.outTensorIds = { INTERNAL_PK };

    atb::infer::LinearParam linearParam = { false, false, false };
    CREATE_OPERATION(linearParam, &bmmQKNode.operation);
    bmmQKNode.inTensorIds = { INTERNAL_PQ, INTERNAL_PK };
    bmmQKNode.outTensorIds = { INTERNAL_BMM_QK_OUT };
    bmmQKNode.inTensorReshapeFuncs = { &BmmReshapeDeFunc, &BmmReshapeDeFunc };

    float scalingAttr = 1.0 / sqrt(param.dk);
    ATB_LOG(INFO) << "Scaling down for query with scaling factor " << scalingAttr;
    atb::infer::ElewiseParam scalingElewiseMulsParam;
    scalingElewiseMulsParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MULS;
    scalingElewiseMulsParam.mulsParam.varAttr = scalingAttr;
    CREATE_OPERATION(scalingElewiseMulsParam, &mulsQNode.operation);
    mulsQNode.inTensorIds = { INTERNAL_BMM_QK_OUT };
    mulsQNode.outTensorIds = { INTERNAL_Q_SCALED_OUT };

    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CREATE_OPERATION(addParam, &addMaskNode.operation);
    addMaskNode.inTensorIds = { IN_ATTENTION_MASK, INTERNAL_Q_SCALED_OUT };
    addMaskNode.outTensorIds = { INTERNAL_ATTENTION_SCORES };
    addMaskNode.inTensorReshapeFuncs.resize(addMaskNode.inTensorIds.size());
    addMaskNode.inTensorReshapeFuncs[1] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 4;
        newShape.dims[0] = oldShape.dims[0] / param.headNum;
        newShape.dims[1] = param.headNum;
        newShape.dims[2] = oldShape.dims[1];
        newShape.dims[3] = oldShape.dims[2];
    };

    atb::infer::ElewiseParam castParam;
    castParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_CAST;
    CREATE_OPERATION(castParam, &castInNode.operation);
    castInNode.inTensorIds = { INTERNAL_ATTENTION_SCORES };
    castInNode.outTensorIds = { INTERNAL_ATTENTION_SCORES_F32 };

    atb::infer::SoftmaxParam softmaxParam = { { -1 } };
    CREATE_OPERATION(softmaxParam, &softMaxNode.operation);
    softMaxNode.inTensorIds = { INTERNAL_ATTENTION_SCORES_F32 };
    softMaxNode.outTensorIds = { INTERNAL_ATTENTION_PROBS_F32 };

    CREATE_OPERATION(castParam, &castOutNode.operation);
    castOutNode.inTensorIds = { INTERNAL_ATTENTION_PROBS_F32 };
    castOutNode.outTensorIds = { INTERNAL_ATTENTION_PROBS };

    CREATE_OPERATION(linearParam, &bmmVNode.operation);
    bmmVNode.inTensorIds = { INTERNAL_ATTENTION_PROBS, INTERNAL_PV };
    bmmVNode.outTensorIds = { INTERNAL_BMM_V_OUT };
    bmmVNode.inTensorReshapeFuncs = { &BmmReshapeDeFunc, &BmmReshapeDeFunc };

    atb::infer::TransposeParam oriTransposeParam = { { 0, 2, 1, 3 } };
    CREATE_OPERATION(oriTransposeParam, &transposeOutputNode.operation);
    transposeOutputNode.inTensorIds = { INTERNAL_BMM_V_OUT };
    transposeOutputNode.outTensorIds = { OUT_CONTEXT_OUT };
    transposeOutputNode.inTensorReshapeFuncs.resize(transposeOutputNode.inTensorIds.size());
    transposeOutputNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 4;
        newShape.dims[0] = oldShape.dims[0] / param.headNum;
        newShape.dims[1] = param.headNum;
        newShape.dims[2] = oldShape.dims[1];
        newShape.dims[3] = oldShape.dims[2];
    };

    opGraph.inferShapeFunc = [](const atb::SVector<atb::TensorDesc> &inTensorDescs,
        atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        outTensorDescs.at(0).shape.dimNum = 3;
        outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
        outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1];
        outTensorDescs.at(0).shape.dims[2] = inTensorDescs.at(0).shape.dims[2] * inTensorDescs.at(0).shape.dims[3];

        outTensorDescs.at(1) = inTensorDescs.at(3);
        outTensorDescs.at(1).shape.dims[1] = outTensorDescs.at(1).shape.dims[1] + 1;
        outTensorDescs.at(2) = inTensorDescs.at(4);
        outTensorDescs.at(2).shape.dims[1] = outTensorDescs.at(2).shape.dims[1] + 1;

        ATB_LOG(INFO) << __func__ << " infer shape success";
        return atb::NO_ERROR;
    };

    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}
} // namespace llama_adapter
} // namespace atb_speed
