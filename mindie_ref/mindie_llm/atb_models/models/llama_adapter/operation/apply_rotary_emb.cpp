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
#include "apply_rotary_emb.h"

namespace atb_speed {
namespace llama_adapter {
enum ApplyRotaryEmbTensorId : int {
    IN_XQ = 0,    // [batch,seqlen,32,128]
    IN_XK,        // [batch,seqlen,32,128]
    IN_FREQS_CIS, // [seqLen,62,2]
    OUT_XQ,
    OUT_XK,
    INTERNAL_XQ_REAL,
    INTERNAL_XQ_IMGA,
    INTERNAL_XK_REAL,
    INTERNAL_XK_IMGA,
    INTERNAL_FREQS_FLOAT16,
    INTERNAL_FREQS_REAL,
    INTERNAL_FREQS_IMGA,
    INTERNAL_MULTMP1,
    INTERNAL_MULTMP2,
    INTERNAL_MULTMP3,
    INTERNAL_MULTMP4,
    INTERNAL_REAL1,
    INTERNAL_IMGA1,
    INTERNAL_XQCONCATFLOAT16,
    INTERNAL_MULTMP5,
    INTERNAL_MULTMP6,
    INTERNAL_MULTMP7,
    INTERNAL_MULTMP8,
    INTERNAL_REAL2,
    INTERNAL_IMGA2,
    INTERNAL_XKCONCATFLOAT16,
};
static const int64_t IN_TENSOR_COUNT = 3;
static const int64_t OUT_TENSOR_COUNT = 2;
static const int64_t INTERNAL_TENSOR_COUNT = 21;
static const int64_t NODE_COUNT = 20;

void RopeAdapterReshapeFunc(const atb::Dims &oldShape, atb::Dims &newShape)
{
    newShape.dimNum = oldShape.dimNum + 1;
    newShape.dims[0] = oldShape.dims[0];
    newShape.dims[1] = oldShape.dims[1];
    newShape.dims[2] = oldShape.dims[2];
    newShape.dims[3] = oldShape.dims[3] / 2;
    newShape.dims[4] = 2;
}

void ReshapeAdapterReshapeFunc(const atb::Dims &oldShape, atb::Dims &newShape)
{
    newShape.dimNum = oldShape.dimNum - 1;
    newShape.dims[0] = oldShape.dims[0];
    newShape.dims[1] = oldShape.dims[1];
    newShape.dims[2] = oldShape.dims[2];
    newShape.dims[3] = oldShape.dims[3] * oldShape.dims[4];
}

atb::Status ApplyRotaryEmb(const ApplyRotaryEmbParam &param, atb::Operation **operation)
{
    ATB_LOG(INFO) << __func__ << " called";
    (void)param;
    atb::GraphParam opGraph;
    opGraph.name = "ApplyRotaryEmb";
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERNAL_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);

    int64_t nodeId = 0;
    auto &xqSplitNode = opGraph.nodes[nodeId++];
    auto &xkSplitNode = opGraph.nodes[nodeId++];
    auto &freqsCastNode = opGraph.nodes[nodeId++];
    auto &freqsSplitNode = opGraph.nodes[nodeId++];
    auto &mul1Node = opGraph.nodes[nodeId++];
    auto &mul2Node = opGraph.nodes[nodeId++];
    auto &mul3Node = opGraph.nodes[nodeId++];
    auto &mul4Node = opGraph.nodes[nodeId++];
    auto &sub1Node = opGraph.nodes[nodeId++];
    auto &add1Node = opGraph.nodes[nodeId++];
    auto &concat1Node = opGraph.nodes[nodeId++];
    auto &reshapeXqOutNode = opGraph.nodes[nodeId++];
    auto &mul5Node = opGraph.nodes[nodeId++];
    auto &mul6Node = opGraph.nodes[nodeId++];
    auto &mul7Node = opGraph.nodes[nodeId++];
    auto &mul8Node = opGraph.nodes[nodeId++];
    auto &sub2Node = opGraph.nodes[nodeId++];
    auto &add2Node = opGraph.nodes[nodeId++];
    auto &concat2Node = opGraph.nodes[nodeId++];
    auto &reshapeXkOutNode = opGraph.nodes[nodeId++];

    // split img to real
    atb::infer::SplitParam splitParam = { 4, 2 };
    CREATE_OPERATION(splitParam, &xqSplitNode.operation);
    xqSplitNode.inTensorIds = { IN_XQ };
    xqSplitNode.outTensorIds = { INTERNAL_XQ_REAL, INTERNAL_XQ_IMGA };
    xqSplitNode.inTensorReshapeFuncs = { &RopeAdapterReshapeFunc };

    CREATE_OPERATION(splitParam, &xkSplitNode.operation);
    xkSplitNode.inTensorIds = { IN_XK };
    xkSplitNode.outTensorIds = { INTERNAL_XK_REAL, INTERNAL_XK_IMGA };
    xkSplitNode.inTensorReshapeFuncs = { &RopeAdapterReshapeFunc };

    atb::infer::ElewiseParam castParam;
    castParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_CAST;
    CREATE_OPERATION(castParam, &freqsCastNode.operation);
    freqsCastNode.inTensorIds = { IN_FREQS_CIS };
    freqsCastNode.outTensorIds = { INTERNAL_FREQS_FLOAT16 };

    CREATE_OPERATION(splitParam, &freqsSplitNode.operation);
    freqsSplitNode.inTensorIds = { INTERNAL_FREQS_FLOAT16 };
    freqsSplitNode.outTensorIds = { INTERNAL_FREQS_REAL, INTERNAL_FREQS_IMGA };
    freqsSplitNode.inTensorReshapeFuncs.resize(freqsSplitNode.inTensorIds.size());
    freqsSplitNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 5;
        newShape.dims[0] = 1;
        newShape.dims[1] = oldShape.dims[0];
        newShape.dims[2] = 1;
        newShape.dims[3] = oldShape.dims[1];
        newShape.dims[4] = oldShape.dims[2];
    };

    atb::infer::ElewiseParam mulParam;
    mulParam.elewiseType = atb::infer::ElewiseParam::ELEWISE_MUL;
    CREATE_OPERATION(mulParam, &mul1Node.operation);
    mul1Node.inTensorIds = { INTERNAL_XQ_REAL, INTERNAL_FREQS_REAL };
    mul1Node.outTensorIds = { INTERNAL_MULTMP1 };

    CREATE_OPERATION(mulParam, &mul2Node.operation);
    mul2Node.inTensorIds = { INTERNAL_XQ_IMGA, INTERNAL_FREQS_IMGA };
    mul2Node.outTensorIds = { INTERNAL_MULTMP2 };

    CREATE_OPERATION(mulParam, &mul3Node.operation);
    mul3Node.inTensorIds = { INTERNAL_XQ_IMGA, INTERNAL_FREQS_REAL };
    mul3Node.outTensorIds = { INTERNAL_MULTMP3 };

    CREATE_OPERATION(mulParam, &mul4Node.operation);
    mul4Node.inTensorIds = { INTERNAL_XQ_REAL, INTERNAL_FREQS_IMGA };
    mul4Node.outTensorIds = { INTERNAL_MULTMP4 };

    atb::infer::ElewiseParam subParam;
    subParam.elewiseType = atb::infer::ElewiseParam::ELEWISE_SUB;
    CREATE_OPERATION(subParam, &sub1Node.operation);
    sub1Node.inTensorIds = { INTERNAL_MULTMP1, INTERNAL_MULTMP2 };
    sub1Node.outTensorIds = { INTERNAL_REAL1 };

    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ELEWISE_ADD;
    CREATE_OPERATION(addParam, &add1Node.operation);
    add1Node.inTensorIds = { INTERNAL_MULTMP3, INTERNAL_MULTMP4 };
    add1Node.outTensorIds = { INTERNAL_IMGA1 };

    atb::infer::ConcatParam catParam = { 4 };
    CREATE_OPERATION(catParam, &concat1Node.operation);
    concat1Node.inTensorIds = { INTERNAL_REAL1, INTERNAL_IMGA1 };
    concat1Node.outTensorIds = { INTERNAL_XQCONCATFLOAT16 };

    atb::infer::TransposeParam permuteKParam = { { 0, 1, 2, 3 } };
    CREATE_OPERATION(permuteKParam, &reshapeXqOutNode.operation);
    reshapeXqOutNode.inTensorIds = { INTERNAL_XQCONCATFLOAT16 };
    reshapeXqOutNode.outTensorIds = { OUT_XQ };
    reshapeXqOutNode.inTensorReshapeFuncs = { &ReshapeAdapterReshapeFunc };

    CREATE_OPERATION(mulParam, &mul5Node.operation);
    mul5Node.inTensorIds = { INTERNAL_XK_REAL, INTERNAL_FREQS_REAL };
    mul5Node.outTensorIds = { INTERNAL_MULTMP5 };

    CREATE_OPERATION(mulParam, &mul6Node.operation);
    mul6Node.inTensorIds = { INTERNAL_XK_IMGA, INTERNAL_FREQS_IMGA };
    mul6Node.outTensorIds = { INTERNAL_MULTMP6 };

    CREATE_OPERATION(mulParam, &mul7Node.operation);
    mul7Node.inTensorIds = { INTERNAL_XK_IMGA, INTERNAL_FREQS_REAL };
    mul7Node.outTensorIds = { INTERNAL_MULTMP7 };

    CREATE_OPERATION(mulParam, &mul8Node.operation);
    mul8Node.inTensorIds = { INTERNAL_XK_REAL, INTERNAL_FREQS_IMGA };
    mul8Node.outTensorIds = { INTERNAL_MULTMP8 };

    CREATE_OPERATION(subParam, &sub2Node.operation);
    sub2Node.inTensorIds = { INTERNAL_MULTMP5, INTERNAL_MULTMP6 };
    sub2Node.outTensorIds = { INTERNAL_REAL2 };

    CREATE_OPERATION(addParam, &add2Node.operation);
    add2Node.inTensorIds = { INTERNAL_MULTMP7, INTERNAL_MULTMP8 };
    add2Node.outTensorIds = { INTERNAL_IMGA2 };

    CREATE_OPERATION(catParam, &concat2Node.operation);
    concat2Node.inTensorIds = { INTERNAL_REAL2, INTERNAL_IMGA2 };
    concat2Node.outTensorIds = { INTERNAL_XKCONCATFLOAT16 };

    CREATE_OPERATION(permuteKParam, &reshapeXkOutNode.operation);
    reshapeXkOutNode.inTensorIds = { INTERNAL_XKCONCATFLOAT16 };
    reshapeXkOutNode.outTensorIds = { OUT_XK };
    reshapeXkOutNode.inTensorReshapeFuncs = { &ReshapeAdapterReshapeFunc };

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
        atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        outTensorDescs.at(1) = inTensorDescs.at(1);
        return atb::NO_ERROR;
    };
    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}
} // namespace llama_adapter
} // namespace atb_speed
