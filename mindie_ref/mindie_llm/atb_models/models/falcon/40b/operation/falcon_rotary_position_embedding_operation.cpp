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

#include "falcon_rotary_position_embedding_operation.h"
#include <atb/atb_infer.h>
#include "atb_speed/log.h"
using namespace std;

namespace atb_speed {
namespace falcon_40b {
static const uint64_t IN_TENSOR_COUNT = 4;
static const uint64_t OUT_TENSOR_COUNT = 3;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 6;
static const uint64_t NODE_COUNT = 7;

enum RotaryPositionEmbeddingTensorId : int {
    IN_FUSED_QKV = 0,
    IN_COS_TABLE,
    IN_SIN_TABLE,
    IN_SEQLEN,
    OUT_Q_EMBEDDED,
    OUT_K_EMBEDDED,
    OUT_V_LAYER,
    INTERMEDIATE_QKV,
    INTERMEDIATE_Q_5D_LAYER,
    INTERMEDIATE_KV_LAYER,
    INTERMEDIATE_K1_LAYER,
    INTERMEDIATE_V1_LAYER,
    INTERMEDIATE_K_5D_LAYER,
};

void Squeeze1(const atb::Dims &oldShape, atb::Dims &newShape)
{
    if (oldShape.dims[1] == 1) {
        newShape.dimNum = oldShape.dimNum - 1;
        newShape.dims[0] = oldShape.dims[0];
        for (size_t i = 1; i < newShape.dimNum; i++) {
            newShape.dims[i] = oldShape.dims[i + 1];
        }
    } else {
        newShape = oldShape;
    }
}

void RopeCosSinReshapeFunc(const atb::Dims &oldShape, atb::Dims &newShape)
{
    newShape.dimNum = 2;
    newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
    newShape.dims[1] = oldShape.dims[2];
}

void RopeQKReshapeFunc(const atb::Dims &oldShape, atb::Dims &newShape)
{ // 5d
    newShape.dimNum = oldShape.dimNum - 3;
    newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
    newShape.dims[1] = oldShape.dims[2] * oldShape.dims[3] * oldShape.dims[4];
}


atb::Status RotaryPositionEmbedding(const RotaryPositionEmbeddingParam &param, atb::Operation **operation)
{
    std::shared_ptr<int64_t> seqLenPtr = std::make_shared<int64_t>(0);
    std::shared_ptr<int64_t> batchSizePtr = std::make_shared<int64_t>(0);
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    auto &viewFusedQkvNode = opGraph.nodes[nodeId++]; // reshape fused_qkv
    auto &sliceQNode = opGraph.nodes[nodeId++]; // slice Q from qkv
    auto &sliceKVNode = opGraph.nodes[nodeId++]; // slice KV from qkv
    auto &splitKVNode = opGraph.nodes[nodeId++]; // split K, V from KV
    auto &keyRepeatNode = opGraph.nodes[nodeId++]; // broadcast   key to shape of Q
    auto &valueRepeatNode = opGraph.nodes[nodeId++]; // broadcast value to shape of Q
    auto &ropeNode = opGraph.nodes[nodeId++];
    
    // [batch_size, seq_length, 9216]
    // qkv = fused_qkv.view(batch, seq_len, -1, self.num_heads // self.num_kv_heads + 2, self.head_dim)
    atb::infer::ElewiseParam viewFusedQkvParam;
    viewFusedQkvParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MULS;
    viewFusedQkvParam.mulsParam.varAttr = 1.0;
    CreateOperation(viewFusedQkvParam, &viewFusedQkvNode.operation);
    viewFusedQkvNode.inTensorIds = {IN_FUSED_QKV};
    viewFusedQkvNode.outTensorIds = {INTERMEDIATE_QKV};
    viewFusedQkvNode.inTensorReshapeFuncs.resize(viewFusedQkvNode.inTensorIds.size());
    viewFusedQkvNode.inTensorReshapeFuncs.at(0) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 5;
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = oldShape.dims[1];
        newShape.dims[3] = param.headNum / param.kvHeadNum + 2;
        newShape.dims[4] = param.headDim;                         // self.head_dim; 64
        // oldShape 是 3 维的, 剩余 1 维的 size 可以计算出来，相当于 view 的 -1, (2304/(18*64)) = 2
        newShape.dims[2] = oldShape.dims[2] / (newShape.dims[3] * newShape.dims[4]);
    };
    
    // query = qkv[:, :, :, :-2, :]
    atb::infer::SliceParam sliceQNodeParam;
    sliceQNodeParam.offsets = {0, 0, 0, 0, 0};
    sliceQNodeParam.size = {-1, -1, -1, param.headNum / param.kvHeadNum, -1};
    CreateOperation(sliceQNodeParam, &sliceQNode.operation);
    sliceQNode.inTensorIds = {INTERMEDIATE_QKV};
    sliceQNode.outTensorIds = {INTERMEDIATE_Q_5D_LAYER};

    // key and value = qkv[:, :, :, -2:, :]
    atb::infer::SliceParam sliceKVNodeParam;
    sliceKVNodeParam.offsets = {0, 0, 0, param.headNum / param.kvHeadNum, 0};
    sliceKVNodeParam.size = {-1, -1, -1, 2, -1};
    CreateOperation(sliceKVNodeParam, &sliceKVNode.operation);
    sliceKVNode.inTensorIds = {INTERMEDIATE_QKV};
    sliceKVNode.outTensorIds = {INTERMEDIATE_KV_LAYER};

    // key = qkv[:, :, :, [-2]],  value = qkv[:, :, :, [-1]]
    atb::infer::SplitParam splitKVParam = {3, 2}; // splitDim
    CreateOperation(splitKVParam, &splitKVNode.operation);
    splitKVNode.inTensorIds = {INTERMEDIATE_KV_LAYER};
    splitKVNode.outTensorIds = {INTERMEDIATE_K1_LAYER, INTERMEDIATE_V1_LAYER};

    // 广播 k v 的 dim=3 到 q 的维度
    atb::infer::RepeatParam kvRepeatParam;
    kvRepeatParam.multiples = {1, 1, 1, param.headNum / param.kvHeadNum, 1};

    // key = torch.broadcast_to(key, query.shape)
    CreateOperation(kvRepeatParam, &keyRepeatNode.operation);
    keyRepeatNode.inTensorIds = {INTERMEDIATE_K1_LAYER};
    keyRepeatNode.outTensorIds = {INTERMEDIATE_K_5D_LAYER};

    CreateOperation(kvRepeatParam, &valueRepeatNode.operation);
    valueRepeatNode.inTensorIds = {INTERMEDIATE_V1_LAYER};
    valueRepeatNode.outTensorIds = {OUT_V_LAYER};

    // 这里 INTERMEDIATE_Q_5D_LAYER 和 INTERMEDIATE_K_5D_LAYER 都是 5 维的 (batch_size, seq_len, 2, 16, head_dim)
    atb::infer::RopeParam ropeParam;
    ropeParam.rotaryCoeff = 2; //  设置旋转系数
    CreateOperation(ropeParam, &ropeNode.operation);
    ropeNode.inTensorIds = {INTERMEDIATE_Q_5D_LAYER, INTERMEDIATE_K_5D_LAYER, IN_COS_TABLE, IN_SIN_TABLE, IN_SEQLEN};
    ropeNode.outTensorIds = {OUT_Q_EMBEDDED, OUT_K_EMBEDDED};
    ropeNode.inTensorReshapeFuncs = {&RopeQKReshapeFunc, &RopeQKReshapeFunc,
                                     &RopeCosSinReshapeFunc, &RopeCosSinReshapeFunc};

    // input tensor 是 3 维的 (batch, seq_len, head_dim*(num_heads+2))
    // 输出是 (batch_size, num_heads, seq_len, head_dim)
    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        outTensorDescs.at(0).shape.dimNum = 4;                                   // 输出维度
        outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];  // batch_size
        outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1];  // query_length==seq_len
        outTensorDescs.at(0).shape.dims[2] = param.headNum;                      // num_heads = 32
        outTensorDescs.at(0).shape.dims[3] = param.headDim;                      // head_dim  = 64
        outTensorDescs.at(1) = outTensorDescs.at(0);                             // Output tensor1 shape 和 tensor0 相同
        outTensorDescs.at(2) = outTensorDescs.at(0);                             // Output tensor2 shape 和 tensor0 相同
        return atb::NO_ERROR;
    };

    atb::CreateOperation(opGraph, operation);
    return atb::NO_ERROR;
}
} // namespace falcon_40b
} // namespace atb_speed
