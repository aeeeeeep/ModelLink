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
#include "flash_attention_layer.h"
#include "layers/parallel_layer_v2.h"
#include "mlp_gate_v2.h"

namespace atb_speed {
namespace falcon_40b {
const int ATTENTION_DIM_NUM = 4;
const int ATTENTION_DIM_2 = 2;
const int ATTENTION_DIM_3 = 3;

enum LayerParallelFlashAttentionTensorId : int {
    IN_NORM_ATTN_WEIGHT = 0,    // 0  ln_attn.weight
    IN_NORM_ATTN_BIAS,          // 1  ln_attn.bias
    IN_NORM_MLP_WEIGHT,         // 2  ln_mlp.weight
    IN_NORM_MLP_BIAS,           // 3  ln_mlp.bias
    IN_QKV_FUSED_WEIGHT,        // 4  self_attention.query_key_value.weight
    IN_ATTN_DENSE_WEIGHT,       // 5  self_attention.dense.weight
    IN_MLP_DENSEWEIGHT_UP,      // 6  mlp.dense_h_to_4h.weight
    IN_MLP_DENSEWEIGHT_DOWN,    // 7  mlp.dense_4h_to_h.weight
    IN_HIDDEN_STATES,
    IN_COS_TABLE,
    IN_SIN_TABLE,
    IN_ATTENTIONMASK,
    IN_CACHE_K,
    IN_CACHE_V,
    IN_TOKENOFFSET,
    IN_SEQLEN,
    IN_LAYERID,
    IN_HOLDER,
    OUT_LAYER_OUT,
    INTERMIDATE_INPUTNORM_OUT_ATTN,
    INTERMIDATE_INPUTNORM_OUT_MLP,
    INTERMIDATE_FUSED_QKV,
    INTERMEDIATE_QKV,
    INTERMEDIATE_Q_5D_LAYER,
    INTERMEDIATE_KV_LAYER,
    INTERMEDIATE_K1_LAYER,
    INTERMEDIATE_V1_LAYER,
    INTERMEDIATE_K_5D_LAYER,
    INTERMIDATE_Q_POSITIONEMBED,
    INTERMIDATE_K_POSITIONEMBED,
    INTERMIDATE_VALUE,
    INTERMEDIATE_ATTN_OUTPUT,
    INTERMEDIATE_ATTN_DENSE_OUT,
    INTERMIDATE_MLP_OUT,
    INTERMEDIATE_RES_OUT,
};

static const uint64_t IN_TENSOR_COUNT = 18;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 16;
static const uint64_t NODE_COUNT = 15;

void RopeCosSinReshapeFunc(const atb::Dims &oldShape, atb::Dims &newShape)
{
    newShape.dimNum = 2;  // new dim num is 2
    newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
    newShape.dims[1] = oldShape.dims[2]; // 最后一个维度
}

void RopeQKReshapeFunc(const atb::Dims &oldShape, atb::Dims &newShape)
{
    newShape.dimNum = oldShape.dimNum - 3;  // new dim num is 2
    newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
    newShape.dims[1] = oldShape.dims[2] * oldShape.dims[3] * oldShape.dims[4];  // 最后一个维度
}

atb::Status LayerParallelFlashAttentionOperation(const LayerParallelFlashAttentionParam &param,
                                                 atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    atb::Node &inputNormAttnNode = opGraph.nodes.at(nodeId++);
    atb::Node &inputNormMlpNode = opGraph.nodes.at(nodeId++);
    atb::Node &fusedQkvLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &viewFusedQkvNode = opGraph.nodes[nodeId++];      // reshape fused_qkv
    atb::Node &sliceQNode = opGraph.nodes[nodeId++];            // slice Q from qkv
    atb::Node &sliceKVNode = opGraph.nodes[nodeId++];           // slice KV from qkv
    atb::Node &splitKVNode = opGraph.nodes[nodeId++];           // split K, V from KV
    atb::Node &keyRepeatNode = opGraph.nodes[nodeId++];         // broadcast   key to shape of Q
    atb::Node &valueRepeatNode = opGraph.nodes[nodeId++];       // broadcast value to shape of Q
    atb::Node &ropeNode = opGraph.nodes[nodeId++];
    atb::Node &selfAttentionFusionNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfOutLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpResidualAddNode = opGraph.nodes.at(nodeId++);
    atb::Node &finalResidualAddNode = opGraph.nodes.at(nodeId++);

    atb::infer::LayerNormParam layerNormParam;
    layerNormParam.layerType = atb::infer::LayerNormParam::LAYER_NORM_NORM;
    layerNormParam.normParam.epsilon = param.layerNormEps;
    layerNormParam.normParam.beginNormAxis = 2;    // beginNormAxis is 2
    layerNormParam.normParam.beginParamsAxis = 1;  // beginParamsAxis is 1
    CreateOperation(layerNormParam, &inputNormAttnNode.operation);
    inputNormAttnNode.inTensorIds = {IN_HIDDEN_STATES, IN_NORM_ATTN_WEIGHT, IN_NORM_ATTN_BIAS};
    inputNormAttnNode.outTensorIds = {INTERMIDATE_INPUTNORM_OUT_ATTN};

    CreateOperation(layerNormParam, &inputNormMlpNode.operation);
    inputNormMlpNode.inTensorIds = {IN_HIDDEN_STATES, IN_NORM_MLP_WEIGHT, IN_NORM_MLP_BIAS};
    inputNormMlpNode.outTensorIds = {INTERMIDATE_INPUTNORM_OUT_MLP};

    atb::infer::LinearParam fusedQkvLinearParam;
    fusedQkvLinearParam.hasBias = false;
    CreateOperation(fusedQkvLinearParam, &fusedQkvLinearNode.operation);
    fusedQkvLinearNode.inTensorIds = {INTERMIDATE_INPUTNORM_OUT_ATTN, IN_QKV_FUSED_WEIGHT};
    fusedQkvLinearNode.outTensorIds = {INTERMIDATE_FUSED_QKV};

    std::shared_ptr<int64_t> batchDimPtr = std::make_shared<int64_t>(0);
    std::shared_ptr<int64_t> seqLenPtr = std::make_shared<int64_t>(0);

    // [batch_size, seq_length, 9216]
    // qkv = fused_qkv.view(batch, seq_len, -1, self.num_heads // self.num_kv_heads + 2, self.head_dim)
    atb::infer::ElewiseParam viewFusedQkvParam;
    viewFusedQkvParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MULS;
    viewFusedQkvParam.mulsParam.varAttr = 1.0;
    CreateOperation(viewFusedQkvParam, &viewFusedQkvNode.operation);
    viewFusedQkvNode.inTensorIds = {INTERMIDATE_FUSED_QKV};
    viewFusedQkvNode.outTensorIds = {INTERMEDIATE_QKV};
    viewFusedQkvNode.inTensorReshapeFuncs.resize(viewFusedQkvNode.inTensorIds.size());
    viewFusedQkvNode.inTensorReshapeFuncs.at(0) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 5;                   // reshape
        *batchDimPtr = oldShape.dims[0];
        *seqLenPtr = oldShape.dims[1];
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = oldShape.dims[1];
        newShape.dims[3] = param.headNum / param.kvHeadNum + 2;   // 2 = 1(K) + 1(V)
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
    sliceQNode.inTensorReshapeFuncs.resize(sliceQNode.inTensorIds.size());

    // key and value = qkv[:, :, :, -2:, :]
    atb::infer::SliceParam sliceKVNodeParam;
    sliceKVNodeParam.offsets = {0, 0, 0, param.headNum / param.kvHeadNum, 0};
    sliceKVNodeParam.size = {-1, -1, -1, 2, -1};
    CreateOperation(sliceKVNodeParam, &sliceKVNode.operation);
    sliceKVNode.inTensorIds = {INTERMEDIATE_QKV};
    sliceKVNode.outTensorIds = {INTERMEDIATE_KV_LAYER};
    sliceKVNode.inTensorReshapeFuncs.resize(sliceKVNode.inTensorIds.size());

    // key = qkv[:, :, :, [-2]],  value = qkv[:, :, :, [-1]]
    atb::infer::SplitParam splitKVParam = {3, 2}; // splitDim
    CreateOperation(splitKVParam, &splitKVNode.operation);
    splitKVNode.inTensorIds = {INTERMEDIATE_KV_LAYER};
    splitKVNode.outTensorIds = {INTERMEDIATE_K1_LAYER, INTERMEDIATE_V1_LAYER};
    splitKVNode.inTensorReshapeFuncs.resize(splitKVNode.inTensorIds.size());

    // 广播 k v 的 dim=3 到 q 的维度
    atb::infer::RepeatParam kvRepeatParam;
    kvRepeatParam.multiples = {1, 1, 1, param.headNum / param.kvHeadNum, 1};

    // key = torch.broadcast_to(key, query.shape)
    CreateOperation(kvRepeatParam, &keyRepeatNode.operation);
    keyRepeatNode.inTensorIds = {INTERMEDIATE_K1_LAYER};
    keyRepeatNode.outTensorIds = {INTERMEDIATE_K_5D_LAYER};

    CreateOperation(kvRepeatParam, &valueRepeatNode.operation);
    valueRepeatNode.inTensorIds = {INTERMEDIATE_V1_LAYER};
    valueRepeatNode.outTensorIds = {INTERMIDATE_VALUE};
    valueRepeatNode.inTensorReshapeFuncs.resize(valueRepeatNode.inTensorIds.size());

    // 这里 INTERMEDIATE_Q_5D_LAYER 和 INTERMEDIATE_K_5D_LAYER 都是 5 维的 (batch_size, seq_len, 2, 16, head_dim)
    atb::infer::RopeParam ropeParam;
    ropeParam.rotaryCoeff = 2;   // 设置旋转系数为 2
    CreateOperation(ropeParam, &ropeNode.operation);
    ropeNode.inTensorIds = {INTERMEDIATE_Q_5D_LAYER, INTERMEDIATE_K_5D_LAYER, IN_COS_TABLE, IN_SIN_TABLE, IN_SEQLEN};
    ropeNode.outTensorIds = {INTERMIDATE_Q_POSITIONEMBED, INTERMIDATE_K_POSITIONEMBED};
    ropeNode.inTensorReshapeFuncs = {&RopeQKReshapeFunc, &RopeQKReshapeFunc,
                                     &RopeCosSinReshapeFunc, &RopeCosSinReshapeFunc};

    // RoPE 这边输出得到的 shape 是 batch_size, query_length, num_heads, head_dim
    atb::infer::SelfAttentionParam selfAttentionParam;
    selfAttentionParam.headNum  = param.headNum;
    selfAttentionParam.qScale   = param.qScale;
    selfAttentionParam.qkScale  = param.qkScale;
    selfAttentionParam.maskType = atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_NORM;
    CreateOperation(selfAttentionParam, &selfAttentionFusionNode.operation);
    selfAttentionFusionNode.inTensorIds = {INTERMIDATE_Q_POSITIONEMBED,  // 2
                                           INTERMIDATE_K_POSITIONEMBED,  // 2
                                           INTERMIDATE_VALUE,            // 5
                                           IN_CACHE_K,
                                           IN_CACHE_V,
                                           IN_ATTENTIONMASK,
                                           IN_TOKENOFFSET,
                                           IN_SEQLEN,
                                           IN_LAYERID};
    selfAttentionFusionNode.outTensorIds = {INTERMEDIATE_ATTN_OUTPUT};
    selfAttentionFusionNode.inTensorReshapeFuncs.resize(selfAttentionFusionNode.inTensorIds.size());
    selfAttentionFusionNode.inTensorReshapeFuncs.at(0) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        (void)oldShape;
        newShape.dimNum = 4;                // reshape
        newShape.dims[0] = *batchDimPtr;    // batch_size
        newShape.dims[1] = *seqLenPtr;      // query_length==seq_len
        newShape.dims[2] = param.headNum;   // num_heads = 32
        newShape.dims[3] = param.headDim;   // head_dim  = 64
    };
    selfAttentionFusionNode.inTensorReshapeFuncs.at(1) = selfAttentionFusionNode.inTensorReshapeFuncs.at(0);
    selfAttentionFusionNode.inTensorReshapeFuncs.at(2) = selfAttentionFusionNode.inTensorReshapeFuncs.at(0);

    // attn_output = attn_output.reshape(batch_size, query_length, self.num_heads * self.head_dim)
    // output_tensor = self.dense(attn_output)
    // 这里 dense 之后需要 all_reduce(SUM) 应该使用 parallel
    atb_speed::common::ParallelParamV2 selfOutLinearParam;
    selfOutLinearParam.commParam.rank = param.rank;
    selfOutLinearParam.commParam.rankSize = param.rankSize;
    selfOutLinearParam.commParam.backend = "lccl";
    selfOutLinearParam.isBias = false;
    atb_speed::common::RowParallelLinearV2(selfOutLinearParam, &selfOutLinearNode.operation);
    selfOutLinearNode.inTensorIds = {INTERMEDIATE_ATTN_OUTPUT, IN_ATTN_DENSE_WEIGHT,
                                     IN_HOLDER, IN_HOLDER, IN_HOLDER, IN_HOLDER, IN_HOLDER};
    selfOutLinearNode.outTensorIds = {INTERMEDIATE_ATTN_DENSE_OUT};

    // 最后会自动完成 all_reduce(SUM) 的 Parallel MLP
    // 由于 new_decoder_architecture 和 parallel_attn 都为 True, 所以还是使用 INTERMIDATE_INPUTNORM_OUT_ATTN

    // MlpGateLayerV2
    atb_speed::common::MlpGateParamV2 mlpParam;
    mlpParam.transposeB = true;
    mlpParam.isBias = false;
    mlpParam.isQuant = false;
    mlpParam.isSparse = false;
    mlpParam.noGate = true;
    mlpParam.commDownParam.rank = param.rank;
    mlpParam.commDownParam.rankSize = param.rankSize;
    mlpParam.commDownParam.backend = "lccl";
    mlpParam.activationType = atb::infer::ActivationType::ACTIVATION_GELU;
    atb_speed::common::MlpGateLayerV2(mlpParam, &mlpNode.operation);
    mlpNode.inTensorIds = {INTERMIDATE_INPUTNORM_OUT_MLP, IN_MLP_DENSEWEIGHT_UP, IN_HOLDER,
                           IN_MLP_DENSEWEIGHT_DOWN, IN_HOLDER, IN_HOLDER, IN_HOLDER, IN_HOLDER,
                           IN_HOLDER, IN_HOLDER, IN_HOLDER, IN_HOLDER, IN_HOLDER, IN_HOLDER,
                           IN_HOLDER, IN_HOLDER, IN_HOLDER, IN_HOLDER, IN_HOLDER};
    mlpNode.outTensorIds = {INTERMIDATE_MLP_OUT};

    atb::infer::ElewiseParam mlpResidualAddParam;
    mlpResidualAddParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CreateOperation(mlpResidualAddParam, &mlpResidualAddNode.operation);
    mlpResidualAddNode.inTensorIds = {INTERMEDIATE_ATTN_DENSE_OUT, INTERMIDATE_MLP_OUT};
    mlpResidualAddNode.outTensorIds = {INTERMEDIATE_RES_OUT};

    atb::infer::ElewiseParam finalResidualAddParam;
    finalResidualAddParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CreateOperation(finalResidualAddParam, &finalResidualAddNode.operation);
    finalResidualAddNode.inTensorIds = {INTERMEDIATE_RES_OUT, IN_HIDDEN_STATES};
    finalResidualAddNode.outTensorIds = {OUT_LAYER_OUT};

    opGraph.nodes.resize(nodeId);

    opGraph.inferShapeFunc = [&](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(IN_HIDDEN_STATES);
        return atb::NO_ERROR;
    };
    atb::CreateOperation(opGraph, operation);
    return atb::NO_ERROR;
}

LayerPrallelFlashAttentionBinder::LayerPrallelFlashAttentionBinder() {}

LayerPrallelFlashAttentionBinder::~LayerPrallelFlashAttentionBinder() {}

void LayerPrallelFlashAttentionBinder::ParseParam(const nlohmann::json &paramJson)
{
    tokenOffset_.clear();
    for (auto item : paramJson["tokenOffset"]) {
        tokenOffset_.push_back(item.get<int>());
    }

    seqLen_.clear();
    for (auto item : paramJson["seqLen"]) {
        seqLen_.push_back(item.get<int>());
    }
    layerId_ = paramJson["layerId"].get<int>();
}

void LayerPrallelFlashAttentionBinder::BindTensor(atb::VariantPack &variantPack)
{
    const uint32_t tokenOffsetTensorId = IN_TOKENOFFSET;
    const uint32_t seqLenTensorId = IN_SEQLEN;
    const uint32_t layerIdTensorId = IN_LAYERID;
    variantPack.inTensors.at(tokenOffsetTensorId).hostData = tokenOffset_.data();
    variantPack.inTensors.at(seqLenTensorId).hostData = seqLen_.data();
    variantPack.inTensors.at(layerIdTensorId).hostData = &layerId_;
}

} // namespace falcon_40b
} // namespace atb_speed