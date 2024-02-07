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
#include "layers/mlp.h"

namespace atb_speed {
namespace falcon_7b {
enum FALCON7BLayerInTensorId : int {
    // weight 0->
    // input_norm
    IN_NORM_WEIGHT = 0,
    IN_NORM_BIAS,

    // mixqkv
    IN_MIXED_QKV_LINEAR_WEIGHT,

    // mlp
    IN_MLP_UP_WEIGHT,
    IN_MLP_DOWN_WEIGHT,

    // dense
    IN_DENSE_WEIGHT,

    // input 6->
    IN_HIDDEN_STATES,
    IN_COS_EMBED,
    IN_SIN_EMBED,
    IN_ATTENTION_MASK,
    IN_CACHE_K,
    IN_CACHE_V,
    IN_TOKEN_OFFSET,
    IN_SEQ_LEN,
    IN_LAYER_ID,

    IN_TENSOR_MAX
};

enum FALCON7BLayerOutTensorId : int {
    // out 15->
    OUT_LAYER_OUT = IN_TENSOR_MAX,
    OUT_TENSOR_MAX
};

enum FALCON7BLayerInternalTensorId : int {
    // 16 ->
    INTERNAL_INPUT_NORM_OUT = OUT_TENSOR_MAX,
    INTERNAL_MIXED_QKV_LINEAR_OUT,
    INTERNAL_QLAYER,
    INTERNAL_KLAYER,
    INTERNAL_VLAYER,
    INTERNAL_POSITION_EMBED_Q,
    INTERNAL_POSITION_EMBED_K,
    INTERNAL_ATTN_OUT,
    INTERNAL_MLP_OUT,
    INTERNAL_RES_OUT,
    INTERNAL_DENSE_B_OUT,
    INTERNAL_DENSE_OUT,
    INTERNAL_TENSOR_MAX
};
static const uint64_t IN_TENSOR_COUNT = IN_TENSOR_MAX;
static const uint64_t OUT_TENSOR_COUNT = OUT_TENSOR_MAX - IN_TENSOR_MAX;
static const uint64_t INTERNAL_TENSOR_COUNT = INTERNAL_TENSOR_MAX - OUT_TENSOR_MAX;
static const uint64_t NODE_COUNT_MAX = 20;

atb::Status FusionLayerOperation(const LayerFusionParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERNAL_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT_MAX);
    opGraph.name = "FalconFusionLayer";
    const int padHead = 3;
    size_t nodeId = 0;
    ATB_LOG(INFO) << "FusionLayerOperation headNum:" <<
                      param.headNum << ", kvHeadNum:" << param.kvHeadNum << ",hiddenSize" << param.hiddenSize;
    
    std::shared_ptr<int64_t> batchNumPtr = std::make_shared<int64_t>(0);

    // [bsz,seq_len,hidden_size]
    atb::Node &inputNormNode = opGraph.nodes.at(nodeId++);
    atb::infer::LayerNormParam inputNormParam;
    inputNormParam.layerType = atb::infer::LayerNormParam::LAYER_NORM_NORM;
    inputNormParam.normParam.epsilon = param.layerNormEps;
    inputNormParam.normParam.beginNormAxis = 2;
    inputNormParam.normParam.beginParamsAxis = 1;
    CreateOperation(inputNormParam, &inputNormNode.operation);
    inputNormNode.inTensorIds = {IN_HIDDEN_STATES, IN_NORM_WEIGHT, IN_NORM_BIAS};
    inputNormNode.outTensorIds = {INTERNAL_INPUT_NORM_OUT};

    // [bsz,seq_len,hidden_size]
    atb::Node &mixedQKVLinearNode = opGraph.nodes.at(nodeId++);
    atb::infer::LinearParam fusedQKVParam;
    fusedQKVParam.transposeB = false;
    fusedQKVParam.hasBias = false;
    CreateOperation(fusedQKVParam, &mixedQKVLinearNode.operation);
    mixedQKVLinearNode.inTensorIds = {INTERNAL_INPUT_NORM_OUT, IN_MIXED_QKV_LINEAR_WEIGHT};
    mixedQKVLinearNode.outTensorIds = {INTERNAL_MIXED_QKV_LINEAR_OUT};

    // q -> [bs, seq_len, q_head, head_size]
    atb::Node &sliceQNode = opGraph.nodes.at(nodeId++);
    atb::infer::SliceParam sliceQNodeParam;
    sliceQNodeParam.offsets = {0, 0, 0, 0};
    sliceQNodeParam.size = {-1, -1, param.headNum, -1};
    CreateOperation(sliceQNodeParam, &sliceQNode.operation);
    sliceQNode.inTensorIds = {INTERNAL_MIXED_QKV_LINEAR_OUT};
    sliceQNode.outTensorIds = {INTERNAL_QLAYER};
    sliceQNode.inTensorReshapeFuncs.resize(sliceQNode.inTensorIds.size());
    sliceQNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 4;
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = oldShape.dims[1];
        newShape.dims[2] = param.headNum + 2 * param.kvHeadNum + padHead;
        newShape.dims[3] = param.hiddenSize;
        };

    // kv -> [bs, seq_len, 2 * kv_head, head_size]
    atb::Node &sliceKNode = opGraph.nodes.at(nodeId++);
    atb::infer::SliceParam sliceKNodeParam;
    sliceKNodeParam.offsets = {0, 0, param.headNum, 0};
    sliceKNodeParam.size = {
        -1, -1, param.kvHeadNum, -1};
    CreateOperation(sliceKNodeParam, &sliceKNode.operation);
    sliceKNode.inTensorIds = {INTERNAL_MIXED_QKV_LINEAR_OUT};
    sliceKNode.outTensorIds = {INTERNAL_KLAYER};
    sliceKNode.inTensorReshapeFuncs.resize(sliceKNode.inTensorIds.size());
    sliceKNode.inTensorReshapeFuncs[0] = sliceQNode.inTensorReshapeFuncs[0];

    // kv -> [bs, seq_len, 2 * kv_head, head_size]
    atb::Node &sliceVNode = opGraph.nodes.at(nodeId++);
    atb::infer::SliceParam sliceVNodeParam;
    sliceVNodeParam.offsets = {0, 0, param.headNum + param.kvHeadNum, 0};
    sliceVNodeParam.size = {
        -1, -1, param.kvHeadNum, -1};
    CreateOperation(sliceVNodeParam, &sliceVNode.operation);
    sliceVNode.inTensorIds = {INTERNAL_MIXED_QKV_LINEAR_OUT};
    sliceVNode.outTensorIds = {INTERNAL_VLAYER};
    sliceVNode.inTensorReshapeFuncs.resize(sliceVNode.inTensorIds.size());
    sliceVNode.inTensorReshapeFuncs[0] = sliceQNode.inTensorReshapeFuncs[0];

    atb::Node &ropeNode = opGraph.nodes.at(nodeId++);
    atb::infer::RopeParam ropeParam;
    ropeParam.rotaryCoeff = 2;
    CreateOperation(ropeParam, &ropeNode.operation);
    ropeNode.inTensorIds = {
        INTERNAL_QLAYER, // [bs, seq_len, q_head, head_size]
        INTERNAL_KLAYER, // [bs, seq_len, kv_head, head_size]
        IN_COS_EMBED,    // [bs, seq_len, head_size]
        IN_SIN_EMBED,    // [bs, seq_len, head_size]
        IN_SEQ_LEN       // [bs]
        };
    ropeNode.outTensorIds = {INTERNAL_POSITION_EMBED_Q, INTERNAL_POSITION_EMBED_K};
    ropeNode.inTensorReshapeFuncs.resize(ropeNode.inTensorIds.size());
    ropeNode.inTensorReshapeFuncs[2] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2;
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
        newShape.dims[1] = oldShape.dims[2];
        };
    ropeNode.inTensorReshapeFuncs[3] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2;
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
        newShape.dims[1] = oldShape.dims[2];
        };
    
    atb::Node &selfAttentionFusionNode = opGraph.nodes.at(nodeId++);
    atb::infer::SelfAttentionParam selfAttentionParam;
    selfAttentionParam.headDim = param.hiddenSize;
    selfAttentionParam.headNum = param.headNum;
    selfAttentionParam.qScale = param.preScale;
    selfAttentionParam.qkScale = param.postScale;
    selfAttentionParam.kvHeadNum = param.kvHeadNum;
    CreateOperation(selfAttentionParam, &selfAttentionFusionNode.operation);
    selfAttentionFusionNode.inTensorIds = {INTERNAL_POSITION_EMBED_Q,  // [bs, seq_len, q_head, head_size]
                                            INTERNAL_POSITION_EMBED_K, // [bs, seq_len, kv_head, head_size]
                                            INTERNAL_VLAYER,           // [bs, seq_len, kv_head, head_size]
                                            IN_CACHE_K,
                                            IN_CACHE_V,
                                            IN_ATTENTION_MASK,
                                            IN_TOKEN_OFFSET,
                                            IN_SEQ_LEN,
                                            IN_LAYER_ID};
    selfAttentionFusionNode.outTensorIds = {INTERNAL_ATTN_OUT};

    // [bsz,seq_len,hidden_size]
    atb::Node &finalDenseNode = opGraph.nodes.at(nodeId++);
    atb::infer::LinearParam denseParam;
    denseParam.transposeB = false;
    denseParam.hasBias = false;
    CreateOperation(denseParam, &finalDenseNode.operation);
    finalDenseNode.inTensorIds = {INTERNAL_ATTN_OUT, IN_DENSE_WEIGHT};
    finalDenseNode.outTensorIds = {INTERNAL_DENSE_OUT};

    atb::Node &sliceDenseNode = opGraph.nodes.at(nodeId++);
    atb::infer::SliceParam sliceDenseParam;
    sliceDenseParam.offsets = {0, 0, 0};
    sliceDenseParam.size = { -1, -1, param.headNum * param.hiddenSize };
    CreateOperation(sliceDenseParam, &sliceDenseNode.operation);
    sliceDenseNode.inTensorIds = {INTERNAL_DENSE_OUT};
    sliceDenseNode.outTensorIds = {INTERNAL_DENSE_B_OUT};

    atb::Node &mlpNode = opGraph.nodes.at(nodeId++);
    atb_speed::common::MlpParam mlpParam;
    mlpParam.activationType = atb::infer::ActivationType::ACTIVATION_GELU;
    atb_speed::common::MlpLayer(mlpParam, &mlpNode.operation);
    mlpNode.inTensorIds = {INTERNAL_INPUT_NORM_OUT, IN_MLP_UP_WEIGHT, IN_MLP_DOWN_WEIGHT};
    mlpNode.outTensorIds = {INTERNAL_MLP_OUT};

    atb::Node &mlpResidualAddNode = opGraph.nodes.at(nodeId++);
    atb::infer::ElewiseParam mlpResidualAddParam;
    mlpResidualAddParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CreateOperation(mlpResidualAddParam, &mlpResidualAddNode.operation);
    mlpResidualAddNode.inTensorIds = {INTERNAL_DENSE_B_OUT, INTERNAL_MLP_OUT};
    mlpResidualAddNode.outTensorIds = {INTERNAL_RES_OUT};

    atb::Node &finalResidualAddNode = opGraph.nodes.at(nodeId++);
    atb::infer::ElewiseParam finalResidualAddParam;
    finalResidualAddParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CreateOperation(finalResidualAddParam, &finalResidualAddNode.operation);
    finalResidualAddNode.inTensorIds = {INTERNAL_RES_OUT, IN_HIDDEN_STATES};
    finalResidualAddNode.outTensorIds = {OUT_LAYER_OUT};

    opGraph.nodes.resize(nodeId);

    opGraph.inferShapeFunc = [&](
        const atb::SVector<atb::TensorDesc> &inTensorDescs,
        atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(IN_HIDDEN_STATES);
        return atb::NO_ERROR;
    };
    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}

LayerFusionBinder::LayerFusionBinder() {}

LayerFusionBinder::~LayerFusionBinder() {}

void LayerFusionBinder::ParseParam(const nlohmann::json &paramJson)
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

    ATB_LOG(INFO) << "FusionLayerOperation ParseParam tokenOffset:" <<
                      tokenOffset_ << ", seqLen:" << seqLen_ << ",layerId_";
}

void LayerFusionBinder::BindTensor(atb::VariantPack &variantPack)
{
    const uint32_t tokenOffsetTensorId = 12;
    const uint32_t seqLenTensorId = 13;
    const uint32_t layerIdTensorId = 14;
    variantPack.inTensors.at(tokenOffsetTensorId).hostData = tokenOffset_.data();
    variantPack.inTensors.at(seqLenTensorId).hostData = seqLen_.data();
    variantPack.inTensors.at(layerIdTensorId).hostData = &layerId_;
}
} // namespace falcon_7b
} // namespace atb_speed