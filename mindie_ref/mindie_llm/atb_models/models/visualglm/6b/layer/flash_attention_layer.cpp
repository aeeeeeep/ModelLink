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

namespace atb_speed {

enum Chatglm6BLayerDecoderFlashAttentionTensorId {
    IN_HIDDENSTATES_ID = 0,
    IN_NORMWEIGHT_ID,
    IN_NORMBIAS_ID,
    IN_QKVMIXEDWEIGHT_ID,
    IN_QKVMIXEDBIAS_ID,
    IN_SELFOUTLINEARWEIGHT_ID,
    IN_SELFOUTLINEARBIAS_ID,
    IN_SELFOUTNORMWEIGHT_ID,
    IN_SELFOUTNORMBIAS_ID,
    IN_FFNLINEARWEIGHT_ID,
    IN_FFNLINEARBIAS_ID,
    IN_FFNOUTLINEARWEIGHT_ID,
    IN_FFNOUTLINEARBIAS_ID,
    IN_POSITIONIDS_ID,
    IN_COSTABLE_ID,
    IN_SINTABLE_ID,
    IN_ATTENTIONMASK_ID,
    IN_CACHEK_ID,
    IN_CACHEV_ID,
    IN_TOKENOFFSET_ID,
    IN_SEQLEN_ID,
    IN_LAYERID_ID,
    OUT_LAYEROUT_ID,
    INTERMEDIATE_INPUTNORMOUT_ID,
    INTERMEDIATE_MIXEDLINEAROUTQKV_ID,
    INTERMEDIATE_POSITIONEMBEDQ_ID,
    INTERMEDIATE_POSITIONEMBEDK_ID,
    INTERMEDIATE_VALUE_ID,
    INTERMEDIATE_SELFOUT_ID,
    INTERMEDIATE_SELFLINEAROUT_ID,
    INTERMEDIATE_SELFRESIDUALOUT_ID,
    INTERMEDIATE_SELFADDOUT_ID,
    INTERMEDIATE_SELFNORMOUT_ID,
    INTERMEDIATE_FFNOUT,
    INTERMEDIATE_FFNLINEAROUT_ID,
    INTERMEDIATE_FFNRESIDUALOUT_ID,

    INTERMEDIATE_QLAYER_ID,
    INTERMEDIATE_KLAYER_ID,
    INTERMEDIATE_POSITION_IDS0_ID,
    INTERMEDIATE_POSITION_IDS1_ID,
    INTERMEDIATE_POSITION_IDS_TRANSPOSE0_ID,
    INTERMEDIATE_POSITION_IDS_TRANSPOSE1_ID,
    INTERMEDIATE_COS0_ID,
    INTERMEDIATE_COS1_ID,
    INTERMEDIATE_SIN0_ID,
    INTERMEDIATE_SIN1_ID,
    INTERMEDIATE_COSSUM_ID,
    INTERMEDIATE_SINSUM_ID
};

static const uint64_t IN_TENSOR_COUNT = 22;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 25;
static const uint64_t NODE_COUNT = 22;

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
    newShape.dimNum = oldShape.dimNum - 1;
    newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
    newShape.dims[1] = oldShape.dims[2]; // 2: 设置新张量第二维的长度
}

void RopeQKReshapeFunc(const atb::Dims &oldShape, atb::Dims &newShape)
{
    newShape.dimNum = oldShape.dimNum - 2;
    newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
    newShape.dims[1] = oldShape.dims[2] * oldShape.dims[3]; // 2, 3: 设置新张量第二维的长度
}

atb::Status CreateGlm6BLayerDecoderFlashAttentionOperation(const Glm6BLayerDecoderFlashAttentionParam &param,
    atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.name = "LayerDecoderFlashAttentionOperation";
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    auto &inputNormNode = opGraph.nodes.at(nodeId++);
    auto &mixdQkvLinearNode = opGraph.nodes.at(nodeId++);

    auto &splitQKVNode = opGraph.nodes.at(nodeId++);
    auto &splitPositionIdsNode = opGraph.nodes.at(nodeId++);
    auto &transposePositionIds0Node = opGraph.nodes.at(nodeId++);
    auto &transposePositionIds1Node = opGraph.nodes.at(nodeId++);
    auto &embeddingCos0Node = opGraph.nodes.at(nodeId++);
    auto &embeddingCos1Node = opGraph.nodes.at(nodeId++);
    auto &embeddingSin0Node = opGraph.nodes.at(nodeId++);
    auto &embeddingSin1Node = opGraph.nodes.at(nodeId++);
    auto &concateCosNode = opGraph.nodes.at(nodeId++);
    auto &concateSinNode = opGraph.nodes.at(nodeId++);
    auto &ropeNode = opGraph.nodes.at(nodeId++);

    auto &selfAttentionFusionNode = opGraph.nodes.at(nodeId++);
    auto &selfOutLinearNode = opGraph.nodes.at(nodeId++);
    auto &selfResidualNode = opGraph.nodes.at(nodeId++);
    auto &selfAddNode = opGraph.nodes.at(nodeId++);
    auto &selfNormNode = opGraph.nodes.at(nodeId++);
    auto &ffnNode = opGraph.nodes.at(nodeId++);
    auto &ffnLinearNode = opGraph.nodes.at(nodeId++);
    auto &ffnResidualNode = opGraph.nodes.at(nodeId++);
    auto &ffnAddNode = opGraph.nodes.at(nodeId++);

    std::shared_ptr<int64_t> batchDimPtr = std::make_shared<int64_t>(0);

    atb::infer::LayerNormParam inputNormParam;
    inputNormParam.layerType = atb::infer::LayerNormParam::LAYER_NORM_NORM;
    inputNormParam.normParam.epsilon = param.layerNormEps;
    inputNormParam.normParam.beginNormAxis = 2;
    inputNormParam.normParam.beginParamsAxis = 1;
    CREATE_OPERATION(inputNormParam, &inputNormNode.operation);
    inputNormNode.inTensorIds = {IN_HIDDENSTATES_ID, IN_NORMWEIGHT_ID, IN_NORMBIAS_ID};
    inputNormNode.outTensorIds = {INTERMEDIATE_INPUTNORMOUT_ID};

    atb::infer::LinearParam mixdQkvLinearParam;
    CREATE_OPERATION(mixdQkvLinearParam, &mixdQkvLinearNode.operation);
    mixdQkvLinearNode.inTensorIds = {INTERMEDIATE_INPUTNORMOUT_ID, IN_QKVMIXEDWEIGHT_ID, IN_QKVMIXEDBIAS_ID};
    mixdQkvLinearNode.outTensorIds = {INTERMEDIATE_MIXEDLINEAROUTQKV_ID};

    atb::infer::SplitParam splitQKVParam;
    splitQKVParam.splitDim = 3; // 3: 在第四维上进行切分
    splitQKVParam.splitNum = 3; // 3: 进行三等分
    CREATE_OPERATION(splitQKVParam, &splitQKVNode.operation);
    splitQKVNode.inTensorIds = {INTERMEDIATE_MIXEDLINEAROUTQKV_ID};
    splitQKVNode.outTensorIds = {INTERMEDIATE_QLAYER_ID, INTERMEDIATE_KLAYER_ID, INTERMEDIATE_VALUE_ID};
    splitQKVNode.inTensorReshapeFuncs.resize(splitQKVNode.inTensorIds.size());
    splitQKVNode.inTensorReshapeFuncs.at(0) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = oldShape.dimNum + 1;
        *batchDimPtr = oldShape.dims[0];
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = oldShape.dims[1];
        newShape.dims[2] = param.headNum;                    // 2: 设置张量第三维的大小
        newShape.dims[3] = oldShape.dims[2] / param.headNum; // 2, 3: 设置张量第四维的大小
    };

    atb::infer::SplitParam splitPositionIdsParam;
    splitPositionIdsParam.splitDim = 1; // 1: 在第二维上进行切分
    splitPositionIdsParam.splitNum = 2; // 2: 进行二等分
    CREATE_OPERATION(splitPositionIdsParam, &splitPositionIdsNode.operation);
    splitPositionIdsNode.inTensorIds = {IN_POSITIONIDS_ID};
    splitPositionIdsNode.outTensorIds = {INTERMEDIATE_POSITION_IDS0_ID, INTERMEDIATE_POSITION_IDS1_ID};

    atb::infer::TransposeParam transposePositionIds0Param;
    transposePositionIds0Param.perm = {1, 0}; // 1, 0: 对前两维进行转置
    CREATE_OPERATION(transposePositionIds0Param, &transposePositionIds0Node.operation);
    transposePositionIds0Node.inTensorIds = {INTERMEDIATE_POSITION_IDS0_ID};
    transposePositionIds0Node.outTensorIds = {INTERMEDIATE_POSITION_IDS_TRANSPOSE0_ID};
    transposePositionIds0Node.inTensorReshapeFuncs.resize(transposePositionIds0Node.inTensorIds.size());
    transposePositionIds0Node.inTensorReshapeFuncs.at(0) = &Squeeze1;

    atb::infer::TransposeParam transposePositionIds1Param;
    transposePositionIds1Param.perm = {1, 0}; // 1, 0: 对前两维进行转置
    CREATE_OPERATION(transposePositionIds1Param, &transposePositionIds1Node.operation);
    transposePositionIds1Node.inTensorIds = {INTERMEDIATE_POSITION_IDS1_ID};
    transposePositionIds1Node.outTensorIds = {INTERMEDIATE_POSITION_IDS_TRANSPOSE1_ID};
    transposePositionIds1Node.inTensorReshapeFuncs.resize(transposePositionIds1Node.inTensorIds.size());
    transposePositionIds1Node.inTensorReshapeFuncs.at(0) = &Squeeze1;

    atb::infer::GatherParam embeddingCos0Param;
    CREATE_OPERATION(embeddingCos0Param, &embeddingCos0Node.operation);
    embeddingCos0Node.inTensorIds = {IN_COSTABLE_ID, INTERMEDIATE_POSITION_IDS_TRANSPOSE0_ID};
    embeddingCos0Node.outTensorIds = {INTERMEDIATE_COS0_ID};
    embeddingCos0Node.inTensorReshapeFuncs.resize(embeddingCos0Node.inTensorIds.size());
    embeddingCos0Node.inTensorReshapeFuncs.at(0) = &Squeeze1;

    atb::infer::GatherParam embeddingCos1Param;
    CREATE_OPERATION(embeddingCos1Param, &embeddingCos1Node.operation);
    embeddingCos1Node.inTensorIds = {IN_COSTABLE_ID, INTERMEDIATE_POSITION_IDS_TRANSPOSE1_ID};
    embeddingCos1Node.outTensorIds = {INTERMEDIATE_COS1_ID};
    embeddingCos1Node.inTensorReshapeFuncs.resize(embeddingCos1Node.inTensorIds.size());
    embeddingCos1Node.inTensorReshapeFuncs.at(0) = &Squeeze1;

    atb::infer::GatherParam embeddingSin0Param;
    CREATE_OPERATION(embeddingSin0Param, &embeddingSin0Node.operation);
    embeddingSin0Node.inTensorIds = {IN_SINTABLE_ID, INTERMEDIATE_POSITION_IDS_TRANSPOSE0_ID};
    embeddingSin0Node.outTensorIds = {INTERMEDIATE_SIN0_ID};
    embeddingSin0Node.inTensorReshapeFuncs.resize(embeddingSin0Node.inTensorIds.size());
    embeddingSin0Node.inTensorReshapeFuncs.at(0) = &Squeeze1;

    atb::infer::GatherParam embeddingSin1Param;
    CREATE_OPERATION(embeddingSin1Param, &embeddingSin1Node.operation);
    embeddingSin1Node.inTensorIds = {IN_SINTABLE_ID, INTERMEDIATE_POSITION_IDS_TRANSPOSE1_ID};
    embeddingSin1Node.outTensorIds = {INTERMEDIATE_SIN1_ID};
    embeddingSin1Node.inTensorReshapeFuncs.resize(embeddingSin1Node.inTensorIds.size());
    embeddingSin1Node.inTensorReshapeFuncs.at(0) = &Squeeze1;

    atb::infer::ConcatParam concateCosParam;
    concateCosParam.concatDim = 2; // 2: 在第三维上进行concat
    CREATE_OPERATION(concateCosParam, &concateCosNode.operation);
    concateCosNode.inTensorIds = {INTERMEDIATE_COS0_ID, INTERMEDIATE_COS1_ID};
    concateCosNode.outTensorIds = {INTERMEDIATE_COSSUM_ID};

    atb::infer::ConcatParam concateSinParam;
    concateSinParam.concatDim = 2; // 2: 在第三维上进行concat
    CREATE_OPERATION(concateSinParam, &concateSinNode.operation);
    concateSinNode.inTensorIds = {INTERMEDIATE_SIN0_ID, INTERMEDIATE_SIN1_ID};
    concateSinNode.outTensorIds = {INTERMEDIATE_SINSUM_ID};

    atb::infer::RopeParam ropeParam;
    ropeParam.rotaryCoeff = 4; // 4: 设置旋转系数
    CREATE_OPERATION(ropeParam, &ropeNode.operation);
    ropeNode.inTensorIds = {INTERMEDIATE_QLAYER_ID, INTERMEDIATE_KLAYER_ID, INTERMEDIATE_COSSUM_ID,
                            INTERMEDIATE_SINSUM_ID, IN_SEQLEN_ID};
    ropeNode.outTensorIds = {INTERMEDIATE_POSITIONEMBEDQ_ID, INTERMEDIATE_POSITIONEMBEDK_ID};
    ropeNode.inTensorReshapeFuncs = {&RopeQKReshapeFunc, &RopeQKReshapeFunc, &RopeCosSinReshapeFunc,
                                     &RopeCosSinReshapeFunc};

    atb::infer::SelfAttentionParam selfAttentionParam;
    selfAttentionParam.headNum = param.headNum;
    selfAttentionParam.qScale = param.qScale;
    selfAttentionParam.qkScale = param.qkScale;
    selfAttentionParam.maskType = atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_NORM;
    CREATE_OPERATION(selfAttentionParam, &selfAttentionFusionNode.operation);
    selfAttentionFusionNode.inTensorIds = {INTERMEDIATE_POSITIONEMBEDQ_ID,
                                           INTERMEDIATE_POSITIONEMBEDK_ID,
                                           INTERMEDIATE_VALUE_ID,
                                           IN_CACHEK_ID,
                                           IN_CACHEV_ID,
                                           IN_ATTENTIONMASK_ID,
                                           IN_TOKENOFFSET_ID,
                                           IN_SEQLEN_ID,
                                           IN_LAYERID_ID};
    selfAttentionFusionNode.outTensorIds = {INTERMEDIATE_SELFOUT_ID};
    selfAttentionFusionNode.inTensorReshapeFuncs.resize(selfAttentionFusionNode.inTensorIds.size());
    selfAttentionFusionNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 4;
        newShape.dims[0] = *batchDimPtr;
        newShape.dims[1] = oldShape.dims[0] / *batchDimPtr;
        newShape.dims[2] = param.headNum;                    // 2: 设置张量第三维的大小
        newShape.dims[3] = oldShape.dims[1] / param.headNum; // 2, 3: 设置张量第四维的大小
    };
    selfAttentionFusionNode.inTensorReshapeFuncs[1] = selfAttentionFusionNode.inTensorReshapeFuncs[0];
    selfAttentionFusionNode.inTensorReshapeFuncs[2] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 4;
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = oldShape.dims[1];
        newShape.dims[2] = param.headNum;                    // 2: 设置张量第三维的大小
        newShape.dims[3] = oldShape.dims[2] * oldShape.dims[3] / param.headNum; // 2, 3: 设置张量第四维的大小
    };

    atb::infer::LinearParam selfOutLinearParam;
    CREATE_OPERATION(selfOutLinearParam, &selfOutLinearNode.operation);
    selfOutLinearNode.inTensorIds = {INTERMEDIATE_SELFOUT_ID, IN_SELFOUTLINEARWEIGHT_ID, IN_SELFOUTLINEARBIAS_ID};
    selfOutLinearNode.outTensorIds = {INTERMEDIATE_SELFLINEAROUT_ID};

    atb::infer::ElewiseParam selfResidualParam;
    selfResidualParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MULS;
    selfResidualParam.mulsParam.varAttr = param.residualAddScale;
    CREATE_OPERATION(selfResidualParam, &selfResidualNode.operation);
    selfResidualNode.inTensorIds = {INTERMEDIATE_INPUTNORMOUT_ID};
    selfResidualNode.outTensorIds = {INTERMEDIATE_SELFRESIDUALOUT_ID};

    atb::infer::ElewiseParam selfAddParam;
    selfAddParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CREATE_OPERATION(selfAddParam, &selfAddNode.operation);
    selfAddNode.inTensorIds = {INTERMEDIATE_SELFRESIDUALOUT_ID, INTERMEDIATE_SELFLINEAROUT_ID};
    selfAddNode.outTensorIds = {INTERMEDIATE_SELFADDOUT_ID};

    atb::infer::LayerNormParam selfNormParam;
    selfNormParam.layerType = atb::infer::LayerNormParam::LAYER_NORM_NORM;
    selfNormParam.normParam.epsilon = param.layerNormEps;
    selfNormParam.normParam.beginNormAxis = 2;
    selfNormParam.normParam.beginParamsAxis = 1;
    CREATE_OPERATION(selfNormParam, &selfNormNode.operation);
    selfNormNode.inTensorIds = {INTERMEDIATE_SELFADDOUT_ID, IN_SELFOUTNORMWEIGHT_ID, IN_SELFOUTNORMBIAS_ID};
    selfNormNode.outTensorIds = {INTERMEDIATE_SELFNORMOUT_ID};

    atb::infer::LinearActivationParam ffnParam;
    CREATE_OPERATION(ffnParam, &ffnNode.operation);
    ffnNode.inTensorIds = {INTERMEDIATE_SELFNORMOUT_ID, IN_FFNLINEARWEIGHT_ID, IN_FFNLINEARBIAS_ID};
    ffnNode.outTensorIds = {INTERMEDIATE_FFNOUT};

    atb::infer::LinearParam ffnLinearParam;
    CREATE_OPERATION(ffnLinearParam, &ffnLinearNode.operation);
    ffnLinearNode.inTensorIds = {INTERMEDIATE_FFNOUT, IN_FFNOUTLINEARWEIGHT_ID, IN_FFNOUTLINEARBIAS_ID};
    ffnLinearNode.outTensorIds = {INTERMEDIATE_FFNLINEAROUT_ID};

    atb::infer::ElewiseParam ffnResidualParam;
    ffnResidualParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MULS;
    ffnResidualParam.mulsParam.varAttr = param.residualAddScale;
    CREATE_OPERATION(ffnResidualParam, &ffnResidualNode.operation);
    ffnResidualNode.inTensorIds = {INTERMEDIATE_SELFNORMOUT_ID};
    ffnResidualNode.outTensorIds = {INTERMEDIATE_FFNRESIDUALOUT_ID};

    atb::infer::ElewiseParam ffnAddParam;
    ffnAddParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CREATE_OPERATION(ffnAddParam, &ffnAddNode.operation);
    ffnAddNode.inTensorIds = {INTERMEDIATE_FFNRESIDUALOUT_ID, INTERMEDIATE_FFNLINEAROUT_ID};
    ffnAddNode.outTensorIds = {OUT_LAYEROUT_ID};

    opGraph.inferShapeFunc = [&](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
    };

    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}

} // namespace atb_speed