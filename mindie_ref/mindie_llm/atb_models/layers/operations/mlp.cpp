/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
#include <atb/atb_infer.h>
#include "layers/operations/mlp.h"

namespace atb_speed {
namespace common {

enum MlpTensorIdx : uint32_t {
    IN_INPUT = 0,
    IN_NORM_WEIGHT,
    IN_NORM_QUANT_WEIGHT,
    IN_NORM_QUANT_BIAS,
    IN_WEIGHT_0,  // gate weight or gate up weight or up only weight
    IN_SCALE_0,
    IN_OFFSET_0,
    IN_DESCALE_0,
    IN_DEOFFSET_0,
    IN_WEIGHT_1,  // up weight
    IN_SCALE_1,
    IN_OFFSET_1,
    IN_DESCALE_1,
    IN_DEOFFSET_1,
    IN_WEIGHT_2,  // down weight
    IN_SCALE_2,
    IN_OFFSET_2,
    IN_DESCALE_2,
    IN_DEOFFSET_2,
    OUT_RESULT,
    INTERMIDATE_GATE_UP_OUT_0,
    INTERMIDATE_SWISH_OUT,
    INTERMIDATE_UP_OUT,
    INTERMIDATE_MUL_OUT,
    INTERMIDATE_GATE_OUT,
};

static const uint64_t IN_TENSOR_COUNT = 19;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t GATE_UP_WEIGHT_PACK_INTERMEDIATE_TENSOR_COUNT = 5;
static const uint64_t GATE_UP_WEIGHT_NO_PACK_INTERMEDIATE_TENSOR_COUNT = 4;
static const uint64_t UP_WEIGHT_ONLY_INTERMEDIATE_TENSOR_COUNT = 2;
static const uint64_t UP_WEIGHT_ONLY_NODE_COUNT = 3;
static const uint64_t GATE_UP_WEIGHT_NODE_COUNT = 5;

template <typename NormParamType>
atb::Status Mlp(const MlpParam<NormParamType> &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    if (param.mlpPackType == MlpPackType::GATE_UP_WEIGHT_PACK) {
        opGraph.internalTensorNum = GATE_UP_WEIGHT_PACK_INTERMEDIATE_TENSOR_COUNT;
        opGraph.name = "MlpGateUpWeightPack";
    } else if (param.mlpPackType == MlpPackType::GATE_UP_WEIGHT_NO_PACK) {
        opGraph.internalTensorNum = GATE_UP_WEIGHT_NO_PACK_INTERMEDIATE_TENSOR_COUNT;
        opGraph.name = "MlpGateUpWeightNoPack";
    } else {
        opGraph.internalTensorNum = UP_WEIGHT_ONLY_INTERMEDIATE_TENSOR_COUNT;
        opGraph.name = "MlpUpWeightOnly";
    }
    opGraph.nodes.resize(param.mlpPackType == MlpPackType::UP_WEIGHT_ONLY ? UP_WEIGHT_ONLY_NODE_COUNT : GATE_UP_WEIGHT_NODE_COUNT);

    size_t nodeId = 0;

    atb::Node &normLinearGateUpNode = opGraph.nodes.at(nodeId++);
    atb_speed::common::NormLinearParam<NormParamType> gateUpNormLinearParam;
    gateUpNormLinearParam.isAnti = param.isAnti;
    gateUpNormLinearParam.fusionLinearParam.quantType \
        = param.layerLinearQuantType[4] == atb_speed::common::LinearType::FP ? NO_QUANT : NORM_QUANT_LINEAR_DEQUANT;
    gateUpNormLinearParam.fusionLinearParam.isBF16 = param.isBF16;
    gateUpNormLinearParam.fusionLinearParam.hasBias = param.hasBias;
    gateUpNormLinearParam.normParamType = param.normParamType;
    gateUpNormLinearParam.normQuantParamType = param.normQuantParamType;
    NormLinear<NormParamType>(gateUpNormLinearParam, &normLinearGateUpNode.operation);
    normLinearGateUpNode.inTensorIds = {
        MlpTensorIdx::IN_INPUT,
        MlpTensorIdx::IN_NORM_WEIGHT,
        MlpTensorIdx::IN_NORM_QUANT_WEIGHT,
        MlpTensorIdx::IN_NORM_QUANT_BIAS,
        MlpTensorIdx::IN_WEIGHT_0,
        MlpTensorIdx::IN_SCALE_0,
        MlpTensorIdx::IN_OFFSET_0,
        MlpTensorIdx::IN_DESCALE_0,
        MlpTensorIdx::IN_DEOFFSET_0,
    };
    normLinearGateUpNode.outTensorIds = {MlpTensorIdx::INTERMIDATE_GATE_UP_OUT_0};

    if (param.mlpPackType == MlpPackType::GATE_UP_WEIGHT_PACK) {
        atb::Node &splitNode = opGraph.nodes.at(nodeId++);
        atb::infer::SplitParam splitParam;
        splitParam.splitDim = -1; // [batchSize, seqLen, 2 * hiddenSize]
        splitParam.splitNum = 2;  // 进行二等分
        CREATE_OPERATION(splitParam, &splitNode.operation);
        splitNode.inTensorIds = {MlpTensorIdx::INTERMIDATE_GATE_UP_OUT_0};
        splitNode.outTensorIds = {MlpTensorIdx::INTERMIDATE_GATE_OUT, MlpTensorIdx::INTERMIDATE_UP_OUT};
    }

    if (param.mlpPackType == MlpPackType::GATE_UP_WEIGHT_NO_PACK) {
        atb::Node &normLinearUpNode = opGraph.nodes.at(nodeId++);
        atb_speed::common::NormLinearParam<NormParamType> upNormLinearParam;
        upNormLinearParam.isAnti = param.isAnti;
        upNormLinearParam.fusionLinearParam.quantType \
            = param.layerLinearQuantType[5] == atb_speed::common::LinearType::FP ? NO_QUANT : NORM_QUANT_LINEAR_DEQUANT;
        upNormLinearParam.fusionLinearParam.isBF16 = param.isBF16;
        upNormLinearParam.fusionLinearParam.hasBias = param.hasBias;
        upNormLinearParam.normParamType = param.normParamType;
        upNormLinearParam.normQuantParamType = param.normQuantParamType;
        NormLinear<NormParamType>(upNormLinearParam, &normLinearUpNode.operation);
        normLinearUpNode.inTensorIds = {
            MlpTensorIdx::IN_INPUT,
            MlpTensorIdx::IN_NORM_WEIGHT,
            MlpTensorIdx::IN_NORM_QUANT_WEIGHT,
            MlpTensorIdx::IN_NORM_QUANT_BIAS,
            MlpTensorIdx::IN_WEIGHT_1,
            MlpTensorIdx::IN_SCALE_1,
            MlpTensorIdx::IN_OFFSET_1,
            MlpTensorIdx::IN_DESCALE_1,
            MlpTensorIdx::IN_DEOFFSET_1
        };
        normLinearUpNode.outTensorIds = {MlpTensorIdx::INTERMIDATE_UP_OUT};
    }

    atb::Node &activationNode = opGraph.nodes.at(nodeId++);
    atb::infer::ActivationParam activationParam;
    activationParam.activationType = atb::infer::ActivationType::ACTIVATION_SWISH;
    CREATE_OPERATION(activationParam, &activationNode.operation);
    activationNode.inTensorIds = {
        param.mlpPackType == MlpPackType::UP_WEIGHT_ONLY ? MlpTensorIdx::INTERMIDATE_GATE_UP_OUT_0 : MlpTensorIdx::INTERMIDATE_GATE_OUT
    };
    activationNode.outTensorIds = {MlpTensorIdx::INTERMIDATE_SWISH_OUT};

    if (param.mlpPackType != MlpPackType::UP_WEIGHT_ONLY) {
        atb::Node &mulNode = opGraph.nodes.at(nodeId++);
        atb::infer::ElewiseParam elewiseParam;
        elewiseParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
        CREATE_OPERATION(elewiseParam, &mulNode.operation);
        mulNode.inTensorIds = {MlpTensorIdx::INTERMIDATE_SWISH_OUT, MlpTensorIdx::INTERMIDATE_UP_OUT};
        mulNode.outTensorIds = {MlpTensorIdx::INTERMIDATE_MUL_OUT};
    }

    atb::Node &linearDownNode = opGraph.nodes.at(nodeId++);
    atb_speed::common::LinearParallelParam downLinearParallelParam;
    downLinearParallelParam.parallelType = atb_speed::common::ROW_PARALLEL;
    downLinearParallelParam.fusionLinearParam.quantType \
        = param.layerLinearQuantType[6] == atb_speed::common::LinearType::FP ? \
        atb_speed::common::LinearQuantType::NO_QUANT : atb_speed::common::LinearQuantType::LINEAR_QUANT;
    downLinearParallelParam.biasAfterSync = param.downLinearTensorParallelInfo.worldSize > 1 \
        && downLinearParallelParam.fusionLinearParam.quantType == atb_speed::common::LinearQuantType::NO_QUANT \
        && param.hasBias;
    downLinearParallelParam.fusionLinearParam.hasBias = param.hasBias && !downLinearParallelParam.biasAfterSync;
    downLinearParallelParam.fusionLinearParam.isBF16 = param.isBF16;
    downLinearParallelParam.tensorParallelInfo = param.downLinearTensorParallelInfo;
    LinearParallel(downLinearParallelParam, &linearDownNode.operation);
    linearDownNode.inTensorIds = {
        param.mlpPackType == MlpPackType::UP_WEIGHT_ONLY ? MlpTensorIdx::INTERMIDATE_SWISH_OUT : MlpTensorIdx::INTERMIDATE_MUL_OUT,
        MlpTensorIdx::IN_WEIGHT_2,
        MlpTensorIdx::IN_SCALE_2,
        MlpTensorIdx::IN_OFFSET_2,
        MlpTensorIdx::IN_DESCALE_2,
        MlpTensorIdx::IN_DEOFFSET_2
    };
    linearDownNode.outTensorIds = {MlpTensorIdx::OUT_RESULT};

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
    };

    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}

template atb::Status Mlp(const MlpParam<atb::infer::RmsNormParam> &param, atb::Operation **operation);

} // namespace common
} // namespace atb_speed