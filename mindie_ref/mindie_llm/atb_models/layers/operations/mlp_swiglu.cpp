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
#include "layers/operations/mlp_swiglu.h"

namespace atb_speed {
namespace common {

enum MlpTensorIdx : uint32_t {
    IN_INPUT = 0,
    IN_NORM_WEIGHT,
    IN_NORM_BIAS,
    IN_NORM_NEW_WEIGHT,
    IN_NORM_NEW_BIAS,
    IN_WEIGHT_0,  // gate weight or gate up weight or up only weight
    IN_SCALE_0,
    IN_OFFSET_0,
    IN_DESCALE_0,
    IN_BIAS_0,
    IN_WEIGHT_1,  // up weight
    IN_SCALE_1,
    IN_OFFSET_1,
    IN_DESCALE_1,
    IN_BIAS_1,
    IN_WEIGHT_2,  // down weight
    IN_SCALE_2,
    IN_OFFSET_2,
    IN_DESCALE_2,
    IN_BIAS_2,
    OUT_RESULT,
    INTERMIDATE_SWISH_OUT,
    INTERMIDATE_GATE_UP_OUT,
    INTERMIDATE_GATE_OUT,
    INTERMIDATE_UP_OUT
};

static const uint64_t IN_TENSOR_COUNT = 20;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t NO_PACK_INTERMEDIATE_TENSOR_COUNT = 4;
static const uint64_t NO_PACK_NODE_COUNT = 5;
static const uint64_t PACK_INTERMEDIATE_TENSOR_COUNT = 2;
static const uint64_t PACK_NODE_COUNT = 3;

template <typename NormParamType>
atb::Status MlpSwiGLU(const MlpParam<NormParamType> &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = param.mlpPackType == atb_speed::common::GATE_UP_WEIGHT_NO_PACK \
        ? NO_PACK_INTERMEDIATE_TENSOR_COUNT : PACK_INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(param.mlpPackType == atb_speed::common::GATE_UP_WEIGHT_NO_PACK ? NO_PACK_NODE_COUNT : PACK_NODE_COUNT);
    opGraph.name = param.mlpPackType == atb_speed::common::GATE_UP_WEIGHT_NO_PACK ? "MlpSwiGLUNoPack" : "MlpSwiGLUPack";

    size_t nodeId = 0;

    if (param.mlpPackType != atb_speed::common::GATE_UP_WEIGHT_NO_PACK) { // Gate Up weight权重合并或者没有Gate weight
        atb::Node &normLinearGateUpNode = opGraph.nodes.at(nodeId++);
        atb_speed::common::NormLinearParam<NormParamType> gateUpNormLinearParam;
        gateUpNormLinearParam.isAntiOutlier = param.isAntiOutlier;
        if (param.packQuantType == atb_speed::common::ALL_W8A16) {
            gateUpNormLinearParam.fusionLinearParam.quantType = W8A16;
        } else {
            gateUpNormLinearParam.fusionLinearParam.quantType \
                = param.layerLinearQuantType[4] == atb_speed::common::LinearType::FP ? NO_QUANT : NORM_QUANT_LINEAR_DEQUANT;
        }
        gateUpNormLinearParam.fusionLinearParam.isBF16 = param.isBF16;
        gateUpNormLinearParam.fusionLinearParam.hasBias = param.gateUpHasBias;
        gateUpNormLinearParam.normParamType = param.normParamType;
        gateUpNormLinearParam.normQuantParamType = param.normQuantParamType;
        NormLinear<NormParamType>(gateUpNormLinearParam, &normLinearGateUpNode.operation);
        normLinearGateUpNode.inTensorIds = {
            MlpTensorIdx::IN_INPUT,
            MlpTensorIdx::IN_NORM_WEIGHT,
            MlpTensorIdx::IN_NORM_BIAS,
            MlpTensorIdx::IN_NORM_NEW_WEIGHT,
            MlpTensorIdx::IN_NORM_NEW_BIAS,
            MlpTensorIdx::IN_WEIGHT_0,
            MlpTensorIdx::IN_SCALE_0,
            MlpTensorIdx::IN_OFFSET_0,
            MlpTensorIdx::IN_DESCALE_0,
            MlpTensorIdx::IN_BIAS_0
        };
        normLinearGateUpNode.outTensorIds = {MlpTensorIdx::INTERMIDATE_GATE_UP_OUT};
    } else {
        atb::Node &normLinearGateNode = opGraph.nodes.at(nodeId++);
        atb_speed::common::NormLinearParam<NormParamType> gateNormLinearParam;
        gateNormLinearParam.isAntiOutlier = param.isAntiOutlier;
        if (param.packQuantType == atb_speed::common::ALL_W8A16) {
            gateNormLinearParam.fusionLinearParam.quantType = W8A16;
        } else {
            gateNormLinearParam.fusionLinearParam.quantType \
                = param.layerLinearQuantType[4] == atb_speed::common::LinearType::FP ? NO_QUANT : NORM_QUANT_LINEAR_DEQUANT;
        }
        gateNormLinearParam.fusionLinearParam.isBF16 = param.isBF16;
        gateNormLinearParam.fusionLinearParam.hasBias = param.gateUpHasBias;
        gateNormLinearParam.normParamType = param.normParamType;
        gateNormLinearParam.normQuantParamType = param.normQuantParamType;
        NormLinear<NormParamType>(gateNormLinearParam, &normLinearGateNode.operation);
        normLinearGateNode.inTensorIds = {
            MlpTensorIdx::IN_INPUT,
            MlpTensorIdx::IN_NORM_WEIGHT,
            MlpTensorIdx::IN_NORM_BIAS,
            MlpTensorIdx::IN_NORM_NEW_WEIGHT,
            MlpTensorIdx::IN_NORM_NEW_BIAS,
            MlpTensorIdx::IN_WEIGHT_0,
            MlpTensorIdx::IN_SCALE_0,
            MlpTensorIdx::IN_OFFSET_0,
            MlpTensorIdx::IN_DESCALE_0,
            MlpTensorIdx::IN_BIAS_0
        };
        normLinearGateNode.outTensorIds = {MlpTensorIdx::INTERMIDATE_GATE_OUT};

        atb::Node &normLinearUpNode = opGraph.nodes.at(nodeId++);
        atb_speed::common::NormLinearParam<NormParamType> upNormLinearParam;
        upNormLinearParam.isAntiOutlier = param.isAntiOutlier;
        if (param.packQuantType == atb_speed::common::ALL_W8A16) {
            upNormLinearParam.fusionLinearParam.quantType = W8A16;
        } else {
            upNormLinearParam.fusionLinearParam.quantType \
                = param.layerLinearQuantType[5] == atb_speed::common::LinearType::FP ? NO_QUANT : NORM_QUANT_LINEAR_DEQUANT;
        }
        upNormLinearParam.fusionLinearParam.isBF16 = param.isBF16;
        upNormLinearParam.fusionLinearParam.hasBias = param.gateUpHasBias;
        upNormLinearParam.normParamType = param.normParamType;
        upNormLinearParam.normQuantParamType = param.normQuantParamType;
        NormLinear<NormParamType>(upNormLinearParam, &normLinearUpNode.operation);
        normLinearUpNode.inTensorIds = {
            MlpTensorIdx::IN_INPUT,
            MlpTensorIdx::IN_NORM_WEIGHT,
            MlpTensorIdx::IN_NORM_BIAS,
            MlpTensorIdx::IN_NORM_NEW_WEIGHT,
            MlpTensorIdx::IN_NORM_NEW_BIAS,
            MlpTensorIdx::IN_WEIGHT_1,
            MlpTensorIdx::IN_SCALE_1,
            MlpTensorIdx::IN_OFFSET_1,
            MlpTensorIdx::IN_DESCALE_1,
            MlpTensorIdx::IN_BIAS_1
        };
        normLinearUpNode.outTensorIds = {MlpTensorIdx::INTERMIDATE_UP_OUT};

        atb::Node &concatNode = opGraph.nodes.at(nodeId++);
        atb::infer::ConcatParam concatParam;
        concatParam.concatDim = -1;
        CREATE_OPERATION(concatParam, &concatNode.operation);
        concatNode.inTensorIds = {MlpTensorIdx::INTERMIDATE_GATE_OUT, MlpTensorIdx::INTERMIDATE_UP_OUT};
        concatNode.outTensorIds = {MlpTensorIdx::INTERMIDATE_GATE_UP_OUT};
    }

    atb::Node &activationNode = opGraph.nodes.at(nodeId++);
    CREATE_OPERATION(param.activationParam, &activationNode.operation);
    activationNode.inTensorIds = {MlpTensorIdx::INTERMIDATE_GATE_UP_OUT};
    activationNode.outTensorIds = {MlpTensorIdx::INTERMIDATE_SWISH_OUT};

    atb::Node &linearDownNode = opGraph.nodes.at(nodeId++);
    atb_speed::common::LinearParallelParam downLinearParallelParam;
    downLinearParallelParam.parallelType = atb_speed::common::ROW_PARALLEL;
    if (param.packQuantType == atb_speed::common::ALL_W8A16) {
        downLinearParallelParam.fusionLinearParam.quantType = W8A16;
    } else {
        downLinearParallelParam.fusionLinearParam.quantType \
            = param.layerLinearQuantType[6] == atb_speed::common::LinearType::FP ? \
            atb_speed::common::LinearQuantType::NO_QUANT : atb_speed::common::LinearQuantType::LINEAR_QUANT;
    }
    downLinearParallelParam.biasAfterSync = param.downLinearTensorParallelInfo.worldSize > 1 \
        && downLinearParallelParam.fusionLinearParam.quantType == atb_speed::common::LinearQuantType::NO_QUANT \
        && param.downHasBias;
    downLinearParallelParam.fusionLinearParam.hasBias = param.downHasBias && !downLinearParallelParam.biasAfterSync;
    downLinearParallelParam.fusionLinearParam.isBF16 = param.isBF16;
    downLinearParallelParam.tensorParallelInfo = param.downLinearTensorParallelInfo;
    downLinearParallelParam.supportLcoc = param.supportLcoc;
    LinearParallel(downLinearParallelParam, &linearDownNode.operation);
    linearDownNode.inTensorIds = {
        MlpTensorIdx::INTERMIDATE_SWISH_OUT,
        MlpTensorIdx::IN_WEIGHT_2,
        MlpTensorIdx::IN_SCALE_2,
        MlpTensorIdx::IN_OFFSET_2,
        MlpTensorIdx::IN_DESCALE_2,
        MlpTensorIdx::IN_BIAS_2
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

template atb::Status MlpSwiGLU(const MlpParam<atb::infer::RmsNormParam> &param, atb::Operation **operation);

} // namespace common
} // namespace atb_speed