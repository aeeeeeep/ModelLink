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
#include "atb_speed/log.h"
#include "layers/operations/norm_linear.h"

namespace atb_speed {
namespace common {

static const uint64_t IN_TENSOR_COUNT = 12;
static const uint64_t OUT_TENSOR_COUNT = 2;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 1;
static const uint64_t NODE_COUNT = 2;

enum NormLinearTensorIdx : uint32_t {
    IN_RESIDUAL_INPUT = 0,
    IN_INPUT,
    IN_NORM_WEIGHT,
    IN_NORM_BIAS,
    IN_NORM_NEW_WEIGHT,
    IN_NORM_NEW_BIAS,
    IN_LINEAR_WEIGHT,
    IN_SCALE,
    IN_OFFSET,
    IN_DESCALE,
    IN_BIAS,
    IN_COMPRESS_IDX,
    OUT_NEXT_RESIDUAL_IN,
    OUT_LINEAR,
    INTERMEDIATE_NORM,
};

template <typename NormParamType>
atb::Status NormLinear(const NormLinearParam<NormParamType> &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);
    opGraph.name = "NormLinear";

    size_t nodeId = 0;

    atb::Node &addNormNode = opGraph.nodes.at(nodeId++);
    AddNorm(param.addNormParam, &addNormNode.operation);
    addNormNode.inTensorIds = {
        NormLinearTensorIdx::IN_RESIDUAL_INPUT, NormLinearTensorIdx::IN_INPUT,
        NormLinearTensorIdx::IN_NORM_WEIGHT, NormLinearTensorIdx::IN_NORM_BIAS,
        NormLinearTensorIdx::IN_NORM_NEW_WEIGHT, NormLinearTensorIdx::IN_NORM_NEW_BIAS,
        NormLinearTensorIdx::IN_SCALE, NormLinearTensorIdx::IN_OFFSET,
    };
    if (param.nextResidualAddIn == NORM_OUT) {
        addNormNode.outTensorIds = {NormLinearTensorIdx::OUT_NEXT_RESIDUAL_IN, NormLinearTensorIdx::IN_RESIDUAL_INPUT};
    } else {
        addNormNode.outTensorIds = {NormLinearTensorIdx::INTERMEDIATE_NORM, NormLinearTensorIdx::OUT_NEXT_RESIDUAL_IN};
    }

    atb::Node &linearNode = opGraph.nodes.at(nodeId++);
    FusionLinear(param.fusionLinearParam, &linearNode.operation);
    linearNode.inTensorIds = {
        param.nextResidualAddIn == NORM_OUT ? NormLinearTensorIdx::OUT_NEXT_RESIDUAL_IN : NormLinearTensorIdx::INTERMEDIATE_NORM,
        NormLinearTensorIdx::IN_LINEAR_WEIGHT, NormLinearTensorIdx::IN_SCALE,
        NormLinearTensorIdx::IN_OFFSET, NormLinearTensorIdx::IN_DESCALE, NormLinearTensorIdx::IN_BIAS,
        NormLinearTensorIdx::IN_COMPRESS_IDX
    };
    linearNode.outTensorIds = {NormLinearTensorIdx::OUT_LINEAR};

    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}

LinearQuantType GetLinearQuantType(const int &packQuantType, const int &linearType, bool hasNorm)
{
    if (linearType == atb_speed::common::LinearType::FP) {
        return atb_speed::common::LinearQuantType::NO_QUANT;
    } else if (packQuantType == atb_speed::common::ALL_W8A16) {
        return atb_speed::common::LinearQuantType::W8A16;
    } else {
        if (packQuantType == atb_speed::common::ALL_W8A8SC || packQuantType == atb_speed::common::MIX_W8A8SC) {
            if (hasNorm) {
                return atb_speed::common::LinearQuantType::LINEAR_W8A8_SC_DEQUANT;
            } else {
                return atb_speed::common::LinearQuantType::LINEAR_W8A8_SC_QUANT;
            }
        } else {
            if (hasNorm) {
                return atb_speed::common::LinearQuantType::LINEAR_W8A8_DEQUANT;
            } else {
                return atb_speed::common::LinearQuantType::LINEAR_W8A8_QUANT;
            }
        }
    }
}

NormQuantType GetNormQuantType(const int &packQuantType)
{
    if (packQuantType == PackQuantType::ALL_W8A16 || packQuantType == PackQuantType::ALL_FP) {
        return NormQuantType::NORM_NO_QUANT;
    } else if (packQuantType == PackQuantType::ALL_W8A8 || packQuantType == PackQuantType::MIX_W8A8) {
        return NormQuantType::NORM_QUANT;
    } else {
        return NormQuantType::NORM_ANTI_OUTLIER_QUANT;
    }
}

template atb::Status NormLinear(const NormLinearParam<atb::infer::RmsNormParam> &param, atb::Operation **operation);
template atb::Status NormLinear(const NormLinearParam<atb::infer::LayerNormParam> &param, atb::Operation **operation);

} // namespace common
} // namespace atb_speed