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

static const uint64_t IN_TENSOR_COUNT = 11;
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
        NormLinearTensorIdx::IN_OFFSET, NormLinearTensorIdx::IN_DESCALE, NormLinearTensorIdx::IN_BIAS
    };
    linearNode.outTensorIds = {NormLinearTensorIdx::OUT_LINEAR};

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(IN_RESIDUAL_INPUT);
        outTensorDescs.at(1) = inTensorDescs.at(IN_INPUT);
        auto outDimSize = outTensorDescs.at(1).shape.dimNum;
        outTensorDescs.at(1).shape.dims[outDimSize - 1] = param.fusionLinearParam.quantType == LINEAR_W8A16_QUANT \
            ? inTensorDescs.at(IN_LINEAR_WEIGHT).shape.dims[1] : inTensorDescs.at(IN_LINEAR_WEIGHT).shape.dims[0];
        return atb::NO_ERROR;
    };

    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
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