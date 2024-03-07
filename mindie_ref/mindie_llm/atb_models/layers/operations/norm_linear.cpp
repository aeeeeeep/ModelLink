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

static const uint64_t IN_TENSOR_COUNT = 10;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 1;
static const uint64_t NODE_COUNT = 2;

enum NormLinearTensorIdx : uint32_t {
    IN_INPUT = 0,
    IN_NORM_WEIGHT,
    IN_NORM_BIAS,
    IN_NORM_NEW_WEIGHT,
    IN_NORM_NEW_BIAS,
    IN_LINEAR_WEIGHT,
    IN_BIAS,
    IN_DESCALE,
    IN_OFFSET,
    IN_SCALE,
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

    atb::Node &normNode = opGraph.nodes.at(nodeId++);
    if (param.fusionLinearParam.quantType == atb_speed::common::LinearQuantType::NORM_QUANT_LINEAR_DEQUANT) {  // W8A8
        CREATE_OPERATION(param.normQuantParamType, &normNode.operation);
        normNode.inTensorIds = {
            NormLinearTensorIdx::IN_INPUT,
            param.isAntiOutlier ? NormLinearTensorIdx::IN_NORM_NEW_WEIGHT : NormLinearTensorIdx::IN_NORM_WEIGHT,
            param.isAntiOutlier ? NormLinearTensorIdx::IN_NORM_NEW_BIAS : NormLinearTensorIdx::IN_NORM_BIAS,
            NormLinearTensorIdx::IN_SCALE, NormLinearTensorIdx::IN_OFFSET
        };
        normNode.outTensorIds = {INTERMEDIATE_NORM};
    } else if (param.fpHasBias) {  // FP
        CREATE_OPERATION(param.normParamType, &normNode.operation);
        normNode.inTensorIds = {NormLinearTensorIdx::IN_INPUT, NormLinearTensorIdx::IN_NORM_WEIGHT, NormLinearTensorIdx::IN_NORM_NEW_BIAS};
        normNode.outTensorIds = {INTERMEDIATE_NORM};
    } else {  // FP
        CREATE_OPERATION(param.normParamType, &normNode.operation);
        normNode.inTensorIds = {NormLinearTensorIdx::IN_INPUT, NormLinearTensorIdx::IN_NORM_WEIGHT};
        normNode.outTensorIds = {INTERMEDIATE_NORM};
    }

    atb::Node &linearNode = opGraph.nodes.at(nodeId++);
    atb_speed::common::FusionLinearParam linearParam = param.fusionLinearParam;
    FusionLinear(linearParam, &linearNode.operation);
    linearNode.inTensorIds = {
        NormLinearTensorIdx::INTERMEDIATE_NORM, NormLinearTensorIdx::IN_LINEAR_WEIGHT, NormLinearTensorIdx::IN_BIAS,
        NormLinearTensorIdx::IN_DESCALE, NormLinearTensorIdx::IN_OFFSET, NormLinearTensorIdx::IN_SCALE,
    };
    linearNode.outTensorIds = {OUT_LINEAR};

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0).format = inTensorDescs.at(0).format;
        if (param.fusionLinearParam.isBF16) {
            outTensorDescs.at(0).dtype = ACL_BF16;
        } else {
            outTensorDescs.at(0).dtype = ACL_FLOAT16;
        }
        outTensorDescs.at(0).shape = inTensorDescs.at(0).shape;
        auto outDimSize = outTensorDescs.at(0).shape.dimNum;
        outTensorDescs.at(0).shape.dims[outDimSize - 1] = inTensorDescs.at(5).shape.dims[0];
        return atb::NO_ERROR;
    };

    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}

template atb::Status NormLinear(const NormLinearParam<atb::infer::RmsNormParam> &param, atb::Operation **operation);

} // namespace common
} // namespace atb_speed