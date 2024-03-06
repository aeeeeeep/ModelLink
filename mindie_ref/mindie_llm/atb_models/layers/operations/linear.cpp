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
#include <cmath>
#include <numeric>
#include "atb_speed/log.h"
#include "layers/operations/linear.h"

namespace atb_speed {
namespace common {

enum LinearTensorIdx : uint32_t {
    IN_INPUT = 0,
    IN_WEIGHT,
    IN_SCALE,    // Quant所需权重
    IN_OFFSET,   // Quant所需权重
    IN_DESCALE,  // Quant所需权重
    IN_BIAS,
    OUT_LINEAR,
    INTERMIDATE_INPUT  // 仅LINEAR_QUANT场景下使用
};

static const uint64_t IN_TENSOR_COUNT = 6;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t QUANT_DEQUANT_INTERMEDIATE_TENSOR_COUNT = 1;
static const uint64_t QUANT_DEQUANT_NODE_COUNT = 2;
static const uint64_t DEFAULT_INTERMEDIATE_TENSOR_COUNT = 0;
static const uint64_t DEFAULT_NODE_COUNT = 1;

atb::Status FusionLinear(const FusionLinearParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = param.quantType == LINEAR_QUANT ? QUANT_DEQUANT_INTERMEDIATE_TENSOR_COUNT : DEFAULT_INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(param.quantType == LINEAR_QUANT ? QUANT_DEQUANT_NODE_COUNT : DEFAULT_NODE_COUNT);
    opGraph.name = param.quantType == NO_QUANT ? "LinearNoQuant" : \
        param.quantType == NORM_QUANT_LINEAR_DEQUANT ? "LinearDequantOnly" : "LinearQuant";

    size_t nodeId = 0;

    if (param.quantType == LINEAR_QUANT) {
        // quant
        atb::Node &inputQuantNode = opGraph.nodes.at(nodeId++);
        atb::infer::ElewiseParam inputQuantParam;
        inputQuantParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_QUANT_PER_CHANNEL;
        CREATE_OPERATION(inputQuantParam, &inputQuantNode.operation);
        inputQuantNode.inTensorIds = {LinearTensorIdx::IN_INPUT, LinearTensorIdx::IN_SCALE, LinearTensorIdx::IN_OFFSET};
        inputQuantNode.outTensorIds = {LinearTensorIdx::INTERMIDATE_INPUT};
    }

    atb::Node &linearNode = opGraph.nodes.at(nodeId++);
    atb::infer::LinearParam linearParam;
    if (param.quantType != NO_QUANT && !param.isBF16) {
        linearParam.linearType = atb::infer::LinearType::LINEAR_INT8INT8_INT32_FP16;
    } else if (param.quantType == NO_QUANT && param.isBF16) {
        linearParam.linearType = atb::infer::LinearType::LINEAR_BF16BF16_FP32_BF16;
    }

    if (param.quantType == NO_QUANT && param.hasBias) {
        linearParam.hasBias = true;
        CREATE_OPERATION(linearParam, &linearNode.operation);
        linearNode.inTensorIds = {LinearTensorIdx::IN_INPUT, LinearTensorIdx::IN_WEIGHT, LinearTensorIdx::IN_BIAS};
    } else if (param.quantType == NO_QUANT && !param.hasBias) {
        linearParam.hasBias = false;
        CREATE_OPERATION(linearParam, &linearNode.operation);
        linearNode.inTensorIds = {LinearTensorIdx::IN_INPUT, LinearTensorIdx::IN_WEIGHT};
    } else {
        linearParam.hasBias = true;
        CREATE_OPERATION(linearParam, &linearNode.operation);
        linearNode.inTensorIds = {
            param.quantType == NORM_QUANT_LINEAR_DEQUANT ? LinearTensorIdx::IN_INPUT : LinearTensorIdx::INTERMIDATE_INPUT,
            LinearTensorIdx::IN_WEIGHT, LinearTensorIdx::IN_BIAS, LinearTensorIdx::IN_DESCALE
        };
    }
    linearNode.outTensorIds = {LinearTensorIdx::OUT_LINEAR};

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0).format = inTensorDescs.at(0).format;
        outTensorDescs.at(0).dtype = param.isBF16 ? ACL_BF16 : ACL_FLOAT16;
        outTensorDescs.at(0).shape = inTensorDescs.at(0).shape;
        auto outDimSize = outTensorDescs.at(0).shape.dimNum;
        outTensorDescs.at(0).shape.dims[outDimSize - 1] = inTensorDescs.at(1).shape.dims[0];
        return atb::NO_ERROR;
    };

    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}

} // namespace common
} // namespace atb_speed