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
#include "layers/plugin_op/w8a16_operation.h"
#include "layers/plugin_op/w8a16_bias_operation.h"

namespace atb_speed {
namespace common {

enum LinearTensorIdx : uint32_t {
    IN_INPUT = 0,
    IN_WEIGHT,
    IN_SCALE,    // Quant所需权重
    IN_OFFSET,   // Quant所需权重
    IN_DESCALE,  // Quant所需权重
    IN_BIAS,
    IN_COMPRESS_IDX,
    OUT_LINEAR,
    INTERMIDATE_INPUT  // 仅LINEAR_QUANT场景下使用
};

static const uint64_t IN_TENSOR_COUNT = 7;
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
    opGraph.internalTensorNum = param.quantType == LINEAR_W8A8_QUANT || param.quantType == LINEAR_W8A8_SC_QUANT \
        ? QUANT_DEQUANT_INTERMEDIATE_TENSOR_COUNT : DEFAULT_INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(
        param.quantType == LINEAR_W8A8_QUANT || param.quantType == LINEAR_W8A8_SC_QUANT \
        ? QUANT_DEQUANT_NODE_COUNT : DEFAULT_NODE_COUNT
    );
    opGraph.name = param.quantType == LINEAR_NO_QUANT ? "LinearNoQuant" : \
        param.quantType == LINEAR_W8A8_DEQUANT || param.quantType == LINEAR_W8A8_SC_DEQUANT ? "LinearDequantOnly" : \
        param.quantType == LINEAR_W8A16_QUANT ? "LinearW8A16" : "LinearQuant";

    size_t nodeId = 0;

    if (param.quantType == LINEAR_W8A8_QUANT || param.quantType == LINEAR_W8A8_SC_QUANT) {
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
    if (param.quantType != LINEAR_NO_QUANT && !param.isBF16) {
        linearParam.outDataType = ACL_FLOAT16;
    }

    if (param.quantType == LINEAR_W8A8_SC_DEQUANT || param.quantType == LINEAR_W8A8_SC_QUANT) {
        atb::infer::LinearSparseParam linearSparseParam;
        linearSparseParam.tilingK = 8;
        linearSparseParam.tilingN = 8;
        CREATE_OPERATION(linearSparseParam, &linearNode.operation);
        linearNode.inTensorIds = {
            param.quantType == LINEAR_W8A8_SC_DEQUANT ? LinearTensorIdx::IN_INPUT : LinearTensorIdx::INTERMIDATE_INPUT,
            LinearTensorIdx::IN_WEIGHT, LinearTensorIdx::IN_BIAS, LinearTensorIdx::IN_DESCALE, LinearTensorIdx::IN_COMPRESS_IDX
        };
    } else if (param.quantType == LINEAR_W8A16_QUANT && param.hasBias) {
        linearNode.operation = new atb_speed::common::W8A16BiasOperation("LinearBiasNode");
        linearNode.inTensorIds = {
            LinearTensorIdx::IN_INPUT, LinearTensorIdx::IN_WEIGHT,
            LinearTensorIdx::IN_SCALE, LinearTensorIdx::IN_OFFSET, LinearTensorIdx::IN_BIAS
        };
    } else if (param.quantType == LINEAR_W8A16_QUANT && !param.hasBias) {
        linearNode.operation = new atb_speed::common::W8A16Operation("LinearNode");
        linearNode.inTensorIds = {LinearTensorIdx::IN_INPUT, LinearTensorIdx::IN_WEIGHT, LinearTensorIdx::IN_SCALE, LinearTensorIdx::IN_OFFSET};
    } else if (param.quantType == LINEAR_NO_QUANT && param.hasBias) {
        linearParam.hasBias = true;
        CREATE_OPERATION(linearParam, &linearNode.operation);
        linearNode.inTensorIds = {LinearTensorIdx::IN_INPUT, LinearTensorIdx::IN_WEIGHT, LinearTensorIdx::IN_BIAS};
    } else if (param.quantType == LINEAR_NO_QUANT && !param.hasBias) {
        linearParam.hasBias = false;
        CREATE_OPERATION(linearParam, &linearNode.operation);
        linearNode.inTensorIds = {LinearTensorIdx::IN_INPUT, LinearTensorIdx::IN_WEIGHT};
    } else {
        linearParam.hasBias = true;
        CREATE_OPERATION(linearParam, &linearNode.operation);
        linearNode.inTensorIds = {
            param.quantType == LINEAR_W8A8_DEQUANT ? LinearTensorIdx::IN_INPUT : LinearTensorIdx::INTERMIDATE_INPUT,
            LinearTensorIdx::IN_WEIGHT, LinearTensorIdx::IN_BIAS, LinearTensorIdx::IN_DESCALE
        };
    }
    linearNode.outTensorIds = {LinearTensorIdx::OUT_LINEAR};

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0).format = inTensorDescs.at(IN_INPUT).format;
        outTensorDescs.at(0).dtype = param.isBF16 ? ACL_BF16 : ACL_FLOAT16;
        outTensorDescs.at(0).shape = inTensorDescs.at(IN_INPUT).shape;
        auto outDimSize = outTensorDescs.at(IN_INPUT).shape.dimNum;
        if (param.quantType == LINEAR_W8A16_QUANT) {
            outTensorDescs.at(0).shape.dims[outDimSize - 1] = inTensorDescs.at(IN_WEIGHT).shape.dims[1];
        } else if (param.quantType == LINEAR_W8A8_SC_DEQUANT || param.quantType == LINEAR_W8A8_SC_QUANT) {
            outTensorDescs.at(0).shape.dims[outDimSize - 1] = inTensorDescs.at(IN_BIAS).shape.dims[0];
        } else {
            outTensorDescs.at(0).shape.dims[outDimSize - 1] = inTensorDescs.at(IN_WEIGHT).shape.dims[0];
        }
        return atb::NO_ERROR;
    };

    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}

} // namespace common
} // namespace atb_speed