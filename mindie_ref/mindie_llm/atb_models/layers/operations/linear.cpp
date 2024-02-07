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
    OUT_LINEAR
};

static const uint64_t IN_TENSOR_COUNT = 5;
static const uint64_t OUT_TENSOR_COUNT = 1;

template <class T>
atb::Status CreateFusionLinear(const FusionLinearParam &param, atb::Operation **operation, T config)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = config.INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(config.NODE_COUNT);
    opGraph.name = param.quantType == NO_QUANT ? "LinearNoQuant" : \
        param.quantType == RMS_NORM_QUANT_LINEAR_DEQUANT ? "LinearDequantOnly" : "LinearQuant";

    size_t nodeId = 0;

    if (param.quantType == LINEAR_QUANT) {
        // quant
        atb::Node &inputQuantNode = opGraph.nodes.at(nodeId++);
        atb::infer::ElewiseParam inputQuantParam;
        inputQuantParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_QUANT_PER_CHANNEL;
        CREATE_OPERATION(inputQuantParam, &inputQuantNode.operation);
        inputQuantNode.inTensorIds = {LinearTensorIdx::IN_INPUT, LinearTensorIdx::IN_SCALE, LinearTensorIdx::IN_OFFSET};
        inputQuantNode.outTensorIds = {config.INTERMIDATE_INPUT};
    }

    if (param.quantType == NO_QUANT) {
        // linear
        atb::Node &linearNode = opGraph.nodes.at(nodeId++);
        atb::infer::LinearParam linearParam;
        linearParam.hasBias = false;
        CREATE_OPERATION(linearParam, &linearNode.operation);
        linearNode.inTensorIds = {LinearTensorIdx::IN_INPUT, LinearTensorIdx::IN_WEIGHT};
        linearNode.outTensorIds = {LinearTensorIdx::OUT_LINEAR};
    } else {
        // linear + dequant
        atb::Node &linearQuantNode = opGraph.nodes.at(nodeId++);
        atb::infer::LinearParam linearQuantParam;
        linearQuantParam.linearType = atb::infer::LinearType::LINEAR_INT8INT8_INT32_FP16;
        linearQuantParam.hasBias = false;
        CREATE_OPERATION(linearQuantParam, &linearQuantNode.operation);
        linearQuantNode.inTensorIds = {
            param.quantType == RMS_NORM_QUANT_LINEAR_DEQUANT ? LinearTensorIdx::IN_INPUT : config.INTERMIDATE_INPUT,
            LinearTensorIdx::IN_WEIGHT, LinearTensorIdx::IN_DESCALE
        };
        linearQuantNode.outTensorIds = {LinearTensorIdx::OUT_LINEAR};
    }

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0).format = inTensorDescs.at(0).format;
        outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
        outTensorDescs.at(0).shape = inTensorDescs.at(0).shape;
        auto outDimSize = outTensorDescs.at(0).shape.dimNum;
        outTensorDescs.at(0).shape.dims[outDimSize - 1] = inTensorDescs.at(1).shape.dims[0];
        return atb::NO_ERROR;
    };

    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}

class LinearNoQuantConfig {
public:
    uint64_t INTERMEDIATE_TENSOR_COUNT = 0;
    uint64_t NODE_COUNT = 1;

    enum LinearNoQuantTensorIdx : uint32_t {
        INTERMIDATE_INPUT = OUT_LINEAR + 1  // no usage
    };
};

class LinearDequantOnlyConfig {
public:
    uint64_t INTERMEDIATE_TENSOR_COUNT = 0;
    uint64_t NODE_COUNT = 1;

    enum LinearDequantOnlyTensorIdx : uint32_t {
        INTERMIDATE_INPUT = OUT_LINEAR + 1  // no usage
    };
};

class LinearQuantConfig {
public:
    uint64_t INTERMEDIATE_TENSOR_COUNT = 1;
    uint64_t NODE_COUNT = 2;

    enum LinearQuantTensorIdx : uint32_t {
        INTERMIDATE_INPUT = OUT_LINEAR + 1,
    };
};

atb::Status FusionLinear(const FusionLinearParam &param_, atb::Operation **operation)
{
    if (param_.quantType == NO_QUANT) {
        LinearNoQuantConfig linearNoQuantConfig;
        return CreateFusionLinear(param_, operation, linearNoQuantConfig);
    } else if (param_.quantType == RMS_NORM_QUANT_LINEAR_DEQUANT) {
        LinearDequantOnlyConfig linearDequantOnlyConfig;
        return CreateFusionLinear(param_, operation, linearDequantOnlyConfig);
    } else if (param_.quantType == LINEAR_QUANT) {
        LinearQuantConfig linearQuantConfig;
        return CreateFusionLinear(param_, operation, linearQuantConfig);
    } else {
        ATB_LOG(ERROR) << "FusionLinear operation doesn't support quantType: " << param_.quantType;
        return atb::ERROR_INVALID_PARAM;
    }
}

} // namespace common
} // namespace atb_speed