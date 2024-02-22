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
    IN_WEIGHT_0,  // gate weight or gate up weight
    IN_SCALE_0,
    IN_OFFSET_0,
    IN_DESCALE_0,
    IN_WEIGHT_1,  // up weight
    IN_SCALE_1,
    IN_OFFSET_1,
    IN_DESCALE_1,
    IN_WEIGHT_2,  // down weight
    IN_SCALE_2,
    IN_OFFSET_2,
    IN_DESCALE_2,
    OUT_RESULT,
};

static const uint64_t IN_TENSOR_COUNT = 13;
static const uint64_t OUT_TENSOR_COUNT = 1;

template <class T>
atb::Status CreateMlpSwiGLU(const MlpSwiGLUParam &param, atb::Operation **operation, T config)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = config.INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(config.NODE_COUNT);
    opGraph.name = param.isPack ? "MlpSwiGLUPack" : "MlpSwiGLUNoPack";

    size_t nodeId = 0;

    if (param.isPack) {
        atb::Node &linearGateUpNode = opGraph.nodes.at(nodeId++);
        atb_speed::common::FusionLinearParam gateUpLinearParam = param.gateUpLinearParam;
        FusionLinear(gateUpLinearParam, &linearGateUpNode.operation);
        linearGateUpNode.inTensorIds = {
            MlpTensorIdx::IN_INPUT,
            MlpTensorIdx::IN_WEIGHT_0,
            MlpTensorIdx::IN_SCALE_0,
            MlpTensorIdx::IN_OFFSET_0,
            MlpTensorIdx::IN_DESCALE_0
        };
        linearGateUpNode.outTensorIds = {config.INTERMIDATE_GATE_UP_OUT};
    } else {
        atb::Node &linearGateNode = opGraph.nodes.at(nodeId++);
        atb_speed::common::FusionLinearParam gateUpLinearParam = param.gateUpLinearParam;
        FusionLinear(gateUpLinearParam, &linearGateNode.operation);
        linearGateNode.inTensorIds = {
            MlpTensorIdx::IN_INPUT,
            MlpTensorIdx::IN_WEIGHT_0,
            MlpTensorIdx::IN_SCALE_0,
            MlpTensorIdx::IN_OFFSET_0,
            MlpTensorIdx::IN_DESCALE_0
        };
        linearGateNode.outTensorIds = {config.INTERMIDATE_GATE_OUT};

        atb::Node &linearUpNode = opGraph.nodes.at(nodeId++);
        FusionLinear(gateUpLinearParam, &linearUpNode.operation);
        linearUpNode.inTensorIds = {
            MlpTensorIdx::IN_INPUT,
            MlpTensorIdx::IN_WEIGHT_1,
            MlpTensorIdx::IN_SCALE_1,
            MlpTensorIdx::IN_OFFSET_1,
            MlpTensorIdx::IN_DESCALE_1
        };
        linearUpNode.outTensorIds = {config.INTERMIDATE_UP_OUT};

        atb::Node &concatNode = opGraph.nodes.at(nodeId++);
        atb::infer::ConcatParam concatParam;
        CREATE_OPERATION(concatParam, &concatNode.operation);
        concatNode.inTensorIds = {config.INTERMIDATE_GATE_OUT, config.INTERMIDATE_UP_OUT};
        concatNode.outTensorIds = {config.INTERMIDATE_GATE_UP_OUT};
    }

    atb::Node &activationNode = opGraph.nodes.at(nodeId++);
    atb::infer::ActivationParam activationParam;
    activationParam.activationType = atb::infer::ActivationType::ACTIVATION_SWIGLU_FORWARD;
    activationParam.dim = -1;
    CREATE_OPERATION(activationParam, &activationNode.operation);
    activationNode.inTensorIds = {config.INTERMIDATE_GATE_UP_OUT};
    activationNode.outTensorIds = {config.INTERMIDATE_SWISH_OUT};

    atb::Node &linearDownNode = opGraph.nodes.at(nodeId++);
    atb_speed::common::LinearParallelParam downLinearParallelParam = param.downLinearParallelParam;
    LinearParallel(downLinearParallelParam, &linearDownNode.operation);
    linearDownNode.inTensorIds = {
        config.INTERMIDATE_SWISH_OUT,
        MlpTensorIdx::IN_WEIGHT_2,
        MlpTensorIdx::IN_SCALE_2,
        MlpTensorIdx::IN_OFFSET_2,
        MlpTensorIdx::IN_DESCALE_2
    };
    linearDownNode.outTensorIds = {MlpTensorIdx::OUT_RESULT};

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        if (inTensorDescs.at(0).dtype == ACL_INT8) {
            outTensorDescs.at(0).dtype = ACL_FLOAT16;
        }
        return atb::NO_ERROR;
    };

    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}

class MlpNoPackConfig {
public:
    uint64_t INTERMEDIATE_TENSOR_COUNT = 4;
    uint64_t NODE_COUNT = 5;
    enum MlpNoPackTensorIdx : uint32_t {
        INTERMIDATE_SWISH_OUT = MlpTensorIdx::OUT_RESULT + 1,
        INTERMIDATE_GATE_UP_OUT,
        INTERMIDATE_GATE_OUT,
        INTERMIDATE_UP_OUT
    };
};

class MlpPackConfig {
public:
    uint64_t INTERMEDIATE_TENSOR_COUNT = 2;
    uint64_t NODE_COUNT = 3;
    enum MlpPackTensorIdx : uint32_t {
        INTERMIDATE_SWISH_OUT = MlpTensorIdx::OUT_RESULT + 1,
        INTERMIDATE_GATE_UP_OUT,
        INTERMIDATE_GATE_OUT,   // no usage
        INTERMIDATE_UP_OUT      // no usage
    };
};

atb::Status MlpSwiGLU(const MlpSwiGLUParam &param_, atb::Operation **operation)
{
    if (param_.isPack) {
        MlpPackConfig mlpPackConfig;
        return CreateMlpSwiGLU(param_, operation, mlpPackConfig);
    } else {
        MlpNoPackConfig mlpNoPackConfig;
        return CreateMlpSwiGLU(param_, operation, mlpNoPackConfig);
    }
}

} // namespace common
} // namespace atb_speed