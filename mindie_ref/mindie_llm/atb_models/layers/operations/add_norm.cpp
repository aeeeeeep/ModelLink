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

static const uint64_t IN_TENSOR_COUNT = 8;
static const uint64_t OUT_TENSOR_COUNT = 2;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 0;
static const uint64_t TWO_NODE = 2;
static const uint64_t ONE_NODE = 1;

enum AddNormTensorIdx : uint32_t {
    IN_RESIDUAL_INPUT = 0,
    IN_INPUT,
    IN_NORM_WEIGHT,
    IN_NORM_BIAS,
    IN_NORM_NEW_WEIGHT,
    IN_NORM_NEW_BIAS,
    IN_SCALE,
    IN_OFFSET,
    OUT_NORM,
    OUT_ADD,
};

void unsqueezeNormWeight(const atb::Dims &oldShape, atb::Dims &newShape)
{
    newShape.dimNum = 2;
    newShape.dims[0] = 1;
    newShape.dims[1] = oldShape.dims[0];
}

template <typename NormParamType>
atb::Status AddNorm(const AddNormParam<NormParamType> &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(param.addNormType == ADD_NORM ? TWO_NODE : ONE_NODE);
    if (param.addNormType == NORM_ONLY) {
        opGraph.name = "AddNorm_NORM_ONLY";
    } else if (param.addNormType == FUSION_ADD_NORM) {
        opGraph.name = "AddNorm_FUSION_ADD_NORM";
    } else {
        opGraph.name = "AddNorm_ADD_NORM";
    }

    size_t nodeId = 0;

    ATB_LOG(INFO) << "param.addNormType " << param.addNormType;
    ATB_LOG(INFO) << "param.normQuantType " << param.normQuantType;

    if (param.addNormType == ADD_NORM) {
        atb::Node &addNode = opGraph.nodes.at(nodeId++);
        atb::infer::ElewiseParam addParam;
        addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
        CREATE_OPERATION(addParam, &addNode.operation);
        addNode.inTensorIds = {
            AddNormTensorIdx::IN_RESIDUAL_INPUT,
            AddNormTensorIdx::IN_INPUT,
        };
        addNode.outTensorIds = {AddNormTensorIdx::OUT_ADD};
    }

    if (param.addNormType == ADD_NORM || param.addNormType == NORM_ONLY) {
        atb::Node &normNode = opGraph.nodes.at(nodeId++);
        if (param.normQuantType != NORM_NO_QUANT) {  // W8A8 or W8A8 anti-outlier
            CREATE_OPERATION(param.normQuantParamType, &normNode.operation);
            normNode.inTensorIds = {
                param.addNormType == NORM_ONLY ? AddNormTensorIdx::IN_INPUT : AddNormTensorIdx::OUT_ADD,
                param.normQuantType == NORM_ANTI_OUTLIER_QUANT ? AddNormTensorIdx::IN_NORM_NEW_WEIGHT : AddNormTensorIdx::IN_NORM_WEIGHT,
                param.normQuantType == NORM_ANTI_OUTLIER_QUANT ? AddNormTensorIdx::IN_NORM_NEW_BIAS : AddNormTensorIdx::IN_NORM_BIAS,
                AddNormTensorIdx::IN_SCALE, AddNormTensorIdx::IN_OFFSET
            };
            normNode.outTensorIds = {AddNormTensorIdx::OUT_NORM};
        } else if (param.normHasBias) {  // FP has bias
            CREATE_OPERATION(param.normParamType, &normNode.operation);
            normNode.inTensorIds = {
                param.addNormType == NORM_ONLY ? AddNormTensorIdx::IN_INPUT : AddNormTensorIdx::OUT_ADD,
                AddNormTensorIdx::IN_NORM_WEIGHT, AddNormTensorIdx::IN_NORM_BIAS
            };
            normNode.outTensorIds = {AddNormTensorIdx::OUT_NORM};
        } else {  // FP no bias
            CREATE_OPERATION(param.normParamType, &normNode.operation);
            normNode.inTensorIds = {
                param.addNormType == NORM_ONLY ? AddNormTensorIdx::IN_INPUT : AddNormTensorIdx::OUT_ADD,
                AddNormTensorIdx::IN_NORM_WEIGHT
            };
            normNode.outTensorIds = {AddNormTensorIdx::OUT_NORM};
        }
    }

    if (param.addNormType == FUSION_ADD_NORM) {
        atb::Node &normNode = opGraph.nodes.at(nodeId++);
        if (param.normQuantType != NORM_NO_QUANT) {  // W8A8 or W8A8 anti-outlier
            CREATE_OPERATION(param.normParamType, &normNode.operation);
            normNode.inTensorIds = {
                AddNormTensorIdx::IN_RESIDUAL_INPUT, AddNormTensorIdx::IN_INPUT,
                param.normQuantType == NORM_ANTI_OUTLIER_QUANT ? AddNormTensorIdx::IN_NORM_NEW_WEIGHT : AddNormTensorIdx::IN_NORM_WEIGHT,
                param.normQuantType == NORM_ANTI_OUTLIER_QUANT ? AddNormTensorIdx::IN_NORM_NEW_BIAS : AddNormTensorIdx::IN_NORM_BIAS,
                AddNormTensorIdx::IN_SCALE, AddNormTensorIdx::IN_OFFSET
            };
            normNode.outTensorIds = {AddNormTensorIdx::OUT_NORM, AddNormTensorIdx::OUT_ADD};
        } else if (param.normHasBias) {  // FP has bias
            CREATE_OPERATION(param.normParamType, &normNode.operation);
            normNode.inTensorIds = {
                AddNormTensorIdx::IN_RESIDUAL_INPUT, AddNormTensorIdx::IN_NORM_BIAS,
                AddNormTensorIdx::IN_INPUT, AddNormTensorIdx::IN_NORM_WEIGHT
            };
            normNode.outTensorIds = {AddNormTensorIdx::OUT_NORM, AddNormTensorIdx::OUT_ADD};
        } else {  // FP no bias
            CREATE_OPERATION(param.normParamType, &normNode.operation);
            normNode.inTensorIds = {
                AddNormTensorIdx::IN_RESIDUAL_INPUT, AddNormTensorIdx::IN_INPUT, AddNormTensorIdx::IN_NORM_WEIGHT
            };
            normNode.outTensorIds = {AddNormTensorIdx::OUT_NORM, AddNormTensorIdx::OUT_ADD};
        }
    }

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(IN_INPUT);
        outTensorDescs.at(0).dtype = inTensorDescs.at(IN_NORM_WEIGHT).dtype;
        outTensorDescs.at(1) = inTensorDescs.at(IN_RESIDUAL_INPUT);
        return atb::NO_ERROR;
    };

    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}

template atb::Status AddNorm(const AddNormParam<atb::infer::RmsNormParam> &param, atb::Operation **operation);
template atb::Status AddNorm(const AddNormParam<atb::infer::LayerNormParam> &param, atb::Operation **operation);

} // namespace common
} // namespace atb_speed