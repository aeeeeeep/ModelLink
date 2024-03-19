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

// static const uint64_t IN_TENSOR_COUNT = 10;
// static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 1;
static const uint64_t SKIP_NORM_INTERMEDIATE_TENSOR_COUNT = 0;
static const uint64_t NODE_COUNT = 2;
static const uint64_t SKIP_NORM_NODE_COUNT = 1;
static const uint64_t FUSION_NORM_GAMMA_DIM_SIZE = 2;

// enum NormLinearTensorIdx : uint32_t {
//     IN_INPUT = 0,
//     IN_NORM_WEIGHT,
//     IN_NORM_BIAS,
//     IN_NORM_NEW_WEIGHT,
//     IN_NORM_NEW_BIAS,
//     IN_LINEAR_WEIGHT,
//     IN_SCALE,
//     IN_OFFSET,
//     IN_DESCALE,
//     IN_BIAS,
//     OUT_LINEAR,
//     INTERMEDIATE_NORM,
// };

void unsqueezeFusionNorm(const atb::Dims &oldShape, atb::Dims &newShape)
{
    newShape.dimNum = FUSION_NORM_GAMMA_DIM_SIZE;
    newShape.dims[0] = 1;
    newShape.dims[1] = oldShape.dims[0];
}

template <class T, typename NormParamType>
atb::Status CreateNormLinear(const NormLinearParam<NormParamType> &param, atb::Operation **operation, T conifg)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = config.IN_TENSOR_COUNT;
    opGraph.outTensorNum = config.OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = param.skipNorm ? SKIP_NORM_INTERMEDIATE_TENSOR_COUNT : INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(param.skipNorm ? SKIP_NORM_NODE_COUNT : NODE_COUNT);
    opGraph.name = "NormLinear";

    size_t nodeId = 0;

    if (!param.skipNorm) {
        atb::Node &normNode = opGraph.nodes.at(nodeId++);
        if (param.fusionLinearParam.quantType == atb_speed::common::LinearQuantType::NORM_QUANT_LINEAR_DEQUANT) {  // W8A8
            if (param.useFusionNorm) {
                CREATE_OPERATION(param.normQuantParamType, &normNode.operation);
                normNode.inTensorIds = {
                    config.IN_INPUT, config.IN_RESIDUAL_ADD,
                    param.isAntiOutlier ? config.IN_NORM_NEW_WEIGHT : config.IN_NORM_WEIGHT,
                    param.isAntiOutlier ? config.IN_NORM_NEW_BIAS : config.IN_NORM_BIAS,
                    config.IN_SCALE, config.IN_OFFSET
                };
                normNode.outTensorIds = {config.INTERMEDIATE_NORM, config.OUT_RESIDUAL_ADD};
            } else {
                CREATE_OPERATION(param.normQuantParamType, &normNode.operation);
                normNode.inTensorIds = {
                    config.IN_INPUT,
                    param.isAntiOutlier ? config.IN_NORM_NEW_WEIGHT : config.IN_NORM_WEIGHT,
                    param.isAntiOutlier ? config.IN_NORM_NEW_BIAS : config.IN_NORM_BIAS,
                    config.IN_SCALE, config.IN_OFFSET
                };
                normNode.outTensorIds = {config.INTERMEDIATE_NORM};
            }
        } else if (param.normHasBias) {  // FP
            CREATE_OPERATION(param.normParamType, &normNode.operation);
            normNode.inTensorIds = {config.IN_INPUT, config.IN_NORM_WEIGHT, config.IN_NORM_BIAS};
            normNode.outTensorIds = {config.INTERMEDIATE_NORM};
        } else if (param.useFusionNorm) { // FP
            CREATE_OPERATION(param.normParamType, &normNode.operation);
            normNode.inTensorIds = {config.IN_INPUT, config.IN_RESIDUAL_ADD, config.IN_NORM_WEIGHT};
            normNode.outTensorIds = {config.INTERMEDIATE_NORM, config.OUT_RESIDUAL_ADD};
            normNode.inTensorReshapeFuncs.resize(normNode.inTensorIds.size());
            normNode.inTensorReshapeFuncs.at(2) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
                unsqueezeFusionNorm(oldShape, newShape);
            };
        } else {  // FP
            CREATE_OPERATION(param.normParamType, &normNode.operation);
            normNode.inTensorIds = {config.IN_INPUT, config.IN_NORM_WEIGHT};
            normNode.outTensorIds = {config.INTERMEDIATE_NORM};
        }
    }

    atb::Node &linearNode = opGraph.nodes.at(nodeId++);
    atb_speed::common::FusionLinearParam linearParam = param.fusionLinearParam;
    FusionLinear(linearParam, &linearNode.operation);
    linearNode.inTensorIds = {
        param.skipNorm ? config.IN_INPUT : config.INTERMEDIATE_NORM,
        config.IN_LINEAR_WEIGHT, config.IN_SCALE,
        config.IN_OFFSET, config.IN_DESCALE, config.IN_BIAS
    };
    linearNode.outTensorIds = {config.OUT_LINEAR};

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
        if (param.useFusionNorm) {
            outTensorDescs.at(0).shape.dims[outDimSize - 1] = param.fusionLinearParam.quantType == W8A16 \
                ? inTensorDescs.at(6).shape.dims[1] : inTensorDescs.at(6).shape.dims[0];
            outTensorDescs.at(1) = inTensorDescs.at(1);
        } else {
            outTensorDescs.at(0).shape.dims[outDimSize - 1] = param.fusionLinearParam.quantType == W8A16 \
                ? inTensorDescs.at(5).shape.dims[1] : inTensorDescs.at(5).shape.dims[0];
        }
        return atb::NO_ERROR;
    };

    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}

class NormLinearConfig{
public:
    uint64_t IN_TENSOR_COUNT = 10;
    uint64_t OUT_TENSOR_COUNT = 1;
    enum NormLinearTensorIdx : uint32_t {
        IN_INPUT = 0,
        IN_NORM_WEIGHT,
        IN_NORM_BIAS,
        IN_NORM_NEW_WEIGHT,
        IN_NORM_NEW_BIAS,
        IN_LINEAR_WEIGHT,
        IN_SCALE,
        IN_OFFSET,
        IN_DESCALE,
        IN_BIAS,
        OUT_LINEAR,
        INTERMEDIATE_NORM,
        IN_RESIDUAL_ADD,
        OUT_RESIDUAL_ADD,
    };
};

class NormLinearFusionConfig{
public:
    uint64_t IN_TENSOR_COUNT = 12;
    uint64_t OUT_TENSOR_COUNT = 2;
    enum NormLinearFusionTensorIdx : uint32_t {
        IN_INPUT = 0,
        IN_RESIDUAL_ADD,
        IN_NORM_WEIGHT,
        IN_NORM_BIAS,
        IN_NORM_NEW_WEIGHT,
        IN_NORM_NEW_BIAS,
        IN_LINEAR_WEIGHT,
        IN_SCALE,
        IN_OFFSET,
        IN_DESCALE,
        IN_BIAS,
        OUT_LINEAR,
        OUT_RESIDUAL_ADD,
        INTERMEDIATE_NORM,
    };
};

template <typename NormParamType>
atb::Status NormLinear(const NormLinearParam<NormParamType> &param, atb::Operation **operation) {
    if (param.useFusionNorm) {
        NormLinearFusionConfig normLinearFusionConfig;
        return CreateNormLinear(param, operation, normLinearFusionConfig);
    } else {
        NormLinearConfig normLinearConfig;
        return CreateNormLinear(param, operation, normLinearConfig);
    }
}

template atb::Status NormLinear(const NormLinearParam<atb::infer::RmsNormParam> &param, atb::Operation **operation);

template atb::Status NormLinear(const NormLinearParam<atb::infer::LayerNormParam> &param, atb::Operation **operation);

} // namespace common
} // namespace atb_speed