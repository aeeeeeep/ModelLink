/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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
#include "atb_speed/log.h"
#include "models/llama_parallel/operation/rms_norm.h"

namespace atb_speed {
namespace llama_parallel {

enum RmsNormTensorIdx : uint32_t {
    IN_INPUT = 0,
    IN_WEIGHT,
    IN_BETA,  // Quant所需入参
    OUT_NORM,
};

static const uint64_t IN_TENSOR_COUNT = 3;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 0;
static const uint64_t NODE_COUNT = 1;

atb::Status FusionRmsNorm(const FusionRmsNormParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);
    opGraph.name = param.quantType == atb_speed::llama_parallel::RMS_NORM_QUANT_LINEAR_DEQUANT ? "RmsNormQuant" : "RmsNormNoQuant";

    size_t nodeId = 0;

    atb::infer::RmsNormParam rmsNormParam;
    if (param.quantType == atb_speed::llama_parallel::RMS_NORM_QUANT_LINEAR_DEQUANT) {
        atb::Node &rmsNormQuantNode = opGraph.nodes.at(nodeId++);
        rmsNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
        rmsNormParam.normParam.epsilon = param.rmsNormEps;
        rmsNormParam.normParam.quantInputScale = param.quantInputScale;
        rmsNormParam.normParam.quantInputOffset = param.quantInputOffset;
        rmsNormParam.normParam.quantType = atb::infer::QUANT_INT8;
        CreateOperation(rmsNormParam, &rmsNormQuantNode.operation);
        rmsNormQuantNode.inTensorIds = {RmsNormTensorIdx::IN_INPUT, RmsNormTensorIdx::IN_WEIGHT, RmsNormTensorIdx::IN_BETA};
        rmsNormQuantNode.outTensorIds = {RmsNormTensorIdx::OUT_NORM};
    } else {
        atb::Node &rmsNormNode = opGraph.nodes.at(nodeId++);
        rmsNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
        rmsNormParam.normParam.epsilon = param.rmsNormEps;
        CreateOperation(rmsNormParam, &rmsNormNode.operation);
        rmsNormNode.inTensorIds = {RmsNormTensorIdx::IN_INPUT, RmsNormTensorIdx::IN_WEIGHT};
        rmsNormNode.outTensorIds = {RmsNormTensorIdx::OUT_NORM};
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

    atb::CreateOperation(opGraph, operation);
    return atb::NO_ERROR;
}

} // namespace llama_parallel
} // namespace atb_speed