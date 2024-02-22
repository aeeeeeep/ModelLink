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
#include "mlp.h"

namespace atb_speed {
namespace glm130b {
enum MlpTensorId : int {
    IN_HIDDEN_STATES_ID = 0,
    IN_DENSE_H_TO_4H_WEIGHT_ID,
    IN_DENSE_H_TO_4H_BIAS_ID,
    IN_DENSE_4H_TO_H_WEIGHT_ID,
    IN_DENSE_4H_TO_H_BIAS_ID,
    OUT_MLP_OUTPUT_ID,
    INTERMEDIATE_DENSE_H_TO_4H_OUTPUT_ID,
    INTERMEDIATE_SPLIT_OUTPUTA_ID,
    INTERMEDIATE_SPLIT_OUTPUTB_ID,
    INTERMEDIATE_GELU_OUTPUT_ID,
    INTERMEDIATE_GEGLU_OUTPUT_ID
};

static const uint64_t IN_TENSOR_COUNT = 5;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 5;
static const uint64_t NODE_COUNT = 5;

atb::Status CreateMlp(const MlpParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    atb::Node &linearNode = opGraph.nodes.at(nodeId++);
    atb::Node &splitNode = opGraph.nodes.at(nodeId++);
    atb::Node &geluNode = opGraph.nodes.at(nodeId++);
    atb::Node &mulNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpLinearParallelNode = opGraph.nodes.at(nodeId++);

    atb::infer::LinearParam linearParam;
    CreateOperation(linearParam, &linearNode.operation);
    linearNode.inTensorIds = {IN_HIDDEN_STATES_ID, IN_DENSE_H_TO_4H_WEIGHT_ID, IN_DENSE_H_TO_4H_BIAS_ID};
    linearNode.outTensorIds = {
        INTERMEDIATE_DENSE_H_TO_4H_OUTPUT_ID}; // [bs, seq_len, hidden_size * 16 / 3 / world_size]

    atb::infer::SplitParam splitParam;
    splitParam.splitDim = 2; // 2: 在第三维上进行切分
    splitParam.splitNum = 2; // 2: 进行二等分
    CreateOperation(splitParam, &splitNode.operation);
    splitNode.inTensorIds = {INTERMEDIATE_DENSE_H_TO_4H_OUTPUT_ID};
    splitNode.outTensorIds = {INTERMEDIATE_SPLIT_OUTPUTA_ID,
                              INTERMEDIATE_SPLIT_OUTPUTB_ID}; // [bs, seq_len, hidden_size * 8 / 3 / world_size]

    atb::infer::ActivationParam activationParam;
    activationParam.activationType = atb::infer::ActivationType::ACTIVATION_GELU;
    CreateOperation(activationParam, &geluNode.operation);
    geluNode.inTensorIds = {INTERMEDIATE_SPLIT_OUTPUTB_ID};
    geluNode.outTensorIds = {INTERMEDIATE_GELU_OUTPUT_ID};

    atb::infer::ElewiseParam elewiseParam;
    elewiseParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    CreateOperation(elewiseParam, &mulNode.operation);
    mulNode.inTensorIds = {INTERMEDIATE_SPLIT_OUTPUTA_ID, INTERMEDIATE_GELU_OUTPUT_ID};
    mulNode.outTensorIds = {INTERMEDIATE_GEGLU_OUTPUT_ID};

    atb::infer::LinearParallelParam linearParallelParam;
    linearParallelParam.transWeight = false;
    linearParallelParam.rank = param.rank;
    linearParallelParam.rankSize = param.rankSize;
    linearParallelParam.rankRoot = 0;
    linearParallelParam.bias = "yes";
    linearParallelParam.parallelType = "RowParallel";
    linearParallelParam.backend = param.backend;
    CreateOperation(linearParallelParam, &mlpLinearParallelNode.operation);
    mlpLinearParallelNode.inTensorIds = {INTERMEDIATE_GEGLU_OUTPUT_ID, IN_DENSE_4H_TO_H_WEIGHT_ID,
                                         IN_DENSE_4H_TO_H_BIAS_ID};
    mlpLinearParallelNode.outTensorIds = {OUT_MLP_OUTPUT_ID};

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        outTensorDescs.at(0).shape.dimNum = 3; // 3表示输出维度
        outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0]; // 0, 0: 设置第一个张量第一维长度, bs
        outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1]; // 1, 1: 设置第一个张量第二维长度, seq_len
        outTensorDescs.at(0).shape.dims[2] = inTensorDescs.at(0).shape.dims[2]; // 2, 2: 设置第一个张量第三维长度, hidden_size
        return atb::NO_ERROR;
    };

    atb::CreateOperation(opGraph, operation);
    return atb::NO_ERROR;
}
} // namespace glm130b
} // namespace atb_speed