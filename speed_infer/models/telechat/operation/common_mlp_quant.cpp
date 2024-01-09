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
#include "common_mlp_quant.h"

namespace atb_speed{
namespace telechat{
enum CommonMlpQuantTensorId {
	IN_HIDDENSTATES_ID = 0,             // [batch, seqLen, hiddenSize], half
	IN_WEIGHT_GATE_ID,                  // [11008, hiddenSize], half
	IN_MLPLINEARDEQSCALEGATE, // quant
	IN_LINEARBIASGATE,  // quant
	IN_WEIGHT_UP_ID,                  // [hiddenSize, 11008], half
	IN_MLPLINEARDEQSCALEUP, // quant
	IN_LINEARBIASUP,        // quant
	IN_WEIGHT_DOWN_ID,                    // [11008, hiddenSize], half
	IN_MLPLINEARDEQSCALEDOWN, // quant
	IN_BIAS_DOWN_ID,
	OUT_TRANSPOSED_RESULT_ID,           // [batch, seqLen, hiddenSize], half
	INTERMEDIATE_MATMUL_GATE_OUT_ND_ID, // [batch, seqlen, 11008], half
	INTERMEDIATE_SWISH_OUT_ID,          // [batch, seqlen, 11008], half
	INTERMEDIATE_MATMUL_UP_OUT_ND_ID,   // [batch, seqlen, 11008], half
	INTERMEDIATE_MUL_OUT_ID,		    // [batch, seqlen, 11008], half
	INTERMEDIATE_SELFQUANTMLPOUT
};
static const uint64_t IN_TENSOR_COUNT = 10;
static const uint64_t OUT_TENSOR_COUNT = 1;

atb::Status CommonMlpQuant(const CommonMlpQuantParam& param, atb::Operation** operation)
{
	atb::GraphParam opGraph;
	opGraph.inTensorNum = IN_TENSOR_COUNT;
	opGraph.outTensorNum = OUT_TENSOR_COUNT;
	size_t internalTensorNum = 0;
	if (param.isFloat) {
		internalTensorNum = 4;
	} else {
		internalTensorNum = 5;
	}
	opGraph.internalTensorNum = internalTensorNum;
	size_t nodeCount = 0;
	if (param.isFloat) {
		nodeCount = 5;
	} else {
		nodeCount = 6;
	}
	opGraph.nodes.resize(nodeCount);

	size_t nodeId = 0;
	auto &linearGateNode = opGraph.nodes.at(nodeId++);
	auto &swishNode = opGraph.nodes.at(nodeId++);
	auto &linearUpNode = opGraph.nodes.at(nodeId++);
	auto &mulNode = opGraph.nodes.at(nodeId++);

	atb::infer::LinearQuantParam linearGateParam = { false, param.transpose, true };
	CreateOperation(linearGateParam, &linearGateNode.operation);
	linearGateNode.inTensorIds = { IN_HIDDENSTATES_ID, IN_WEIGHT_GATE_ID, IN_LINEARBIASGATE, IN_MLPLINEARDEQSCALEGATE }; // quant
	linearGateNode.outTensorIds = { INTERMEDIATE_MATMUL_GATE_OUT_ND_ID };

	atb::infer::ActivationParam swishParam;
	swishParam.activationType = atb::infer::ActivationType::ACTIVATION_SWISH;
	CreateOperation(swishParam, &swishNode.operation);
	swishNode.inTensorIds = { INTERMEDIATE_MATMUL_GATE_OUT_ND_ID };
	swishNode.outTensorIds = { INTERMEDIATE_SWISH_OUT_ID };

	atb::infer::LinearQuantParam linearUpParam = { false, param.transpose, true };
	CreateOperation(linearGateParam, &linearUpNode.operation);
	linearUpNode.inTensorIds = { IN_HIDDENSTATES_ID, IN_WEIGHT_UP_ID, IN_LINEARBIASUP, IN_MLPLINEARDEQSCALEUP };
	linearUpNode.outTensorIds = { INTERMEDIATE_MATMUL_UP_OUT_ND_ID };

	atb::infer::ElewiseParam mulParam;
	mulParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
	CreateOperation(mulParam, &mulNode.operation);
	mulNode.inTensorIds = { INTERMEDIATE_SWISH_OUT_ID, INTERMEDIATE_MATMUL_UP_OUT_ND_ID };
	mulNode.outTensorIds = { INTERMEDIATE_MUL_OUT_ID };

	ATB_LOG(INFO) << "enter mlp down linear";
	ATB_LOG(INFO) << "param.isFloat " << param.isFloat;

	if (param.isFloat) {
		auto &linearDownNode = opGraph.nodes.at(nodeId++);

		atb::infer::LinearParam linearDownParam = { false, true, true };
		CreateOperation(linearDownParam, &linearDownNode.operation);

		linearDownNode.inTensorIds = { INTERMEDIATE_MUL_OUT_ID, IN_WEIGHT_DOWN_ID, IN_BIAS_DOWN_ID };
		linearDownNode.outTensorIds = { OUT_TRANSPOSED_RESULT_ID };
	} else {
		auto &selfMlpOutQuantNode = opGraph.nodes.at(nodeId++);
		// add quant op
		atb::infer::ElewiseParam mlpQuantParam;
		mlpQuantParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_QUANT;
		mlpQuantParam.quantParam.inputScale = param.inputScale_down_proj;
		mlpQuantParam.quantParam.inputOffset = param.inputOffset_down_proj;
		CreateOperation(mlpQuantParam, &selfMlpOutQuantNode.operation);
		selfMlpOutQuantNode.inTensorIds = {INTERMEDIATE_MUL_OUT_ID};
		selfMlpOutQuantNode.outTensorIds = {INTERMEDIATE_SELFQUANTMLPOUT};  // quant

		auto &linearDownNode = opGraph.nodes.at(nodeId++);

		atb::infer::LinearQuantParam linearDownParam = { false, param.transpose, true };
		CreateOperation(linearDownParam, &linearDownNode.operation);

		linearDownNode.inTensorIds = { INTERMEDIATE_SELFQUANTMLPOUT, IN_WEIGHT_DOWN_ID, IN_BIAS_DOWN_ID, IN_MLPLINEARDEQSCALEDOWN };
		linearDownNode.outTensorIds = { OUT_TRANSPOSED_RESULT_ID };
	}

	opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc>& inTensorDescs,
		atb::SVector<atb::TensorDesc>& outTensorDescs) {
			outTensorDescs.at(0) = inTensorDescs.at(0);
			outTensorDescs.at(0).dtype = ACL_FLOAT16;
			outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
			outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1];
			outTensorDescs.at(0).shape.dims[2] = inTensorDescs.at(0).shape.dims[2];
			return atb::NO_ERROR;
	};

	atb::CreateOperation(opGraph, operation);
	return atb::NO_ERROR;
}
} // namespace telechat
} // namespace atb_speed