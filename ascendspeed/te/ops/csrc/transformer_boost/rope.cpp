// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <string>
#include <vector>
#include <ATen/Tensor.h>

#include "../ops_interface.h"
#include "inc/adapter.h"
#include "atb/operation.h"
#include "atb/train_op_params.h"
#include "atb/infer_op_params.h"

using namespace std;

std::tuple<at::Tensor, at::Tensor> rope(const at::Tensor &input1, const at::Tensor &input2, 
    const at::Tensor &input3, const at::Tensor &input4, const at::Tensor &input5, int rotaryCoeff, int cosFormat)
{
    atb::infer::RopeParam param;
    param.rotaryCoeff = rotaryCoeff;
    param.cosFormat = cosFormat;

    atb::Operation* op = nullptr;
    atb::CreateOperation(param, &op);
    TORCH_CHECK(op != nullptr, "get op failed!");

    at::Tensor outputAtTensor1 = CreateAtTensor(input1.sizes(), input1.scalar_type());
    at::Tensor outputAtTensor2 = CreateAtTensor(input2.sizes(), input2.scalar_type());
    atb::Tensor outputTensor1 = AtTensor2Tensor(outputAtTensor1);
    atb::Tensor outputTensor2 = AtTensor2Tensor(outputAtTensor2);
    atb::Tensor inputTensor1 = AtTensor2Tensor(input1);
    atb::Tensor inputTensor2 = AtTensor2Tensor(input2);
    atb::Tensor inputTensor3 = AtTensor2Tensor(input3);
    atb::Tensor inputTensor4 = AtTensor2Tensor(input4);
    atb::Tensor inputTensor5 = AtTensor2Tensor(input5);

    atb::VariantPack variantPack;
    variantPack.inTensors.push_back(inputTensor1);
    variantPack.inTensors.push_back(inputTensor2);
    variantPack.inTensors.push_back(inputTensor3);
    variantPack.inTensors.push_back(inputTensor4);
    variantPack.inTensors.push_back(inputTensor5);
    variantPack.outTensors.push_back(outputTensor1);
    variantPack.outTensors.push_back(outputTensor2);

    uint64_t workspaceSize = 0;
    auto contextPtr = GetContext();
    atb::Status st = op->Setup(variantPack, workspaceSize, contextPtr);
    TORCH_CHECK(st == 0, "setup failed!");
    at::TensorOptions options = at::TensorOptions(torch_npu::utils::get_npu_device_type());
    void *workspacePtr = nullptr;
    if (workspaceSize > 0) {
        auto workspaceTensor = at::empty({workspaceSize}, options.dtype(at::kByte));
        workspacePtr = (void*)workspaceTensor.storage().data();
    }

    auto acl_call = [op, contextPtr, variantPack, workspacePtr, workspaceSize]() -> int {
        auto st = op->Execute(variantPack, (uint8_t *)workspacePtr, workspaceSize, contextPtr);
        DestroyOperation(op);
        return 0;
    };
    at_npu::native::OpCommand cmd;
    cmd.Name("RopeOperation");
    cmd.SetCustomHandler(acl_call);
    cmd.Run();
    return std::make_tuple(outputAtTensor1, outputAtTensor2);
}


std::tuple<at::Tensor, at::Tensor> rope_grad(const at::Tensor &input1, const at::Tensor &input2,
    const at::Tensor &input3, const at::Tensor &input4, const std::vector<int> qSeqLen)
{
    atb::train::RopeGradParam param;
    for (auto item : qSeqLen) {
        param.qSeqLen.push_back(item);
    }

    atb::Operation* op = nullptr;
    atb::CreateOperation(param, &op);
    TORCH_CHECK(op != nullptr, "get op failed!");

    at::Tensor outputAtTensor1 = CreateAtTensor(input1.sizes(), input1.scalar_type());
    at::Tensor outputAtTensor2 = CreateAtTensor(input2.sizes(), input2.scalar_type());
    atb::Tensor outputTensor1 = AtTensor2Tensor(outputAtTensor1);
    atb::Tensor outputTensor2 = AtTensor2Tensor(outputAtTensor2);
    atb::Tensor inputTensor1 = AtTensor2Tensor(input1);
    atb::Tensor inputTensor2 = AtTensor2Tensor(input2);
    atb::Tensor inputTensor3 = AtTensor2Tensor(input3);
    atb::Tensor inputTensor4 = AtTensor2Tensor(input4);

    atb::VariantPack variantPack;
    variantPack.inTensors.push_back(inputTensor1);
    variantPack.inTensors.push_back(inputTensor2);
    variantPack.inTensors.push_back(inputTensor3);
    variantPack.inTensors.push_back(inputTensor4);
    variantPack.outTensors.push_back(outputTensor1);
    variantPack.outTensors.push_back(outputTensor2);

    uint64_t workspaceSize = 0;
    auto contextPtr = GetContext();
    atb::Status st = op->Setup(variantPack, workspaceSize, contextPtr);
    TORCH_CHECK(st == 0, "setup failed!");
    at::TensorOptions options = at::TensorOptions(torch_npu::utils::get_npu_device_type());
    void *workspacePtr = nullptr;
    if (workspaceSize > 0) {
        auto workspaceTensor = at::empty({workspaceSize}, options.dtype(at::kByte));
        workspacePtr = (void*)workspaceTensor.storage().data();
    }

    auto acl_call = [op, contextPtr, variantPack, workspacePtr, workspaceSize]() -> int {
        auto st = op->Execute(variantPack, (uint8_t *)workspacePtr, workspaceSize, contextPtr);
        DestroyOperation(op);
        return 0;
    };
    at_npu::native::OpCommand cmd;
    cmd.Name("RopeGradOperation");
    cmd.SetCustomHandler(acl_call);
    cmd.Run();
    return std::make_tuple(outputAtTensor1, outputAtTensor2);
}