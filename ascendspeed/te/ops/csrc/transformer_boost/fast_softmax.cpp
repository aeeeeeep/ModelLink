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

using namespace std;

at::Tensor fast_softmax(const at::Tensor &dataInput, const std::vector<int32_t> &seqLen, int32_t headNum)
{
    atb::train::FastSoftMaxParam param;
    param.headNum = headNum;
    for (auto item : seqLen) {
        param.qSeqLen.push_back(item);
    }

    atb::Operation* op = nullptr;
    atb::CreateOperation(param, &op);
    TORCH_CHECK(op != nullptr, "get op failed!");

    at::Tensor outputAtTensor = CreateAtTensor(dataInput.sizes(), dataInput.scalar_type());
    atb::Tensor outputTensor = AtTensor2Tensor(outputAtTensor);
    atb::Tensor inputTensor = AtTensor2Tensor(dataInput);

    atb::VariantPack variantPack;
    variantPack.inTensors.push_back(inputTensor);
    variantPack.outTensors.push_back(outputTensor);

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
    cmd.Name("FastSoftMaxOperation");
    cmd.SetCustomHandler(acl_call);
    cmd.Run();
    return outputAtTensor;
}

at::Tensor fast_softmax_grad(const at::Tensor &yInput, const at::Tensor &yGrad,const std::vector<int32_t> &seqLen,
    int32_t headNum)
{
    atb::train::FastSoftMaxGradParam param;
    param.headNum = headNum;
    for (auto item : seqLen) {
        param.qSeqLen.push_back(item);
    }

    atb::Operation* op = nullptr;
    atb::CreateOperation(param, &op);
    TORCH_CHECK(op != nullptr, "get op failed!");

    at::Tensor outputAtTensor = CreateAtTensor(yInput.sizes(), yInput.scalar_type());
    atb::Tensor outputTensor = AtTensor2Tensor(outputAtTensor);
    atb::Tensor inputTensor1 = AtTensor2Tensor(yInput);
    atb::Tensor inputTensor2 = AtTensor2Tensor(yGrad);

    atb::VariantPack variantPack;
    variantPack.inTensors.push_back(inputTensor1);
    variantPack.inTensors.push_back(inputTensor2);
    variantPack.outTensors.push_back(outputTensor);

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
    cmd.Name("FastSoftMaxGradOperation");
    cmd.SetCustomHandler(acl_call);
    cmd.Run();
    return outputAtTensor;
}
