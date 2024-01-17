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

void infer_shape_strided_batch_matmul(c10::SmallVector<int64_t, N> &size, int headNum,
    const std::vector<int32_t> m, const std::vector<int32_t> n)
{
    int outputShape = 0;
    for (int i = 0; i < m.size(); i++) {
        outputShape += headNum * m[i] * n[i];
    }
    size = {outputShape};
}

at::Tensor strided_batch_matmul(const at::Tensor &input1, const at::Tensor &input2, int32_t transA, int32_t transB,
    const std::vector<int32_t> m, const std::vector<int32_t> k, const std::vector<int32_t> n,
    const std::vector<int32_t> lda, const std::vector<int32_t> ldb, const std::vector<int32_t> ldc,
    const std::vector<int32_t> strideA, const std::vector<int32_t> strideB, const std::vector<int32_t> strideC,
    int32_t batch, int32_t headNum) 
{
    atb::train::StridedBatchMatmulParam param;
    param.transposeA = transA;
    param.transposeB = transB;
    for (auto item : m) {
        param.m.push_back(item);
    }
    for (auto item : k) {
        param.k.push_back(item);
    }
    for (auto item : n) {
        param.n.push_back(item);
    }
    for (auto item : lda) {
        param.lda.push_back(item);
    }
    for (auto item : ldb) {
        param.ldb.push_back(item);
    }
    for (auto item : ldc) {
        param.ldc.push_back(item);
    }
    for (auto item : strideA) {
        param.strideA.push_back(item);
    }
    for (auto item : strideB) {
        param.strideB.push_back(item);
    }
    for (auto item : strideC) {
        param.strideC.push_back(item);
    }
    param.batch = batch;
    param.headNum = headNum;

    atb::Operation* op = nullptr;
    atb::CreateOperation(param, &op);
    TORCH_CHECK(op != nullptr, "get op failed!");

    c10::SmallVector<int64_t, N> outputShape;
    infer_shape_strided_batch_matmul(outputShape, headNum, m, n);
    at::Tensor outputAtTensor = CreateAtTensor(outputShape, input1.scalar_type());
    atb::Tensor outputTensor = AtTensor2Tensor(outputAtTensor);
    atb::Tensor inputTensor1 = AtTensor2Tensor(input1);
    atb::Tensor inputTensor2 = AtTensor2Tensor(input2);

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
    cmd.Name("StridedBatchMatmulOperation");
    cmd.SetCustomHandler(acl_call);
    cmd.Run();
    return outputAtTensor;
}