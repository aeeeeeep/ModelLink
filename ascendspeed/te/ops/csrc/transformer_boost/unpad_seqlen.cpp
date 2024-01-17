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

#ifdef ENABLE_ATB
#include "inc/atb_adapter.h"
#include "atb/operation.h"
#include "atb/train_op_params.h"
#endif 

using namespace std;

const static int HIDDEN_DIM = 2;

void infer_shape_unpad(c10::SmallVector<int64_t, N> &size, const at::Tensor input_tensor, const std::vector<int> seqLen, int maxSeqLen)
{
    int sumSeqlen = 0;
    for (auto item : seqLen) {
        sumSeqlen += item;
    }
    size = {sumSeqlen, input_tensor.size(HIDDEN_DIM)};
}

at::Tensor unpad_seqlen(const at::Tensor &dataInput, const std::vector<int> seqLen, int maxSeqLen)
{
    atb::train::UnpadWithHiddenStateParam param;
    param.maxSeqLen = maxSeqLen;
    for (auto item : seqLen) {
        param.qSeqLen.push_back(item);
    }

    atb::Operation* op = nullptr;
    atb::CreateOperation(param, &op);
    TORCH_CHECK(op != nullptr, "get op failed!");

    c10::SmallVector<int64_t, N> outputShape;
    infer_shape_unpad(outputShape, dataInput, seqLen, maxSeqLen);
    at::Tensor outputAtTensor = CreateAtTensor(outputShape, dataInput.scalar_type());
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
    cmd.Name("UnpadWithHiddenStateOperation");
    cmd.SetCustomHandler(acl_call);
    cmd.Run();
    return outputAtTensor;
}