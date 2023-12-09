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

#include "torch_npu/csrc/framework/utils/RandomOpAdapter.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "common.h"
#include "aclnn_common.h"

std::vector<at::Tensor> npu_scatter_list(
    std::vector<at::Tensor> self,
    const at::Tensor &indice,
    const at::Tensor &updates,
    const c10::optional<at::Tensor> &mask,
    const std::string reduce,
    int64_t axis)
{
    char *reduce_ptr = const_cast<char *>(reduce.c_str());
    // The attribute 'reduce' of ScatterList only supports setting it to 'update'.
    std::vector<at::Tensor> result;
    for (const at::Tensor &tensor : self) {
        result.push_back(tensor.clone());
    }
    at::TensorList result_ = at::TensorList(result);
    ACLNN_CMD(aclnnScatterList, result_, indice, updates, mask, reduce_ptr, axis);

    return result;
}

void npu_scatter_list_(
    std::vector<at::Tensor> self,
    const at::Tensor &indice,
    const at::Tensor &updates,
    const c10::optional<at::Tensor> &mask,
    const std::string reduce,
    int64_t axis)
{
    char *reduce_ptr = const_cast<char *>(reduce.c_str());
    at::TensorList result_ = at::TensorList(self);
    ACLNN_CMD(aclnnScatterList, result_, indice, updates, mask, reduce_ptr, axis);

    return;
}