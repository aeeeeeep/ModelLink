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


at::Tensor npu_masked_softmax_with_rel_pos_bias(
    const at::Tensor& x,
    const c10::optional<at::Tensor> &atten_mask,
    const at::Tensor& relative_pos_bias,
    double scale_value,
    int64_t inner_precision_mode)
{
    at::Tensor format_x = format_trans(x);
    at::Tensor result = at::empty(format_x.sizes(), format_x.options());
    ACLNN_CMD(aclnnMaskedSoftmaxWithRelPosBias, x, atten_mask, relative_pos_bias, scale_value, inner_precision_mode, result);
    return result;
}