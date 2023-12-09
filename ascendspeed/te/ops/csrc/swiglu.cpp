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

#include "common.h"
#include "aclnn_common.h"

inline std::vector<int64_t> swiglu_grad_infershape(const at::Tensor &x, int64_t dim)
{
    if (dim < 0) {
        dim += x.sizes().size();
    }
    TORCH_CHECK(dim < x.sizes().size(), "dim out of range", dim)

    auto input_sizes = x.sizes();
    std::vector<int64_t> output_sizes;
    output_sizes.reserve(input_sizes.size());
    for (size_t i = 0; i < input_sizes.size(); ++i) {
        output_sizes.push_back(input_sizes[i]);
    }
    output_sizes[dim] /= 2;
    return output_sizes;
}

torch::Tensor npu_swiglu_backward(const at::Tensor &grad_output, const at::Tensor &x, int64_t dim)
{
    at::Tensor result = at::empty(x.sizes(), x.options());
    ACLNN_CMD(aclnnSwiGluGrad, grad_output, x, dim, result);
    return result;
}

torch::Tensor npu_swiglu(const at::Tensor &x, int64_t dim)
{
    auto output_sizes = swiglu_grad_infershape(x, dim);
    at::Tensor result = at::empty(output_sizes, x.options());
    ACLNN_CMD(aclnnSwiGlu, x, dim, result);
    return result;
}
