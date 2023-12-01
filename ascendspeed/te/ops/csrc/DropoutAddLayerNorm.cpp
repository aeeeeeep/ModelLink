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

#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "common.h"
#include "aclnn_common.h"

inline void npu_dropout_add_layer_norm_check(
    const at::Tensor &x0, // BxSxhidden_size
    const at::Tensor &weight, // hidden_size
    const c10::optional<at::Tensor> &residual_opt, // BxSxhidden_size
    const c10::optional<at::Tensor> &bias_opt, // hidden_size
    const c10::optional<at::Tensor> &rowscale_opt, // BxS
    const c10::optional<at::Tensor> &layerscale_opt, // hidden_size
    double p,
    double eps)
{
    TORCH_CHECK(
        torch_npu::utils::is_npu(x0),
        "npu_dropout_add_layer_norm only supports device for NPU!");
    
    auto itype = x0.scalar_type();
    auto wtype = weight.scalar_type();

    TORCH_CHECK(
        !(itype == at::kBFloat16 && wtype == at::kHalf),
        "weight_dtype == torch.float16 and input_dtype == torch.bfloat16 was not supported");
    
    if (bias_opt.has_value()) {
        auto bias = bias_opt.value();
        TORCH_CHECK(bias.dtype() == wtype);
        TORCH_CHECK(bias.sizes() == weight.sizes());
    }

    if (residual_opt.has_value()) {
        auto residual = residual_opt.value();
        TORCH_CHECK(residual.sizes() == x0.sizes());
    }

    if (rowscale_opt.has_value()) {
        auto rowscale = rowscale_opt.value();
        TORCH_CHECK(rowscale.dim() == x0.dim() - 1);
        TORCH_CHECK(rowscale.dtype() == itype);
    }

    if (layerscale_opt.has_value()) {
        auto layerscale = layerscale_opt.value();
        TORCH_CHECK(layerscale.sizes()[0] == x0.sizes().back());
        TORCH_CHECK(layerscale.dtype() == wtype);
    }

    TORCH_CHECK(
        p >= 0 && p <= 1,
        "dropout probability has to be between 0 and 1, but got ", p);
    
    TORCH_CHECK(eps >= 0.f);

    auto hidden_size = weight.numel();
    TORCH_CHECK((hidden_size % 8 == 0) && (hidden_size <= 8192));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_dropout_add_layer_norm(
    const at::Tensor &x0,
    const at::Tensor &weight,
    const c10::optional<at::Tensor> &residual_opt,
    const c10::optional<at::Tensor> &bias_opt,
    const c10::optional<at::Tensor> &rowscale_opt,
    const c10::optional<at::Tensor> &layerscale_opt,
    double p,
    double eps,
    bool prenorm,
    bool residual_in_fp32,
    bool is_rms_norm,
    bool return_dropout_mask)
{
    npu_dropout_add_layer_norm_check(
        x0, weight, residual_opt, bias_opt, rowscale_opt, layerscale_opt, p, eps);

    const at::Tensor &residual_ = c10::value_or_else(residual_opt, [] { return at::Tensor(); });
    const at::Tensor &bias_ = c10::value_or_else(bias_opt, [] { return at::Tensor(); });
    const at::Tensor &rowscale_ = c10::value_or_else(rowscale_opt, [] { return at::Tensor(); });
    const at::Tensor &layerscale_ = c10::value_or_else(layerscale_opt, [] { return at::Tensor(); });

    at::Tensor residual = residual_;
    at::Tensor bias = bias_;
    at::Tensor rowscale = rowscale_;
    at::Tensor layerscale = layerscale_;

    // residual_in_fp32 only has an effect if residual is None.
    // Otherwise residual dtype is residual.dtype.
    at::ScalarType residual_dtype = residual.defined() ?
                                    residual.scalar_type() :
                                    (residual_in_fp32 ? at::kFloat : x0.scalar_type());

    // Calculate scaled_x0
    at::Tensor scaled_x0 = x0.clone();
    if (rowscale.defined()) {
        rowscale = at::repeat_interleave(rowscale, x0.size(-1));
        rowscale = rowscale.view(x0.sizes());
        scaled_x0.mul_(rowscale);
    }
    if (layerscale.defined()) {
        scaled_x0.mul_(layerscale);
    }

    // Apply dropout to scaled_x0
    at::Tensor dropout_result;
    at::Tensor mask;
    bool train = p == 0.0 ? false : true;
    if (train) {
        double p1m = 1. - p;
        double scale = p1m == 0 ? 0. : 1. / p1m;
        mask = at::empty_like(scaled_x0, scaled_x0.options().dtype(c10::CppTypeToScalarType<bool>::value));
        mask.bernoulli_(p1m);
        dropout_result = scaled_x0.mul(mask).mul_(scale);
    } else {
        mask = at::ones_like(scaled_x0, scaled_x0.options().dtype(c10::CppTypeToScalarType<bool>::value));
        dropout_result = scaled_x0.clone();
    }

    // Apply layer_norm or rms_norm to (dropout_result + residual)
    at::Tensor norm_result;
    at::Tensor mean;
    at::Tensor rstd;
    at::Tensor pre_norm = dropout_result.clone();
    if (residual.defined()) {
        pre_norm.add_(residual);
    }
    int hidden_size = weight.numel();
    if (!is_rms_norm) {
        auto native_layer_norm_output = at::native_layer_norm(pre_norm, hidden_size, weight, bias, eps);
        norm_result = std::get<0>(native_layer_norm_output);
        mean = std::get<1>(native_layer_norm_output);
        rstd = std::get<2>(native_layer_norm_output);
    } else {
        at::Tensor norm_x = pre_norm.norm(2, -1, true);
        at::Tensor temp_ones = at::ones({1}, pre_norm.options());
        at::Tensor rms_x = norm_x.mul(temp_ones.mul(hidden_size).pow_(-0.5));
        rstd = rms_x.add(eps).pow_(-1);
        norm_result = pre_norm.mul(rstd).mul_(weight);
    }

    // Update outputs
    norm_result = (norm_result.scalar_type() == x0.scalar_type()) ?
                  norm_result : norm_result.to(x0.scalar_type());
    at::Tensor pre_norm_result;
    if (prenorm) {
        pre_norm_result = (pre_norm.scalar_type() == residual_dtype) ?
                          pre_norm.clone() : pre_norm.clone().to(residual_dtype);
    }
    at::Tensor mask_result;
    if (return_dropout_mask) {
        mask_result = mask.clone();
    }

    return std::tie(norm_result, pre_norm_result, mask_result);
}
