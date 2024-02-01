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
#ifndef ASCENDSPEED_TE_OPS_CSRC_COMMON_H
#define ASCENDSPEED_TE_OPS_CSRC_COMMON_H

#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/DeviceUtils.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <torch/script.h>
#include <torch/custom_class.h>

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t, int64_t> npu_flash_attention(
    const at::Tensor &query, const at::Tensor &key,
    const at::Tensor &value, int64_t head_num, std::string input_layout,
    const c10::optional<at::Tensor> &pse_opt, const c10::optional<at::Tensor> &padding_mask_opt,
    const c10::optional<at::Tensor> &atten_mask_opt,
    double scale, double keep_prob, int64_t pre_tockens, int64_t next_tockens, int64_t inner_precise,
    c10::optional<at::IntArrayRef> prefix_opt, int64_t sparse_mode, bool gen_mask_parallel, bool sync);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_flash_attention_grad(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    const at::Tensor &dy,
    int64_t head_num,
    std::string input_layout,
    const c10::optional<at::Tensor> &pse,
    const c10::optional<at::Tensor> &padding_mask,
    const c10::optional<at::Tensor> &atten_mask,
    const c10::optional<at::Tensor> &softmax_max,
    const c10::optional<at::Tensor> &softmax_sum,
    const c10::optional<at::Tensor> &softmax_in,
    const c10::optional<at::Tensor> &attention_in,
    double scale_value,
    double keep_prob,
    int64_t pre_tockens,
    int64_t next_tockens,
    int64_t inner_precise,
    int64_t seed,
    int64_t offset,
    int64_t numels,
    c10::optional<at::IntArrayRef> prefix,
    int64_t sparse_mode,
    bool gen_mask_parallel,
    bool sync);

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
    bool return_dropout_mask);

std::tuple<at::Tensor, at::Tensor> fa(const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
                                      const c10::optional<at::Tensor> &atten_mask, const c10::optional<at::Tensor> &alibi_mask,
                                      const c10::optional<at::Tensor> &drop_mask, float scale_value, int64_t head_num,
                                      int64_t io_layout, float keep_prob, int64_t pre_tokens, int64_t next_tokens,
                                      int64_t precise_mode, int64_t groups);

std::tuple<at::Tensor, at::Tensor, at::Tensor> fag(const at::Tensor &dy, const at::Tensor &softmax_log_max_sum, const at::Tensor &attention_out,
                                                   const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
                                                   const c10::optional<at::Tensor> &atten_mask, const c10::optional<at::Tensor> &alibi_mask,
                                                   const c10::optional<at::Tensor> &drop_mask, float scale_value, int64_t head_num, int64_t io_layout,
                                                   float keep_prob, int64_t pre_tokens, int64_t next_tokens, int64_t precise_mode, int64_t groups);
                                                   
std::tuple<at::Tensor, int64_t, int64_t, int64_t> gen_mask(const at::Tensor &self, double keep_prob,
    int64_t head_num, std::string input_layout, bool gen_mask_parallel, bool sync);

at::Tensor exist_gen_mask(const at::Tensor &self, double keep_prob, bool gen_mask_parallel, bool sync,
    int64_t seed, int64_t offset, int64_t numels);

at::Tensor genattentionmask(const torch::Tensor &input1, const std::vector<int> seqLen, int headSize);

at::Tensor fastsoftmax(const at::Tensor &dataInput, const std::vector<int32_t> &seqLen, int32_t headNum);

at::Tensor fastsoftmaxgrad(const at::Tensor &yInput, const at::Tensor &yGrad,
    const std::vector<int32_t> &seqLen, int32_t headNum);

std::tuple<at::Tensor, at::Tensor> rope(const torch::Tensor &input1, const torch::Tensor &input2, 
    const torch::Tensor &input3, const torch::Tensor &input4, const torch::Tensor &input5, int rotaryCoeff, int cosFormat);

std::tuple<at::Tensor, at::Tensor> rope_grad(const torch::Tensor &input1, const torch::Tensor &input2,
    const torch::Tensor &input3, const torch::Tensor &input4, const std::vector<int> qSeqLen);

at::Tensor stridedbatchmatmul(const torch::Tensor &input1, const torch::Tensor &input2, int32_t transA,
    int32_t transB, const std::vector<int32_t> m, const std::vector<int32_t> k, const std::vector<int32_t> n,
    const std::vector<int32_t> lda, const std::vector<int32_t> ldb, const std::vector<int32_t> ldc,
    const std::vector<int32_t> strideA, const std::vector<int32_t> strideB,
    const std::vector<int32_t> strideC, int32_t batch, int32_t headNum);

at::Tensor unpad(const torch::Tensor &input, const std::vector<int> seqLen, int maxSeqLen);

at::Tensor pad(const torch::Tensor &input, const std::vector<int> seqLen, int maxSeqLen);

#endif
