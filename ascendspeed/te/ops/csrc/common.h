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

std::vector<at::Tensor> fa(const at::Tensor &query, const at::Tensor &key, const at::Tensor &value, 
                           const c10::optional<at::Tensor> &atten_mask_opt, const c10::optional<at::Tensor> &alibi_mask_opt, 
                           const at::Tensor &drop_mask_opt, float scale_value, int64_t head_num, int64_t io_layout, 
                           float keep_prob, int64_t pre_tokens, int64_t next_tokens, int64_t precise_mode, int64_t groups);

std::vector<at::Tensor> fag(const at::Tensor &dy, const at::Tensor &softmax_log_max_sum, const at::Tensor &attention_out, 
                            const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
                            const c10::optional<at::Tensor> &atten_mask_opt, const c10::optional<at::Tensor> &alibi_mask_opt, 
                            const c10::optional<at::Tensor> &drop_mask_opt, float scale_value, int64_t head_num, int64_t io_layout, 
                            float keep_prob, int64_t pre_tokens, int64_t next_tokens, int64_t precise_mode, int64_t groups);

std::vector<at::Tensor> ascend_flash_attention(
      const at::Tensor &query, const at::Tensor &key,
      const at::Tensor &value, const c10::optional<at::Tensor> &atten_mask, const c10::optional<at::Tensor> &alibi_mask,
      float scale_value, float q_scale, int64_t head_num, std::string io_layout, 
      float keep_prob, int64_t pre_tokens, int64_t next_tokens, int64_t precise_mode);
#endif
