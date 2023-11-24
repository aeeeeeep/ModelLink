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

#include <torch/csrc/autograd/custom_function.h>
#include "torch_npu/csrc/framework/utils/RandomOpAdapter.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include "common.h"

using namespace at_npu::native;
namespace at_npu {
namespace native {
using torch::autograd::AutogradContext;
using torch::autograd::Function;

enum class DropOutStatus
{
  DROPOUT_NORMAL = 0,
  DROPOUT_NONE,
  DROPOUT_ALL
};

DropOutStatus get_status(double keep_prob)
{
  if (keep_prob == 0) {
    return DropOutStatus::DROPOUT_ALL;
  }
  if (keep_prob == 1.) {
    return DropOutStatus::DROPOUT_NONE;
  }
  return DropOutStatus::DROPOUT_NORMAL;
}

static at::Tensor tensor_format_trans(const at::Tensor &at_tensor)
{
  if (at_tensor.defined()) {
    TORCH_CHECK(at_npu::key::isDeviceTensor(at_tensor), "only npu tensor is supported");
    return NPUNativeFunctions::npu_format_cast(at_tensor, ACL_FORMAT_ND);
  }
  return at_tensor;
}

at::Tensor gen_mask_impl(const at::Tensor &self, const at::Scalar &keep_prob, const at::Scalar &seed,
                         const int64_t offset, const int64_t numels)
{
  int64_t length = (numels + 256 - 1) / 256 * 256 / 8;
  c10::TensorOptions options = self.options();
  at::Tensor mask = OpPreparation::ApplyTensorWithoutFormat(at::IntArrayRef{length + 32}, options.dtype(at::kByte));
  at::SmallVector<int64_t, N> offsetList = {0, offset};
  const int64_t seed1 = 0;
  OpCommand cmd;
  cmd.Name("StatelessDropOutGenMask")
      .Input(at::IntArrayRef{numels})
      .Input(keep_prob, self.scalar_type(), CompileType::MEMORY_HOST_COMPILE_DEPENDENT)
      .Input(seed, at::ScalarType::Int)
      .Input(at::Scalar(seed1), at::ScalarType::Int)
      .Input(offsetList, at::kLong, CompileType::MEMORY_HOST_COMPILE_INDEPENDENT)
      .Output(mask)
      .Run();
  return mask;
}

at::Tensor gen_mask_dispatch(const at::Tensor &self, const at::Scalar &keep_prob, const at::Scalar &seed,
                             const int64_t offset, const int64_t numels, const bool gen_mask_parallel,
                             const bool sync)
{
  at::Tensor mask;

  if (gen_mask_parallel) {
    auto original_stream = c10_npu::getCurrentNPUStream();
    {
      // During the life cycle of this raii instance, the calcu stream is set as the
      // secondary stream, and tasks are distributed to the secondary stream. At the
      // same time, according to the one-stream-one-pool principle, memory is also
      // alloced from the pool of the secondary stream.
      c10_npu::SecondaryStreamGuard guard(c10_npu::getCurrentSecondaryStream());
      mask = gen_mask_impl(self, keep_prob, seed, offset, numels);
      if (sync) {
        NPU_CHECK_ERROR(c10_npu::acl::AclrtSynchronizeStreamWithTimeout(original_stream));
      }
    }
  } else {
    mask = gen_mask_impl(self, keep_prob, seed, offset, numels);
  }
  return mask;
}

at::Tensor gen_mask(const at::Tensor &self, double keep_prob, int64_t head_num, std::string input_layout,
                    bool gen_mask_parallel, bool sync, int64_t &seed, int64_t &offset, int64_t &numels)
{
  at::Tensor drop_mask;
  if (input_layout == "BSH") {
    numels = self.size(0) * head_num * self.size(1) * self.size(1); // [B,N,S,S]
  } else if (input_layout == "SBH") {
    numels = self.size(1) * head_num * self.size(0) * self.size(0); // [B,N,S,S]
  } else if (input_layout == "BNSD") {
    numels = self.size(0) * self.size(1) * self.size(2) * self.size(2); // [B,N,S,S]
  }
  int64_t length = (numels + 256 - 1) / 256 * 256 / 8;
  length += 32;
  if (get_status(keep_prob) == DropOutStatus::DROPOUT_NORMAL) {
    const auto gen = at_npu::detail::getDefaultNPUGenerator();
    auto pair = at::check_generator<NPUGeneratorImpl>(gen)->philox_engine_inputs(10);
    seed = pair.first;
    offset = pair.second;
    drop_mask = gen_mask_dispatch(self, at::Scalar(keep_prob), at::Scalar(seed), offset, numels,
                                  gen_mask_parallel, sync);
  } else if (get_status(keep_prob) == DropOutStatus::DROPOUT_ALL) {
    drop_mask = at::zeros(at::IntArrayRef{length}, self.options().dtype(at::kByte));
  }
  return drop_mask;
}

std::vector<at::Tensor> ascend_flash_attention_backward(
    const at::Tensor &attention_score_grad,
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    const at::Tensor &softmax_log_max_sum,
    const at::Tensor &attention_score,
    const c10::optional<at::Tensor> &atten_mask,
    const c10::optional<at::Tensor> &alibi_mask,
    const c10::optional<at::Tensor> &drop_mask,
    double scale_value,
    int64_t head_num,
    const std::string input_layout,
    double keep_prob,
    int64_t pre_tokens,
    int64_t next_tokens,
    bool gen_mask_parallel,
    bool sync)
{
  const at::Tensor &drop_mask_const = drop_mask.value_or(at::Tensor());
  at::Tensor format_drop_mask = tensor_format_trans(drop_mask_const);

  at::Tensor format_attention_score_grad = tensor_format_trans(attention_score_grad);
  at::Tensor format_query = tensor_format_trans(query);
  at::Tensor format_key = tensor_format_trans(key);
  at::Tensor format_value = tensor_format_trans(value);
  at::Tensor format_softmax_log_max_sum = tensor_format_trans(softmax_log_max_sum);
  at::Tensor format_attention_score = tensor_format_trans(attention_score);

  const at::Tensor &atten_mask_const = atten_mask.value_or(at::Tensor());
  at::Tensor format_atten_mask = tensor_format_trans(atten_mask_const);
  at::Tensor dtype_atten_mask = (format_atten_mask.defined() && format_atten_mask.scalar_type() != query.scalar_type())
                                 ? format_atten_mask.to(query.scalar_type())
                                 : format_atten_mask;
  const at::Tensor &alibi_mask_const = alibi_mask.value_or(at::Tensor());
  at::Tensor format_alibi_mask = tensor_format_trans(alibi_mask_const);
  at::Tensor dtype_alibi_mask = (format_alibi_mask.defined() && format_alibi_mask.scalar_type() != query.scalar_type())
                                 ? format_alibi_mask.to(query.scalar_type())
                                 : format_alibi_mask;

  at::Tensor query_grad;
  at::Tensor key_grad;
  at::Tensor value_grad;

  query_grad = OpPreparation::ApplyTensorWithFormat(format_query.sizes(), format_query.options().dtype(at::kFloat),
                                                    ACL_FORMAT_ND);
  key_grad = OpPreparation::ApplyTensorWithFormat(format_key.sizes(), format_key.options().dtype(at::kFloat),
                                                  ACL_FORMAT_ND);
  value_grad = OpPreparation::ApplyTensorWithFormat(format_value.sizes(), format_value.options().dtype(at::kFloat),
                                                    ACL_FORMAT_ND);

  query_grad.zero_();
  key_grad.zero_();
  value_grad.zero_();

  char* input_layout_ptr = const_cast<char *>(input_layout.c_str());
  EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnAscendFlashAttentionGrad,
                               format_query, format_key, format_value,
                               format_softmax_log_max_sum, format_attention_score, format_attention_score_grad,
                               dtype_atten_mask, dtype_alibi_mask, format_drop_mask,
                               scale_value, head_num, input_layout_ptr, keep_prob, pre_tokens, next_tokens,
                               query_grad, key_grad, value_grad);

  query_grad = query_grad.to(query.scalar_type());
  key_grad = key_grad.to(query.scalar_type());
  value_grad = value_grad.to(query.scalar_type());

  return {query_grad, key_grad, value_grad, at::Tensor(), at::Tensor(),
          at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
          at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
}


class NPUAscendFlashAttentionFunction : public torch::autograd::Function<NPUAscendFlashAttentionFunction> {
 public:
  static std::vector<at::Tensor> forward(AutogradContext *ctx, const at::Tensor &query, const at::Tensor &key,
      const at::Tensor &value, const c10::optional<at::Tensor> &atten_mask_opt,
      const c10::optional<at::Tensor> &alibi_mask_opt, double scale, int64_t head_num,
      c10::string_view input_layout, double keep_prob, int64_t pre_tokens, int64_t next_tokens,
      bool gen_mask_parallel, bool sync)
    {
    const at::Tensor &atten_mask = atten_mask_opt.value_or(at::Tensor());
    const at::Tensor &alibi_mask = alibi_mask_opt.value_or(at::Tensor());

    TORCH_CHECK(keep_prob >= 0 && keep_prob <= 1,
      "The keep_prob value must be in range of [0, 1], but got ", keep_prob);

    std::string input_layout_str = std::string(input_layout);
    TORCH_CHECK(input_layout_str == "BSH" || input_layout_str == "SBH" || input_layout_str == "BNSD",
        "The input_layout should be BSH/SBH/BNSD(case-insensitive), but got ", input_layout);
    char* input_layout_ptr = const_cast<char *>(input_layout_str.c_str());

    int64_t seed;
    int64_t offset;
    int64_t numels;

    at::Tensor drop_mask = gen_mask(query, keep_prob, head_num, input_layout_str,
                                    gen_mask_parallel, sync, seed, offset, numels);

    at::Tensor format_drop_mask = tensor_format_trans(drop_mask);
    at::Tensor format_query = tensor_format_trans(query);
    at::Tensor format_key = tensor_format_trans(key);
    at::Tensor format_value = tensor_format_trans(value);
    at::Tensor format_atten_mask = tensor_format_trans(atten_mask);
    at::Tensor dtype_atten_mask = (format_atten_mask.defined() && format_atten_mask.scalar_type() != query.scalar_type())
                                   ? format_atten_mask.to(query.scalar_type())
                                   : format_atten_mask;
    at::Tensor format_alibi_mask = tensor_format_trans(alibi_mask);
    at::Tensor dtype_alibi_mask = (format_alibi_mask.defined() && format_alibi_mask.scalar_type() != query.scalar_type())
                                   ? format_alibi_mask.to(query.scalar_type())
                                   : format_alibi_mask;

    int64_t B = 0;
    int64_t S = 0;
    if (input_layout == "BSH") {
      B = query.size(0);
      S = query.size(1);
    } else if (input_layout == "SBH") {
      B = query.size(1);
      S = query.size(0);
    } else if (input_layout == "BNSD") {
      B = query.size(0);
      S = query.size(2);
    }
    at::Tensor softmax_log_max_sum = OpPreparation::ApplyTensorWithoutFormat({B, head_num, S},
                                                                             query.options().dtype(at::kFloat));
    at::Tensor attention_score = OpPreparation::ApplyTensorWithoutFormat(query);

    EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnAscendFlashAttention,
                                 format_query, format_key, format_value, dtype_atten_mask, dtype_alibi_mask,
                                 format_drop_mask, scale, head_num, input_layout_ptr, keep_prob, pre_tokens,
                                 next_tokens, softmax_log_max_sum, attention_score);

    if (!sync) {
      c10_npu::NPUEvent npu_event;
      npu_event.record(c10_npu::getCurrentNPUStream());
      npu_event.block(c10_npu::getCurrentSecondaryStream());
    }

    at::AutoNonVariableTypeMode g;

    ctx->save_for_backward({format_query, format_key, format_value, format_atten_mask, format_alibi_mask,
                            softmax_log_max_sum, attention_score});
    ctx->saved_data["scale"] = scale;
    ctx->saved_data["keep_prob"] = keep_prob;
    ctx->saved_data["head_num"] = head_num;
    ctx->saved_data["input_layout_ptr"] = input_layout_ptr;
    ctx->saved_data["pre_tokens"] = pre_tokens;
    ctx->saved_data["next_tokens"] = next_tokens;
    ctx->saved_data["gen_mask_parallel"] = gen_mask_parallel;
    ctx->saved_data["sync"] = sync;
    ctx->saved_data["seed"] = seed;
    ctx->saved_data["offset"] = offset;
    ctx->saved_data["numels"] = numels;

    return {attention_score, softmax_log_max_sum};
  }

  static std::vector<at::Tensor> backward(AutogradContext* ctx, std::vector<at::Tensor> grad_outputs)
  {
    auto scale = ctx->saved_data["scale"].toDouble();
    auto keep_prob = ctx->saved_data["keep_prob"].toDouble();
    auto head_num = ctx->saved_data["head_num"].toInt();
    auto input_layout = ctx->saved_data["input_layout_ptr"].toStringRef();
    auto pre_tokens = ctx->saved_data["pre_tokens"].toInt();
    auto next_tokens = ctx->saved_data["next_tokens"].toInt();
    auto gen_mask_parallel = ctx->saved_data["gen_mask_parallel"].toBool();
    auto sync = ctx->saved_data["sync"].toBool();
    auto seed = ctx->saved_data["seed"].toInt();
    auto offset = ctx->saved_data["offset"].toInt();
    auto numels = ctx->saved_data["numels"].toInt();
    auto saved = ctx->get_saved_variables();

    auto query = saved[0];
    auto key = saved[1];
    auto value = saved[2];
    auto atten_mask = saved[3];
    auto alibi_mask = saved[4];
    auto softmax_log_max_sum = saved[5];
    auto attention_score = saved[6];

    int64_t length = (numels + 256 - 1) / 256 * 256 / 8;
    length += 32;
    at::Tensor drop_mask;
    if (get_status(keep_prob) == DropOutStatus::DROPOUT_NORMAL) {
      drop_mask = gen_mask_dispatch(query, at::Scalar(keep_prob), at::Scalar(seed), offset, numels,
                                    gen_mask_parallel, sync);
    } else if (get_status(keep_prob) == DropOutStatus::DROPOUT_ALL) {
      drop_mask = at::zeros(at::IntArrayRef{length}, query.options().dtype(at::kByte));
    }

    auto results = ascend_flash_attention_backward(grad_outputs[0], query, key, value,
                                                   softmax_log_max_sum, attention_score,
                                                   atten_mask, alibi_mask, drop_mask,
                                                   scale, head_num, input_layout, keep_prob,
                                                   pre_tokens, next_tokens, gen_mask_parallel, sync);
    if (!sync) {
      c10_npu::NPUEvent npu_event;
      npu_event.record(c10_npu::getCurrentNPUStream());
      npu_event.block(c10_npu::getCurrentSecondaryStream());
    }
    return results;
  }
};
}  // namespace native
}  // namespace at_npu

std::vector<at::Tensor> ascend_flash_attention_grad(
    const at::Tensor &attention_score_grad,
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    const at::Tensor &softmax_log_max_sum,
    const at::Tensor &attention_score,
    const c10::optional<at::Tensor> &atten_mask,
    const c10::optional<at::Tensor> &alibi_mask,
    double scale_value,
    int64_t head_num,
    c10::string_view input_layout,
    double keep_prob,
    int64_t pre_tokens,
    int64_t next_tokens,
    bool gen_mask_parallel,
    bool sync)
{
  TORCH_CHECK(keep_prob >= 0 && keep_prob <= 1,
    "The keep_prob value must be in range of [0, 1], but got ", keep_prob);

  std::string input_layout_str = std::string(input_layout);
  TORCH_CHECK(input_layout_str == "BSH" || input_layout_str == "SBH" || input_layout_str == "BNSD",
    "The input_layout should be BSH/SBH/BNSD(case-insensitive), but got ", input_layout);

  int64_t seed;
  int64_t offset;
  int64_t numels;
  at::Tensor drop_mask = gen_mask(query, keep_prob, head_num, input_layout_str,
                                  gen_mask_parallel, sync, seed, offset, numels);
  auto result = ascend_flash_attention_backward(attention_score_grad, query, key, value,
                                                 softmax_log_max_sum, attention_score,
                                                 atten_mask, alibi_mask, drop_mask,
                                                 scale_value, head_num, input_layout_str, keep_prob,
                                                 pre_tokens, next_tokens, gen_mask_parallel, sync);
  if (!sync) {
    c10_npu::NPUEvent npu_event;
    npu_event.record(c10_npu::getCurrentNPUStream());
    npu_event.block(c10_npu::getCurrentSecondaryStream());
  }
  return result;
}

std::vector<at::Tensor> ascend_flash_attention(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    const c10::optional<at::Tensor> &atten_mask,
    const c10::optional<at::Tensor> &alibi_mask,
    double scale,
    int64_t head_num,
    c10::string_view input_layout,
    double keep_prob,
    int64_t pre_tokens,
    int64_t next_tokens,
    bool gen_mask_parallel,
    bool sync)
{
    return NPUAscendFlashAttentionFunction::apply(query, key, value, atten_mask, alibi_mask, scale,
                                                  head_num, input_layout, keep_prob, pre_tokens,
                                                  next_tokens, gen_mask_parallel, sync);
}
