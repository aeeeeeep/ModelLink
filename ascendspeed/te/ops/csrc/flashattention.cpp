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

#include <torch_npu/csrc/core/npu/SecondaryStreamGuard.h>
#include <torch_npu/csrc/aten/NPUGeneratorImpl.h>
#include "atb/operation.h"
#include "atb/train_op_params.h"
#include "common.h"
#include "adapter.h"

using namespace std;
using torch::autograd::AutogradContext;
using torch::autograd::Function;

OP_SETPARAM(atb::train::FlashAttentionParam)
OP_SETPARAM(atb::train::FlashAttentionBackwardParam)

std::vector<at::Tensor> fa(const at::Tensor &query, const at::Tensor &key, const at::Tensor &value, 
                           const c10::optional<at::Tensor> &atten_mask_opt, const c10::optional<at::Tensor> &alibi_mask_opt, 
                           const at::Tensor &drop_mask, float scale_value, int64_t head_num, int64_t io_layout, float keep_prob, 
                           int64_t pre_tokens, int64_t next_tokens, int64_t precise_mode, int64_t groups, bool drop_mask_is_none)
{
    atb::train::FlashAttentionParam param;
    param.scaleValue = scale_value;
    param.headNum = head_num;
    param.preTokens = pre_tokens;
    param.nextTokens = next_tokens;
    param.preciseMode = precise_mode;
    param.ioLayout = (atb::train::FlashAttentionParam::IoLayout)io_layout;
    param.keepProb = keep_prob;
    param.groups = groups;

    std::vector<at::Tensor> outTensors;
    TECommand command;
    SetParam(param, command);
    command.Input(query)
           .Input(key)
           .Input(value)
           .Input(atten_mask_opt)
           .Input(alibi_mask_opt)
           .Input(drop_mask, drop_mask_is_none)
           .Output(outTensors);
    
    return outTensors;
}

std::vector<at::Tensor> fag(const at::Tensor &dy, const at::Tensor &softmax_log_max_sum, const at::Tensor &attention_out, 
                            const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
                            const at::Tensor &atten_mask, const at::Tensor &alibi_mask, 
                            const at::Tensor &drop_mask, float scale_value, int64_t head_num, int64_t io_layout, 
                            float keep_prob, int64_t pre_tokens, int64_t next_tokens, int64_t precise_mode, int64_t groups,
                            bool atten_mask_is_none, bool alibi_mask_is_none, bool drop_mask_is_none)
{
    atb::train::FlashAttentionBackwardParam param;
    param.scaleValue = scale_value;
    param.headNum = head_num;
    param.preTokens = pre_tokens;
    param.nextTokens = next_tokens;
    param.preciseMode = precise_mode;
    param.ioLayout = (atb::train::FlashAttentionBackwardParam::IoLayout)io_layout;
    param.keepProb = keep_prob;
    param.groups = groups;

    std::vector<at::Tensor> outTensors;
    TECommand command;
    SetParam(param, command);
    command.Input(dy)
           .Input(softmax_log_max_sum)
           .Input(attention_out)
           .Input(query)
           .Input(key)
           .Input(value)
           .Input(atten_mask, atten_mask_is_none)
           .Input(alibi_mask, alibi_mask_is_none)
           .Input(drop_mask, drop_mask_is_none)
           .Output(outTensors);
    
    return outTensors;
}

enum class DropOutStatus {
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

at::Tensor gen_mask_impl(const at::Tensor &self, const at::Scalar &keep_prob, const at::Scalar &seed,
    const int64_t offset, const int64_t numels)
{
    int64_t length = (numels + 256 - 1) / 256 * 256 / 8;
    c10::TensorOptions options = self.options();
    at::Tensor mask = at::empty(at::IntArrayRef{length + 32}, options.dtype(at::kByte));
    at::SmallVector<int64_t, at_npu::native::N> offsetList = {0, offset};
    const int64_t seed1 = 0;
    at_npu::native::OpCommand cmd;
    cmd.Name("StatelessDropOutGenMask")
        .Input(at::IntArrayRef{numels})
        .Input(keep_prob, self.scalar_type(), at_npu::native::CompileType::MEMORY_HOST_COMPILE_DEPENDENT)
        .Input(seed, at::ScalarType::Int)
        .Input(at::Scalar(seed1), at::ScalarType::Int)
        .Input(offsetList, at::kLong, at_npu::native::CompileType::MEMORY_HOST_COMPILE_INDEPENDENT)
        .Output(mask)
        .Run();
    return mask;
}

at::Tensor gen_mask_dispatch(const at::Tensor &self, const at::Scalar &keep_prob, const at::Scalar &seed,
    const int64_t offset, const int64_t numels, const bool gen_mask_parallel, const bool sync)
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

at::Tensor gen_mask(const at::Tensor &self, double keep_prob,
    int64_t head_num, std::string input_layout, bool gen_mask_parallel, bool sync,
    int64_t &seed, int64_t &offset, int64_t &numels, bool &is_none)
{
    at::Tensor drop_mask;
    is_none = false;
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
        auto pair = at::check_generator<at_npu::NPUGeneratorImpl>(gen)->philox_engine_inputs(10);
        seed = pair.first;
        offset = pair.second;
        drop_mask = gen_mask_dispatch(self, at::Scalar(keep_prob), at::Scalar(seed),
            offset, numels, gen_mask_parallel, sync);
    } else if (get_status(keep_prob) == DropOutStatus::DROPOUT_ALL) {
        drop_mask = at::zeros(at::IntArrayRef{length}, self.options().dtype(at::kByte));
    } else {
        is_none = true;
    }
    return drop_mask;
}

class NPUAscendFlashAttentionFunction : public torch::autograd::Function<NPUAscendFlashAttentionFunction> {
public:
  static std::vector<at::Tensor> forward(
      torch::autograd::AutogradContext *ctx, const at::Tensor &query, const at::Tensor &key,
      const at::Tensor &value, const c10::optional<at::Tensor> &atten_mask_opt, const c10::optional<at::Tensor> &alibi_mask_opt,
      float scale_value, float q_scale, int64_t head_num, std::string io_layout, 
      float keep_prob, int64_t pre_tokens, int64_t next_tokens, int64_t precise_mode)
  {
    int64_t layout = 0;
    int64_t seed;
    int64_t offset;
    int64_t numels;
    bool drop_is_none;

    if (io_layout == "BSH") {
        layout = 1;
    } else if (io_layout == "SBH") {
        layout = 2;
    }

    at::Tensor drop_mask = gen_mask(query, keep_prob, head_num, io_layout,
                                    true, false, seed, offset, numels, drop_is_none);

    std::vector<at::Tensor> results = fa(query, key, value, atten_mask_opt, alibi_mask_opt, drop_mask, 
    scale_value, head_num, layout, keep_prob, pre_tokens, next_tokens, precise_mode, -1, drop_is_none);

    at::AutoNonVariableTypeMode g;
    const at::Tensor& atten_mask = c10::value_or_else(atten_mask_opt, [] {return at::Tensor();});
    const at::Tensor& alibi_mask = c10::value_or_else(alibi_mask_opt, [] {return at::Tensor();});
    ctx->save_for_backward({query, key, value, atten_mask, alibi_mask, results[0], results[1]});

    ctx->saved_data["scale_value"] = scale_value;
    ctx->saved_data["q_scale"] = q_scale;
    ctx->saved_data["head_num"] = head_num;
    ctx->saved_data["sync"] = false;
    ctx->saved_data["gen_mask_parallel"] = true;
    ctx->saved_data["layout"] = layout;
    ctx->saved_data["groups"] = -1;
    ctx->saved_data["keep_prob"] = keep_prob;
    ctx->saved_data["pre_tokens"] = pre_tokens;
    ctx->saved_data["next_tokens"] = next_tokens;
    ctx->saved_data["precise_mode"] = precise_mode;
    ctx->saved_data["seed"] = seed;
    ctx->saved_data["offset"] = offset;
    ctx->saved_data["numels"] = numels;
    ctx->saved_data["atten_mask_is_none"] = !atten_mask_opt.has_value();
    ctx->saved_data["alibi_mask_is_none"] = !alibi_mask_opt.has_value();
    ctx->saved_data["drop_mask_is_none"] = drop_is_none;

    return results;
  }

  static std::vector<at::Tensor> backward(torch::autograd::AutogradContext *ctx, std::vector<at::Tensor> grad_outputs)
  {
    auto scale_value = ctx->saved_data["scale_value"].toDouble();
    auto q_scale = ctx->saved_data["q_scale"].toDouble();
    auto head_num = ctx->saved_data["head_num"].toInt();
    auto sync = ctx->saved_data["sync"].toBool();
    auto gen_mask_parallel = ctx->saved_data["gen_mask_parallel"].toBool();
    auto layout = ctx->saved_data["layout"].toInt();
    auto groups = ctx->saved_data["groups"].toInt();
    auto keep_prob = ctx->saved_data["keep_prob"].toDouble();
    auto pre_tokens = ctx->saved_data["pre_tokens"].toInt();
    auto next_tokens = ctx->saved_data["next_tokens"].toInt();
    auto precise_mode = ctx->saved_data["precise_mode"].toInt();
    auto seed = ctx->saved_data["seed"].toInt();
    auto offset = ctx->saved_data["offset"].toInt();
    auto numels = ctx->saved_data["numels"].toInt();
    auto atten_mask_is_none = ctx->saved_data["atten_mask_is_none"].toBool();
    auto alibi_mask_is_none = ctx->saved_data["alibi_mask_is_none"].toBool();
    auto drop_mask_is_none = ctx->saved_data["drop_mask_is_none"].toBool();

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

    auto results = fag(grad_outputs[0], softmax_log_max_sum, attention_score, query, key, value,
                       atten_mask, alibi_mask, drop_mask, scale_value, head_num, layout, 
                       keep_prob, pre_tokens, next_tokens, precise_mode, groups, atten_mask_is_none, alibi_mask_is_none, drop_mask_is_none);

    return results;
  }
};

std::vector<at::Tensor> ascend_flash_attention(
      const at::Tensor &query, const at::Tensor &key,
      const at::Tensor &value, const c10::optional<at::Tensor> &atten_mask, const c10::optional<at::Tensor> &alibi_mask,
      float scale_value, float q_scale, int64_t head_num, std::string io_layout, 
      float keep_prob, int64_t pre_tokens, int64_t next_tokens, int64_t precise_mode)
{
  return NPUAscendFlashAttentionFunction::apply(query, key, value, atten_mask, alibi_mask, scale_value, q_scale, head_num,
                                                io_layout, keep_prob, pre_tokens, next_tokens, precise_mode);
}
