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

#ifndef ATB_SPEED_MODELS_LLAMA_FAMILY_ATTENTION_H
#define ATB_SPEED_MODELS_LLAMA_FAMILY_ATTENTION_H

#define QKV_TENSOR_IDX(x) static_cast<uint32_t>(QKVLinearSplitTensorIdx::x)
#define SELF_ATTENTION_TENSOR_IDX(x) static_cast<uint32_t>(SelfAttentionTensorIdx::x)
#define ATTENTION_TENSOR_IDX(x) static_cast<uint32_t>(AttentionTensorIdx::x)

#include <atb/atb_infer.h>
#include "atb_speed/log.h"

namespace atb_speed {
namespace llama_family {
struct FusionAttentionParam {
    bool isFA = true;
    // QKV linear param
    bool isPack = true;
    int isGroupedQueryAttention = false;
    atb_speed::llama_family::FusionLinearParam qkvLinearParam;
    // rope param
    int rotaryCoeff = 2;
    // self attention param
    bool isPrefill = false;
    bool isBF16 = false;
    atb::infer::SelfAttentionParam selfAttentionParam;
    atb::infer::PagedAttentionParam pageAttentionParam;
    // self out linear param
    atb_speed::llama_family::LinearParallelParam selfOutLinearParallelParam;
};

class FusionAttention {
public:
    virtual atb::Status Attention(const FusionAttentionParam &param, atb::Operation **operation) final;
    virtual atb::Status QKVLinearSplit(const FusionAttentionParam &param, atb::Operation **operation) final;
    virtual atb::Status SelfAttention(const FusionAttentionParam &param, atb::Operation **operation) final;
    std::shared_ptr<int64_t> batchNumPtr = std::make_shared<int64_t>(0);
};
} // namespace llama_family
} // namespace atb_speed
#endif