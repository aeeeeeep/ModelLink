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
#ifndef ATB_SPEED_MODELS_LLAMA_ADAPTER_SELF_ATTETNTION_CROSS_H
#define ATB_SPEED_MODELS_LLAMA_ADAPTER_SELF_ATTETNTION_CROSS_H

#include "atb/atb_infer.h"
#include "atb_speed/log.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtype-limits"
#include "nlohmann/json.hpp"
#pragma GCC diagnostic pop

namespace atb_speed {
namespace llama_adapter {
struct SelfAttentionCrossParam {
    int64_t dk = 0;
    int64_t headNum = 0;
    std::string model = "llama_adapter";
};

atb::Status SelfAttentionCrossEn(const SelfAttentionCrossParam &param, atb::Operation **operation);

atb::Status SelfAttentionCrossEnAdapter(const SelfAttentionCrossParam &param, atb::Operation **operation);

atb::Status SelfAttentionCrossDe(const SelfAttentionCrossParam &param, atb::Operation **operation);

atb::Status SelfAttentionCrossDeAdapter(const SelfAttentionCrossParam &param, atb::Operation **operation);

} // namespace llama_adapter
} // namespace atb_speed
#endif