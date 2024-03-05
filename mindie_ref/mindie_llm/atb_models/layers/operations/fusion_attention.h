/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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

#ifndef ATB_SPEED_MODELS_COMMON_ATTENTION_H
#define ATB_SPEED_MODELS_COMMON_ATTENTION_H

#include <atb/atb_infer.h>
#include "atb_speed/log.h"
#include "layers/operations/linear.h"
#include "layers/operations/linear_parallel.h"

namespace atb_speed {
namespace common {
struct FusionAttentionParam {
    bool isFA = true;
    // QKV linear param
    bool isPack = true;
    int isGroupedQueryAttention = false;
    atb_speed::common::FusionLinearParam qkvLinearParam;
    // rope param
    int rotaryCoeff = 2;
    // self attention param
    bool isPrefill = false;
    bool isBF16 = false;
    int faHeadDim = 0;
    atb::infer::SelfAttentionParam selfAttentionParam;
    atb::infer::PagedAttentionParam pageAttentionParam;
    // self out linear param
    atb_speed::common::LinearParallelParam selfOutLinearParallelParam;
};

class FusionAttention {
public:
    static atb::Status Attention(const FusionAttentionParam &param, atb::Operation **operation);
    static atb::Status QKVLinearSplit(const FusionAttentionParam &param, atb::Operation **operation);
    static atb::Status SelfAttention(const FusionAttentionParam &param, atb::Operation **operation);
};
} // namespace common
} // namespace atb_speed
#endif