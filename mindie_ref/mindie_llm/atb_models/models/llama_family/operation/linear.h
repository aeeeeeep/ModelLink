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
#ifndef ATB_SPEED_MODELS_LLAMA_FAMILY_LINEAR_H
#define ATB_SPEED_MODELS_LLAMA_FAMILY_LINEAR_H

#include "nlohmann/json.hpp"
#include "atb/atb_infer.h"
#include "atb_speed/log.h"

namespace atb_speed {
namespace llama_family {

enum LinearQuantType : unsigned int {
    NO_QUANT = 0,
    RMS_NORM_QUANT_LINEAR_DEQUANT = 1,  // QUANT在RMS_NORM中执行，DEQUANT在此operaion中执行
    LINEAR_QUANT = 2,         // QUANT和DEQUANT操作都在此Operation中执行
};

struct FusionLinearParam {
    int quantType = NO_QUANT;
};

atb::Status FusionLinear(const FusionLinearParam &param, atb::Operation **operation);
} // namespace llama_family
} // namespace atb_speed
#endif
