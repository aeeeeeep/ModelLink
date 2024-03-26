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
#ifndef ATB_SPEED_MODELS_COMMON_LINEAR_H
#define ATB_SPEED_MODELS_COMMON_LINEAR_H

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtype-limits"
#include "nlohmann/json.hpp"
#pragma GCC diagnostic pop
#include "atb/atb_infer.h"
#include "atb_speed/log.h"

namespace atb_speed {
namespace common {

enum LinearQuantType : unsigned int {
    LINEAR_NO_QUANT = 0,
    LINEAR_W8A8_DEQUANT,  // QUANT在RMS_NORM中执行，DEQUANT在此operaion中执行
    LINEAR_W8A8_QUANT,         // QUANT和DEQUANT操作都在此Operation中执行
    LINEAR_W8A16_QUANT,
    LINEAR_W8A8_SC_DEQUANT,
    LINEAR_W8A8_SC_QUANT,
};

enum LinearType : int {
    INVALID = -1,
    FP = 0,
    INT = 1,
};

struct FusionLinearParam {
    LinearQuantType quantType = LINEAR_NO_QUANT;
    bool isBF16 = false;
    bool hasBias = false;
};

atb::Status FusionLinear(const FusionLinearParam &param, atb::Operation **operation);
} // namespace common
} // namespace atb_speed
#endif
