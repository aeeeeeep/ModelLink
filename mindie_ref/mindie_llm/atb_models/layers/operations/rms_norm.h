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
#ifndef ATB_SPEED_MODELS_COMMON_RMS_NORM_H
#define ATB_SPEED_MODELS_COMMON_RMS_NORM_H

#include "nlohmann/json.hpp"
#include "atb/atb_infer.h"
#include "atb_speed/log.h"
#include "layers/operations/linear.h"

namespace atb_speed {
namespace common {

struct FusionRmsNormParam {
    int quantType = atb_speed::common::NO_QUANT;
    float rmsNormEps = 0;
    float quantInputScale = 1.0f;
    int quantInputOffset = 0;
};

atb::Status FusionRmsNorm(const FusionRmsNormParam &param, atb::Operation **operation);
} // namespace common
} // namespace atb_speed
#endif
