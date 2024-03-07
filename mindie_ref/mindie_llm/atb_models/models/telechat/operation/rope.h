/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,

 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ATB_SPEED_MODELS_TELECHAT_ROPE_H
#define ATB_SPEED_MODELS_TELECHAT_ROPE_H

#include "atb/atb_infer.h"
#include "atb_speed/log.h"
#include <functional>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtype-limits"
#include "nlohmann/json.hpp"
#pragma GCC diagnostic pop

namespace atb_speed {
namespace telechat {
struct RopeParam {
    int32_t rotaryCoeff = 2;
    int32_t headNum = 0;
};

atb::Status Rope(const RopeParam &param, atb::Operation **operation);

} // namespace telechat
} // namespace atb_speed
#endif