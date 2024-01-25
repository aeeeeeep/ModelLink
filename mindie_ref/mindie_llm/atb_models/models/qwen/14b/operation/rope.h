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
#ifndef ATB_SPEED_MODELS_QWEN_14B_ROPE_H
#define ATB_SPEED_MODELS_QWEN_14B_ROPE_H
#include <atb/atb_infer.h>
#include <nlohmann/json.hpp>

#include "atb_speed/log.h"
#include "atb_speed/utils/str_split.h"

namespace atb_speed {
namespace qwen_14b {
struct RopeParam {
    int64_t rotaryCoeff = 2;
    int64_t headNum = 0;
};

atb::Status Rope(const RopeParam &param, atb::Operation **operation);

static atb::Operation *CreateRope(const nlohmann::json &paramJson)
{
    RopeParam param;
    if (paramJson.contains("rotaryCoeff")) {
        param.rotaryCoeff = paramJson["rotaryCoeff"].get<int>();
    }
    if (paramJson.contains("headNum")) {
        param.headNum = paramJson["headNum"].get<int>();
    }
    atb::Operation *op;
    Rope(param, &op);
    return op;
}

} // namespace qwen_14b
} // namespace atb_speed