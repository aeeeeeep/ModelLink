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
#ifndef ATB_SPEED_MODELS_GLM130B_LMHEAD_H
#define ATB_SPEED_MODELS_GLM130B_LMHEAD_H
#include <atb/atb_infer.h>
#include "atb_speed/log.h"

namespace atb_speed {
namespace glm130b {
struct LmHeadParam {
    int rank = 0;
    int rankSize = 0;
    int rankRoot = 0;
    std::string backend = "hccl";
    atb::SVector<int32_t> perm = {1, 2, 0, 3};
};

atb::Status CreateLmHead(const LmHeadParam &param, atb::Operation **operation);
} // namespace glm130b
} // namespace atb_speed
#endif