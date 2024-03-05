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
#ifndef ATB_SPEED_MODELS_COMMON_LAYER_POSITIONAL_EMBEDDING_H
#define ATB_SPEED_MODELS_COMMON_LAYER_POSITIONAL_EMBEDDING_H

#include "nlohmann/json.hpp"
#include "atb/atb_infer.h"
#include "atb_speed/log.h"
#include "layers/operations/linear_parallel.h"

namespace atb_speed {
namespace common {

struct RotaryPositionEmbeddingParam {
    bool isHalfRotary = false;
    bool isFA = true;
    int headNum = 0;
    int headDim = 0;
    int kvHeadNum = 0;
    int rotaryCoeff = 2;
};
atb::Status RotaryPositionEmbedding(const RotaryPositionEmbeddingParam &param, atb::Operation **operation);

atb::Status PEGather(atb::Operation **operation);
}  // namespace common
}  // namespace atb_speed
#endif
