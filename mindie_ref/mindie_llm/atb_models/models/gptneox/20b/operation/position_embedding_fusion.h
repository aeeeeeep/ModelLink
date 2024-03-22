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
#ifndef ATB_SPEED_MODELS_GPTNEOX_20B_OPERATION_POSITION_EMBEDDING_FUSION_OPERATION_H
#define ATB_SPEED_MODELS_GPTNEOX_20B_OPERATION_POSITION_EMBEDDING_FUSION_OPERATION_H

#include "atb/atb_infer.h"
#include "atb_speed/log.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtype-limits"
#include "nlohmann/json.hpp"
#pragma GCC diagnostic pop

namespace atb_speed {
namespace gptneox_20b {
struct PositionEmbeddingFusionParam {
    int32_t headNum = 0;
    int32_t dk = 0;
    float rotaryPct = 0.25;
};

atb::Status PositionEmbeddingFusionOperation(const PositionEmbeddingFusionParam &param, atb::Operation **operation);

atb::Operation *CreatePositionEmbeddingFusionOperation(const nlohmann::json &paramJson);
} // namespace gpt_neox20b
} // namespace atb_speed

#endif // ATB_SPEED_MODELS_GPTNEOX_20B_OPERATION_POSITION_EMBEDDING_FUSION_OPERATION_H
