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
#ifndef ATB_SPEED_MODELS_FUSION_MLP_H
#define ATB_SPEED_MODELS_FUSION_MLP_H

#include "atb/atb_infer.h"
#include "atb_speed/log.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtype-limits"
#include <nlohmann/json.hpp>
#pragma GCC diagnostic pop
#include "atb_speed/log.h"
#include "common.h"

namespace atb_speed {
namespace glm130b {

struct FusionMlpGlmParam {
    bool transpose = false;
    int rank = 0;
    int rankSize = 1;
    int rankRoot = 0;
    void *hcclComm = nullptr;
    atb::infer::ActivationType activationType;
    bool transposeB = false;
    bool isBias = false;
    bool isPack = false;
    std::string backend = "hccl";
};

atb::Status FusionMlpGlm(const FusionMlpGlmParam &param, atb::Operation **operation);

} // namespace llama_7b
} // namespace atb_speed
#endif