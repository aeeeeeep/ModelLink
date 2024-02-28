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
#ifndef ATB_SPEED_MODELS_LLAMA2_70B_PA_LAYER_EMBEDDING_H
#define ATB_SPEED_MODELS_LLAMA2_70B_PA_LAYER_EMBEDDING_H

#include "atb/atb_infer.h"
#include "atb_speed/log.h"
#include "nlohmann/json.hpp"

namespace atb_speed {
namespace llama2_70b {
struct PALayerEmbeddingParam {
    int axis = 0;
    int rank = 0;
    int rankSize = 0;
    int rankRoot = 0;
    std::string backend = "hccl";
};

atb::Status PALayerEmbedding(const PALayerEmbeddingParam &param, atb::Operation **operation);
} // namespace llama2_70b
} // namespace atb_speed
#endif
