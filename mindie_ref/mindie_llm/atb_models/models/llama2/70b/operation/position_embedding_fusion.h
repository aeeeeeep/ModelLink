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
#ifndef ATB_SPEED_MODELS_LLAMA_POSITION_EMBEDDING_1DSPLIT_FUSION_OPERATION_H
#define ATB_SPEED_MODELS_LLAMA_POSITION_EMBEDDING_1DSPLIT_FUSION_OPERATION_H
#include <atb/atb_infer.h>
#include "nlohmann/json.hpp"

namespace atb_speed {
namespace llama2_70b {
struct PositionEmbedding1dFusionParam {
    std::string model = "llama2_70b";
    int64_t rotaryCoeff = 2;
    int64_t headNum = 0;
};

atb::Status PositionEmbeddingFusionOperation(const PositionEmbedding1dFusionParam &param,
                                             atb::Operation **operation);

} // namespace llama2_70b
} // namespace atb_speed
#endif