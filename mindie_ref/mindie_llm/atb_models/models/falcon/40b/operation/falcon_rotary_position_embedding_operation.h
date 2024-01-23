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

#ifndef ATB_SPEED_MODELS_FALCON_40B_ROTARY_POSITION_EMBEDDING_H
#define ATB_SPEED_MODELS_FALCON_40B_ROTARY_POSITION_EMBEDDING_H
#include "atb/atb_infer.h"
#include "atb_speed/log.h"
#include "nlohmann/json.hpp"

namespace atb_speed {
namespace falcon_40b {
struct RotaryPositionEmbeddingParam {
    int32_t hiddenSize = 8192;  // hidden_size
    int32_t headNum = 128;   // config.num_attention_heads
    int32_t kvHeadNum = 8;     // config.num_kv_heads
    int32_t headDim = 64;
};

atb::Status RotaryPositionEmbedding(const RotaryPositionEmbeddingParam &param, atb::Operation **operation);

static atb::Operation *CreateRotaryPositionEmbedding(const nlohmann::json &paramJson)
{
    RotaryPositionEmbeddingParam param;
    if (paramJson.contains("hiddenSize")) {
        param.hiddenSize = paramJson["hiddenSize"].get<int>();
    }
    if (paramJson.contains("headNum")) {
        param.headNum = paramJson["headNum"].get<int>();
    }
    if (paramJson.contains("headDim")) {
        param.headDim = paramJson["headDim"].get<int>();
    }
    if (paramJson.contains("kvHeadNum")) {
        param.kvHeadNum = paramJson["kvHeadNum"].get<int>();
    }

    ATB_LOG(INFO) << "Falcon RotaryPositionEmbeddingParam hiddenSize:" << param.hiddenSize
                  << ", headNum:" << param.headNum << ", kvHeadNum:" << param.kvHeadNum;
    atb::Operation *op;
    RotaryPositionEmbedding(param, &op);
    return op;
}
} // namespace falcon_40b
} // namespace atb_speed
#endif
