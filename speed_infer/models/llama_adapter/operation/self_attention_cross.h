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
#ifndef ATB_SPEED_MODELS_LLAMA_ADAPTER_SELF_ATTETNTION_CROSS_H
#define ATB_SPEED_MODELS_LLAMA_ADAPTER_SELF_ATTETNTION_CROSS_H

#include "atb/atb_infer.h"
#include "atb_speed/log.h"
#include "nlohmann/json.hpp"

namespace atb_speed {
namespace llama_adapter {
struct SelfAttentionCrossParam {
    int64_t dk = 0;
    int64_t headNum = 0;
    std::string model = "llama_adapter";
};

atb::Status SelfAttentionCrossEn(const SelfAttentionCrossParam &param, atb::Operation **operation);

atb::Status SelfAttentionCrossEnAdapter(const SelfAttentionCrossParam &param, atb::Operation **operation);

atb::Status SelfAttentionCrossDe(const SelfAttentionCrossParam &param, atb::Operation **operation);

atb::Status SelfAttentionCrossDeAdapter(const SelfAttentionCrossParam &param, atb::Operation **operation);

inline static atb::Operation *CreateSelfAttentionCrossEn(const nlohmann::json &paramJson)
{
    SelfAttentionCrossParam param;
    if (paramJson.contains("dk")) {
        param.dk = paramJson["dk"].get<int>();
    }
    if (paramJson.contains("headNum")) {
        param.headNum = paramJson["headNum"].get<int>();
    }
    if (paramJson.contains("model")) {
        param.model = paramJson["model"].get<std::string>();
    }

    ATB_LOG(INFO) << "LLaMA_adapter_SelfAttentionCrossEnParam headNum:" << param.headNum << ", dk:" << param.dk <<
        ", model:" << param.model;
    atb::Operation *op;
    SelfAttentionCrossEn(param, &op);
    return op;
}

inline static atb::Operation *CreateSelfAttentionCrossEnAdapter(const nlohmann::json &paramJson)
{
    SelfAttentionCrossParam param;
    if (paramJson.contains("dk")) {
        param.dk = paramJson["dk"].get<int>();
    }
    if (paramJson.contains("headNum")) {
        param.headNum = paramJson["headNum"].get<int>();
    }
    if (paramJson.contains("model")) {
        param.model = paramJson["model"].get<std::string>();
    }
    ATB_LOG(INFO) << "LLaMA_adapter_SelfAttentionCrossEnAdapterParam headNum:" << param.headNum << ", dk:" <<
        param.dk << ", model:" << param.model;
    atb::Operation *op;
    SelfAttentionCrossEnAdapter(param, &op);
    return op;
}

inline static atb::Operation *CreateSelfAttentionCrossDe(const nlohmann::json &paramJson)
{
    SelfAttentionCrossParam param;
    if (paramJson.contains("dk")) {
        param.dk = paramJson["dk"].get<int>();
    }
    if (paramJson.contains("headNum")) {
        param.headNum = paramJson["headNum"].get<int>();
    }
    if (paramJson.contains("model")) {
        param.model = paramJson["model"].get<std::string>();
    }

    ATB_LOG(INFO) << "LLaMA_adapter_SelfAttentionCrossDeParam headNum:" << param.headNum << ", dk:" << param.dk <<
        ", model:" << param.model;
    atb::Operation *op;
    SelfAttentionCrossDe(param, &op);
    return op;
}

inline static atb::Operation *CreateSelfAttentionCrossDeAdapter(const nlohmann::json &paramJson)
{
    SelfAttentionCrossParam param;
    if (paramJson.contains("dk")) {
        param.dk = paramJson["dk"].get<int>();
    }
    if (paramJson.contains("headNum")) {
        param.headNum = paramJson["headNum"].get<int>();
    }
    if (paramJson.contains("model")) {
        param.model = paramJson["model"].get<std::string>();
    }
    ATB_LOG(INFO) << "LLaMA_adapter_SelfAttentionCrossDeAdapterParam headNum:" << param.headNum << ", dk:" <<
        param.dk << ", model:" << param.model;
    atb::Operation *op;
    SelfAttentionCrossDeAdapter(param, &op);
    return op;
}
} // namespace llama_adapter
} // namespace atb_speed
#endif