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
#ifndef ATB_SPEED_MODELS_GPTNEOX_20B_FLASHATTENTION_KVCACHE_ROPE_LAYER_H
#define ATB_SPEED_MODELS_GPTNEOX_20B_FLASHATTENTION_KVCACHE_ROPE_LAYER_H

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtype-limits"
#include <nlohmann/json.hpp>
#pragma GCC diagnostic pop
#include <atb/atb_infer.h>
#include "atb_speed/log.h"
#include "atb_speed/base/hosttensor_binder.h"

namespace atb_speed {
namespace gptneox_20b {
struct FlashAttentionKvCacheRopeParam {
    float layerNormEps = 0;
    int headNum = 0;
    int dk = 0;
    float rotaryPct = 0.0;
    float qScale = 1.0;
    float qkScale = 1.0;
    bool transposedWeight = true;
    std::string model = "gptneox_20b";
    bool isPrefill = false;
    int rank = 0;
    int rankSize = 1;
    std::string backend = "hccl";
};

atb::Status FlashAttentionKvCacheRopeLayer(const FlashAttentionKvCacheRopeParam &param, atb::Operation **operation);

atb::Operation *CreateFlashAttentionKvCacheRopeLayer(const nlohmann::json &paramJson);

class FlashAttentionRopeHostBinder : public HostTensorBinder {
public:
    FlashAttentionRopeHostBinder();

    ~FlashAttentionRopeHostBinder() override;

    void ParseParam(const nlohmann::json &paramJson) override;

    void BindTensor(atb::VariantPack &variantPack) override;

private:
    atb::SVector<int64_t> tokenOffset_;
    atb::SVector<int64_t> seqLen_;
};
} // namespace gptneox_20b
} // namespace atb_speed

#endif // ATB_SPEED_MODELS_GPTNEOX_20B_FLASHATTENTION_KVCACHE_ROPE_LAYER_H
