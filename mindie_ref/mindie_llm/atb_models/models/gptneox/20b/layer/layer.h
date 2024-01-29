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
#ifndef ATB_SPEED_MODELS_GPTNEOX_20B_LAYER_H
#define ATB_SPEED_MODELS_GPTNEOX_20B_LAYER_H

#include <nlohmann/json.hpp>
#include <atb/atb_infer.h>
#include "atb_speed/log.h"
#include "atb_speed/base/hosttensor_binder.h"

namespace atb_speed {
namespace gptneox_20b {
struct LayerParam {
    float layerNormEps = 0;
    int headNum = 0;
    int dk = 0;
    float rotaryPct = 0.0;
    float qScale = 1.0;
    float qkScale = 1.0;
    bool transposedWeight = false;
    std::string model = "gptneox_20b";
    bool isPrefill = false;
    int rank = 0;
    int rankSize = 1;
};

struct EmbeddingLayerParam {
    int axis = 0;
};

atb::Status EmbeddingLayer(const EmbeddingLayerParam &param, atb::Operation **operation);

atb::Status FlashAttentionKvCacheLayer(const LayerParam &param, atb::Operation **operation);

static atb::Operation *CreateFlashAttentionKvCacheLayer(const nlohmann::json &paramJson)
{
    LayerParam param;
    param.layerNormEps = paramJson["layerNormEps"].get<float>();
    param.headNum = paramJson["headNum"].get<int>();
    param.dk = paramJson["dk"].get<int>();
    param.model = paramJson["model"].get<std::string>();
    param.qScale = paramJson["qScale"].get<float>();
    if (paramJson.contains("rotaryPct")) {
        param.rotaryPct = paramJson["rotaryPct"].get<float>();
    }
    if (paramJson.contains("isPrefill")) {
        param.isPrefill = paramJson["isPrefill"].get<bool>();
    }
    if (paramJson.contains("rank")) {
        param.rank = paramJson["rank"].get<int>();
    }
    if (paramJson.contains("rankSize")) {
        param.rankSize = paramJson["rankSize"].get<int>();
    }
    if (paramJson.contains("qkScale")) {
        param.qkScale = paramJson["qkScale"].get<int>();
    }

    ATB_LOG(INFO) << __func__ << " layerNormEps:" << param.layerNormEps << ", headNum:" << param.headNum << ", dk:" <<
        param.dk << ", model:" << param.model;
    atb::Operation *op;
    FlashAttentionKvCacheLayer(param, &op);
    return op;
}

class FlashAttentionHostBinder : public HostTensorBinder {
public:
    FlashAttentionHostBinder();

    virtual ~FlashAttentionHostBinder();

    void ParseParam(const nlohmann::json &paramJson) override;

    void BindTensor(atb::VariantPack &variantPack) override;

private:
    atb::SVector<int64_t> tokenOffset_;
    atb::SVector<int64_t> seqLen_;
};
} // namespace gptneox_20b
} // namespace atb_speed
#endif