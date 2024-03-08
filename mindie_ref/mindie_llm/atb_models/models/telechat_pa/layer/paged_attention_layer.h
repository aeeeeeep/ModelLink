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

#ifndef ATB_SPEED_MODELS_TELECHAT_COMMON_PA_LAYER_H
#define ATB_SPEED_MODELS_TELECHAT_COMMON_PA_LAYER_H

#include <atb/atb_infer.h>
#include <nlohmann/json.hpp>

#include "atb_speed/base/hosttensor_binder.h"
#include "atb_speed/log.h"
#include "atb_speed/utils/str_split.h"

namespace atb_speed {
namespace telechat {
struct PALayerParam {
    int rank = 0;
    int rankSize = 1;
    float rmsNormEps = 0;
    int headNum = 0;
    int dk = 0; // headDim
    bool transposedWeight = true;
    bool isPrefill = false;
    std::string backend = "hccl";
    std::string model = "telechat_7B";
    bool isQuant = false;   // 量化开关
};

void from_json(const nlohmann::json &paramJson, PALayerParam &param);

atb::Status PALayer(const PALayerParam &param, atb::Operation **operation);

class FlashAttentionHostBinder : public HostTensorBinder {
public:
    FlashAttentionHostBinder();

    virtual ~FlashAttentionHostBinder();

    void ParseParam(const nlohmann::json &paramJson) override;

    void BindTensor(atb::VariantPack &variantPack) override;

private:
    atb::SVector<int32_t> seqLen_;
};

} // namespace telechat
} // namespace atb_speed
#endif