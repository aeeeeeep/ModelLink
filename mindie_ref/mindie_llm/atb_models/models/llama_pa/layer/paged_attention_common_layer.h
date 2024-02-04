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
#ifndef ATB_SPEED_MODELS_LLAMA_COMMON_PA_LAYER_H
#define ATB_SPEED_MODELS_LLAMA_COMMON_PA_LAYER_H

#include <atb/atb_infer.h>
#include <nlohmann/json.hpp>

#include "atb_speed/base/hosttensor_binder.h"
#include "atb_speed/log.h"
#include "atb_speed/utils/str_split.h"

namespace atb_speed {
namespace llama_pa {
struct PaCommonLayerParam {
    int rank = 0;
    int rankSize = 1;
    float rmsNormEps = 0;
    int headNum = 0;
    int dk = 0;
    bool transposedWeight = false;
    bool isPrefill = false;
    std::string backend = "hccl";
    std::string model = "llama_small_pa";
    bool isBF16 = false;
    bool isQuant = false;   // 量化开关
    // 量化参数
    float qkvInputScale = 1;
    int qkvInputOffset = 0;
    float denseInputScale = 1;
    int denseInputOffset = 0;
    float selfLnInputScale = 1;
    int selfLnInputOffset = 0;
    float ffnOutInputScale = 1;
    int ffnOutInputOffset = 0;
};

void from_json(const nlohmann::json &paramJson, PaCommonLayerParam &param);

atb::Status PaCommonLayer(const PaCommonLayerParam &param, atb::Operation **operation);

class CommonFlashAttentionHostBinder : public HostTensorBinder {
public:
    CommonFlashAttentionHostBinder();

    virtual ~CommonFlashAttentionHostBinder();

    void ParseParam(const nlohmann::json &paramJson) override;

    void BindTensor(atb::VariantPack &variantPack) override;

private:
    atb::SVector<int32_t> seqLen_;
};

} // namespace llama_pa
} // namespace atb_speed
#endif