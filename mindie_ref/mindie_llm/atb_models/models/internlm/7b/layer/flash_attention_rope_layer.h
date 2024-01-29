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
#ifndef INTERNLM_7B_FLASH_ATTENTION_ROPE_LAYER_H
#define INTERNLM_7B_FLASH_ATTENTION_ROPE_LAYER_H

#include <atb/atb_infer.h>
#include <atb/svector.h>

#include "atb_speed/base/hosttensor_binder.h"
#include "atb_speed/log.h"
#include "atb_speed/utils/str_split.h"

namespace atb_speed {
namespace internlm_7b {
struct FlashAttentionRopeLayerParam {
    float rmsNormEps = 0;
    int headNum = 0;
    int dk = 0;
    int rank = 0;
    int rankSize = 1;
    std::string model = "internlm_7b";
};

atb::Status FlashAttentionRopeLayer(const FlashAttentionRopeLayerParam &param, atb::Operation **operation);

void from_json(const nlohmann::json &paramJson, FlashAttentionRopeLayerParam &param);

atb::Operation *CreateFlashAttentionRopeLayer(const nlohmann::json &paramJson);

class FlashAttentionRopeLayerBinder : public HostTensorBinder {
public:
    FlashAttentionRopeLayerBinder();
    ~FlashAttentionRopeLayerBinder() override;
    void ParseParam(const nlohmann::json &paramJson) override;
    void BindTensor(atb::VariantPack &variantPack) override;

private:
    std::vector<int32_t> tokenOffset_;
    std::vector<int32_t> seqLen_;
};
} // namespace internlm_7b
} // namespace atb_speed
#endif