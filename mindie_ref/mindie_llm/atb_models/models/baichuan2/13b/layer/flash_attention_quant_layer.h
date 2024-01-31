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
#ifndef BAICHUAN2_13B_FLASH_ATTENTION_QUANT_LAYER_H
#define BAICHUAN2_13B_FLASH_ATTENTION_QUANT_LAYER_H

#include <atb/atb_infer.h>
#include <atb/svector.h>

#include "atb_speed/base/hosttensor_binder.h"
#include "atb_speed/log.h"
#include "atb_speed/utils/str_split.h"

namespace atb_speed {
namespace baichuan2_13b {
struct FlashAttentionQuantLayerParam {
    float rmsNormEps = 0;
    int headNum = 0;
    int dk = 0;
    int rank = 0;
    int rankSize = 1;
    std::string backend = "hccl";
    float wPackInputScale = 1;
    int wPackInputOffset = 0;
    float oProjInputScale = 1;
    int oProjInputOffset = 0;
    float gateProjInputScale = 1;
    int gateProjInputOffset = 0;
    float downProjInputScale = 1;
    int downProjInputOffset = 0;
    float upProjInputScale = 1;
    int upProjInputOffset = 0;
    std::string model = "baichuan2_13b";
};

void from_json(const nlohmann::json &paramJson, FlashAttentionQuantLayerParam &param);

atb::Status FlashAttentionQuantLayer(const FlashAttentionQuantLayerParam &param, atb::Operation **operation);

atb::Operation *CreateFlashAttentionQuantLayer(const nlohmann::json &paramJson);

class FlashAttentionQuantLayerBinder : public HostTensorBinder {
public:
    FlashAttentionQuantLayerBinder();
    ~FlashAttentionQuantLayerBinder() override;
    void ParseParam(const nlohmann::json &paramJson) override;
    void BindTensor(atb::VariantPack &variantPack) override;

private:
    std::vector<int32_t> tokenOffset_;
    std::vector<int32_t> seqLen_;
};
} // namespace baichuan2_13b
} // namespace atb_speed
#endif