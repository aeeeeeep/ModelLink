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
#ifndef INTERNLM_20B_FLASH_ATTENTION_ROPE_ANTIOUTLIER_LAYER_H
#define INTERNLM_20B_FLASH_ATTENTION_ROPE_ANTIOUTLIER_LAYER_H

#include <atb/atb_infer.h>
#include <atb/svector.h>

#include "atb_speed/base/hosttensor_binder.h"
#include "atb_speed/log.h"
#include "atb_speed/utils/str_split.h"

namespace atb_speed {
namespace internlm_20b {
struct FlashAttentionRopeAntiOutlierLayerParam {
    float rmsNormEps = 0; // 模型config：rms_norm_eps
    int headNum = 0;      // 计算方式：（config.num_attention_heads // rankSize）
    int dk = 0;           // 计算方式：（config.hidden_size // config.num_attention_heads）
    int rank = 0;         // 多卡并行模型id
    int rankSize = 1;     // 模型切分数量
    std::string backend = "hccl";
    std::string model = "internlm_20b";
};

void from_json(const nlohmann::json &paramJson, FlashAttentionRopeAntiOutlierLayerParam &param);

atb::Status FlashAttentionRopeAntiOutlierLayer(const FlashAttentionRopeAntiOutlierLayerParam &param,
                                               atb::Operation **operation);

atb::Operation *CreateFlashAttentionRopeAntiOutlierLayer(const nlohmann::json &paramJson);

class FlashAttentionRopeAntiOutlierLayerBinder : public HostTensorBinder {
public:
    FlashAttentionRopeAntiOutlierLayerBinder();

    ~FlashAttentionRopeAntiOutlierLayerBinder() override;

    void ParseParam(const nlohmann::json &paramJson) override;

    void BindTensor(atb::VariantPack &variantPack) override;

private:
    std::vector<int32_t> tokenOffset_;
    std::vector<int32_t> seqLen_;
};
} // namespace internlm_20b
} // namespace atb_speed
#endif