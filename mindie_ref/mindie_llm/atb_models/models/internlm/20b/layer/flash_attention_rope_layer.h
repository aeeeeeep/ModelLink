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
#ifndef INTERNLM_20B_FLASH_ATTENTION_ROPE_LAYER_H
#define INTERNLM_20B_FLASH_ATTENTION_ROPE_LAYER_H

#include <atb/atb_infer.h>
#include <atb/svector.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtype-limits"
#include "nlohmann/json.hpp"
#pragma GCC diagnostic pop
#include "atb_speed/base/hosttensor_binder.h"

namespace atb_speed {
namespace internlm_20b {
struct FlashAttentionLayerParam {
    float rmsNormEps = 0;
    int headNum = 0;
    int kvHeadNum = 0;
    int dk = 0; // headDim
    int rank = 0;
    int rankSize = 1;
    int isTriuMask = 0;
    std::string backend = "hccl";
    std::string model = "internlm20b";
    float qScale = 1.0;
    bool quantModel = false;
    bool sparseModel = false;
    bool isEncoder = false;
    bool isBF16 = false;
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

atb::Status FlashAttentionLayer(const FlashAttentionLayerParam &param, atb::Operation **operation);

class FlashAttentionRopeLayerBinder : public HostTensorBinder {
public:
    FlashAttentionRopeLayerBinder();
    virtual ~FlashAttentionRopeLayerBinder();
    void ParseParam(const nlohmann::json &paramJson) override;
    void BindTensor(atb::VariantPack &variantPack) override;

private:
    std::vector<int32_t> tokenOffset_;
    std::vector<int32_t> seqLen_;
};
} // namespace internlm_20b
} // namespace atb_speed
#endif