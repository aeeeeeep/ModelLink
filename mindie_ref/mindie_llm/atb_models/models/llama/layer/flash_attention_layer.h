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
#ifndef ATB_SPEED_MODELS_LLAMA_FLASHATTENTION_LAYER_OPERATION_H
#define ATB_SPEED_MODELS_LLAMA_FLASHATTENTION_LAYER_OPERATION_H

#include <atb/atb_infer.h>
#include <atb/svector.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtype-limits"
#include "nlohmann/json.hpp"
#pragma GCC diagnostic pop
#include "atb_speed/base/hosttensor_binder.h"

namespace atb_speed {
namespace llama {
struct FlashAttentionLayerParam {
    float rmsNormEps = 0;
    int headNum = 0;
    int kvHeadNum = 0;
    int dk = 0; // headDim
    int rank = 0;
    int rankSize = 1;
    int isTriuMask = 0;
    std::string backend = "hccl";
    std::string model = "llama_13b";
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

class FlashAttentionLayerBinder : public HostTensorBinder {
public:
    FlashAttentionLayerBinder();
    virtual ~FlashAttentionLayerBinder();
    void ParseParam(const nlohmann::json &paramJson) override;
    void BindTensor(atb::VariantPack &variantPack) override;

private:
    std::vector<int32_t> tokenOffset_;
    std::vector<int32_t> seqLen_;
};
} // namespace llama
} // namespace atb_speed
#endif