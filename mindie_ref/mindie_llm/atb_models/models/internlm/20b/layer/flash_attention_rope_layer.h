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

#include "atb_speed/base/hosttensor_binder.h"
#include "atb_speed/log.h"
#include "atb_speed/utils/str_split.h"

namespace atb_speed {
namespace internlm_20b {
struct FlashAttentionRopeLayerParam {
    float rmsNormEps = 0;
    int headNum = 0;
    int dk = 0;
    int rank = 0;
    int rankSize = 1;
    std::string backend = "hccl";
    std::string model = "internlm_20b";
};

struct SelfAttentionParam {
    int32_t headDim = 0;
    int32_t headNum = 0;
    uint32_t isTriuMask = 0;
    float qScale = 1;  // qtensor scale before qkbmm
    float qkScale = 1; // scale after qkbmm
    bool batchRunStatusEnable = false;
    int32_t kvHeadNum = 0;
    bool isEncoder = false; // encoder for pagedAttention
    enum CoderType : int {
        UNDEFINED = 0,
        ENCODER, // encoder for flashAttention
        DECODER  // decoder for flashAttention
    };
    CoderType coderType = UNDEFINED;
    bool isSupportAlibi = false;
    bool isFp32 = false; // high precision mode
    bool isClamp = false;
    float clampMin = 0;
    float clampMax = 0;
    enum MaskType : int {
        MASK_TYPE_UNDEFINED = 0,
        MASK_TYPE_NORM,
        MASK_TYPE_ALIBI
    };
    MaskType maskType = MASK_TYPE_UNDEFINED;
};


void from_json(const nlohmann::json &paramJson, FlashAttentionRopeLayerParam &param);

atb::Status FlashAttentionRopeLayer(const FlashAttentionRopeLayerParam &param, atb::Operation **operation);

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
} // namespace internlm_20b
} // namespace atb_speed
#endif