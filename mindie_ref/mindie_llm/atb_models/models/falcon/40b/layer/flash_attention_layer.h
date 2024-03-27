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
#ifndef FALCON_40B_LAYER_FLASHATTENTION_OPERATION_H
#define FALCON_40B_LAYER_FLASHATTENTION_OPERATION_H

#include <atb/atb_infer.h>
#include <atb/svector.h>
#include "atb_speed/log.h"
#include "nlohmann/json.hpp"
#include "atb_speed/base/hosttensor_binder.h"

namespace atb_speed {
namespace falcon_40b {
struct LayerParallelFlashAttentionParam {
    int hiddenSize = 8192;         // hidden_size
    int headNum = 128;             // num_attention_heads
    int kvHeadNum = 8;             // num_kv_heads
    int headDim = 0;               // headDim = hidden_size/num_attention_heads
    int rank = 0;
    int rankSize = 1;
    int layerId = 0;
    float qScale = 1.0;
    float qkScale = 1.0;
    float layerNormEps = 0.00001;  // layer_norm_epsilon
    std::string model = "falcon_40b";
};

atb::Status LayerParallelFlashAttentionOperation(const LayerParallelFlashAttentionParam &param,
                                                 atb::Operation **operation);

class LayerPrallelFlashAttentionBinder : public HostTensorBinder {
public:
    LayerPrallelFlashAttentionBinder();
    virtual ~LayerPrallelFlashAttentionBinder();
    void ParseParam(const nlohmann::json &paramJson) override;
    void BindTensor(atb::VariantPack &variantPack) override;

private:
    std::vector<int32_t> tokenOffset_;
    std::vector<int32_t> seqLen_;
    int32_t layerId_ = 0;
};

} // namespace falcon_40b
} // namespace atb_speed
#endif