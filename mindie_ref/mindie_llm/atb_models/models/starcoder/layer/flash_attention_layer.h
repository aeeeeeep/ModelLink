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
#ifndef ATB_SPEED_MODELS_STAR_CODER_PARALLEL_FA_LAYER_H
#define ATB_SPEED_MODELS_STAR_CODER_PARALLEL_FA_LAYER_H

#include <nlohmann/json.hpp>
#include <atb/atb_infer.h>
#include "atb_speed/log.h"
#include "atb_speed/base/hosttensor_binder.h"

namespace atb_speed {
namespace star_coder {
struct FlashAttentionLayerParam {
    float layerNormEps = 0;
    int headNum = 0;
    int dk = 0;
    float qScale = 1.0;
    std::string model = "star_coder";
    int rank = 0;
    int rankSize = 1;
    int kvHead = 1;
    bool isEncoder = false;
};

atb::Status FlashAttentionLayer(const FlashAttentionLayerParam &param, atb::Operation **operation);

class FlashAttentionHostBinder : public HostTensorBinder {
public:
    FlashAttentionHostBinder();

    virtual ~FlashAttentionHostBinder();

    void ParseParam(const nlohmann::json &paramJson) override;

    void BindTensor(atb::VariantPack &variantPack) override;

private:
    std::vector<int32_t> tokenOffset_;
    std::vector<int32_t> seqLen_;
};
} // namespace star_coder
} // namespace atb_speed
#endif