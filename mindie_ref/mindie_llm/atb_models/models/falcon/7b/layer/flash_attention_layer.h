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
#ifndef FALCON_7BLAYER_FUSION_OPERATION_H
#define FALCON_7BLAYER_FUSION_OPERATION_H

#include <atb/atb_infer.h>
#include <atb/svector.h>
#include "atb_speed/log.h"
#include "atb_speed/base/hosttensor_binder.h"

namespace atb_speed {
namespace falcon_7b {
struct LayerFusionParam {
    float layerNormEps = 0.0f;
    int headNum = 0;
    int kvHeadNum = 0;
    int hiddenSize = 0;
    int layerId = 0;
    float preScale = 1.0f;
    float postScale = 1.0f;
    std::string model = "falcon7b";
    int rotaryCoeff = 2;
};
atb::Status FusionLayerOperation(const LayerFusionParam &param, atb::Operation **operation);

class LayerFusionBinder : public HostTensorBinder {
public:
    LayerFusionBinder();
    virtual ~LayerFusionBinder();
    void ParseParam(const nlohmann::json &paramJson) override;
    void BindTensor(atb::VariantPack &variantPack) override;

private:
    atb::SVector<int32_t> tokenOffset_;
    atb::SVector<int32_t> seqLen_;
    int32_t layerId_ = 0;
};
} // namespace falcon_7b
} // namespace atb_speed
#endif