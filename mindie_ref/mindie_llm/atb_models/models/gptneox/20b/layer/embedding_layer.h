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
#ifndef ATB_SPEED_MODELS_GPTNEOX_20B_LAYER_H
#define ATB_SPEED_MODELS_GPTNEOX_20B_LAYER_H

#include <nlohmann/json.hpp>
#include <atb/atb_infer.h>
#include "atb_speed/log.h"
#include "atb_speed/base/hosttensor_binder.h"

namespace atb_speed {
namespace gptneox_20b {
struct EmbeddingLayerParam {
    int axis = 0;
};

atb::Status EmbeddingLayer(const EmbeddingLayerParam &param, atb::Operation **operation);

atb::Operation *CreateEmbeddingLayer(const nlohmann::json &paramJson);

class EmbeddingHostBinder : public HostTensorBinder {
public:
    EmbeddingHostBinder();

    virtual ~EmbeddingHostBinder();

    void ParseParam(const nlohmann::json &paramJson) override;

    void BindTensor(atb::VariantPack &variantPack) override;

private:
    atb::SVector<int64_t> tokenOffset_;
    atb::SVector<int64_t> seqLen_;
};
} // namespace gptneox_20b
} // namespace atb_speed
#endif