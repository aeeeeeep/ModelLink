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
#ifndef ATB_SPEED_MODELS_CHATGLM2_6B_LAYER_PAGE_ATTENTION_H
#define ATB_SPEED_MODELS_CHATGLM2_6B_LAYER_PAGE_ATTENTION_H
#include "atb/atb_infer.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtype-limits"
#include "nlohmann/json.hpp"
#pragma GCC diagnostic pop
#include "atb_speed/base/hosttensor_binder.h"
#include "atb_speed/log.h"
#include "atb_speed/utils/operation_util.h"

namespace atb_speed {
namespace chatglm2_6b {
struct LayerParamPa {
    bool isPrefill = true;
    double rmsNormEps = 0;
    int headNum = 0;
    int dk = 0;
    int numHeadsPerPartition = 0;
    int hiddenSizePerHead = 1;
    int numGroupsPerPartition = 1;
    bool transKey = false;
    int layerId = 0;
    float preScale = 0;
    float postScale = 0;
    float residualAddScale = 0;
    int rank = 0;
    int rankSize = 1;
};

atb::Status DecoderLayer(const LayerParamPa &param, atb::Operation **operation);

atb::Status DecoderPALayer(const LayerParamPa &param, atb::Operation **operation);

class FlashAttentionHostBinder : public HostTensorBinder {
public:
    FlashAttentionHostBinder();

    virtual ~FlashAttentionHostBinder();

    void ParseParam(const nlohmann::json &paramJson) override;

    void BindTensor(atb::VariantPack &variantPack) override;

private:
    std::vector<int64_t> seqLen_;
};
} // namespace chatglm2_6b
} // namespace atb_speed
#endif