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
#ifndef AQUILA_7B_PAGED_ATTENTION_LAYER_H
#define AQUILA_7B_PAGED_ATTENTION_LAYER_H

#include <atb/atb_infer.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtype-limits"
#include <nlohmann/json.hpp>
#pragma GCC diagnostic pop

#include "atb_speed/base/hosttensor_binder.h"
#include "atb_speed/log.h"
#include "atb_speed/utils/str_split.h"

namespace atb_speed {
namespace aquila_7b {
struct PagedAttentionRopeLayerParam {
    float rmsNormEps = 0;
    int headNum = 0;
    int dk = 0;
    int rank = 0;
    int rankSize = 1;
    bool isPrefill = false;
    bool transposedWeight = true;
    std::string backend = "hccl";
    std::string model = "aquila_7b";
};

void from_json(const nlohmann::json &paramJson, PagedAttentionRopeLayerParam &param);

void reshapeHeads(const atb::Dims &oldShape, atb::Dims &newShape, int headNum);

atb::Operation *CreatePagedAttentionRopeLayer(const nlohmann::json &paramJson);

atb::Status PagedAttentionRopeLayer(const PagedAttentionRopeLayerParam &param, atb::Operation **operation);

class PagedAttentionRopeLayerBinder : public HostTensorBinder {
public:
    PagedAttentionRopeLayerBinder();

    ~PagedAttentionRopeLayerBinder() override;

    void ParseParam(const nlohmann::json &paramJson) override;

    void BindTensor(atb::VariantPack &variantPack) override;

private:
    std::vector<int32_t> seqLen_;
};
} // namespace aquila_7b
} // namespace atb_speed
#endif