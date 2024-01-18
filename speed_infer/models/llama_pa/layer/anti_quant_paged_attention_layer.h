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
#ifndef ATB_SPEED_MODELS_LLAMA_QUANT_PA_LAYER_H
#define ATB_SPEED_MODELS_LLAMA_QUANT_PA_LAYER_H

#include <atb/atb_infer.h>
#include <nlohmann/json.hpp>
#include "atb_speed/log.h"
#include "atb_speed/base/hosttensor_binder.h"
#include "atb_speed/utils/str_split.h"

namespace atb_speed {
namespace llama_pa {
struct QuantPALayerParam {
    int rank = 0;
    int rankSize = 1;
    float rmsNormEps = 0;
    int headNum = 0;
    int dk = 0;
    bool transposedWeight = false;
    bool isPrefill = false;
    std::string backend = "hccl";
    std::string model = "llama";
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

static void from_json(const nlohmann::json &paramJson, QuantPALayerParam &param)
{
    paramJson.at("rmsNormEps").get_to(param.rmsNormEps);
    paramJson.at("headNum").get_to(param.headNum);
    paramJson.at("dk").get_to(param.dk);
    if (paramJson.contains("rank")) {
        paramJson.at("rank").get_to(param.rank);
    }
    if (paramJson.contains("rankSize")) {
        paramJson.at("rankSize").get_to(param.rankSize);
    }
    if (paramJson.contains("transposedWeight")) {
        paramJson.at("transposedWeight").get_to(param.transposedWeight);
    }
    if (paramJson.contains("isPrefill")) {
        paramJson.at("isPrefill").get_to(param.isPrefill);
    }
    if (paramJson.contains("backend")) {
        paramJson.at("backend").get_to(param.backend);
    }
    // 量化参数
    param.qkvInputScale = paramJson["qkvInputScale"].get<float>();
    param.qkvInputOffset = paramJson["qkvInputOffset"].get<int>();
    param.denseInputScale = paramJson["denseInputScale"].get<float>();
    param.denseInputOffset = paramJson["denseInputOffset"].get<int>();
    param.selfLnInputScale = paramJson["selfLnInputScale"].get<float>();
    param.selfLnInputOffset = paramJson["selfLnInputOffset"].get<int>();
    param.ffnOutInputScale = paramJson["ffnOutInputScale"].get<float>();
    param.ffnOutInputOffset = paramJson["ffnOutInputOffset"].get<int>();
}

atb::Status QuantPALayer(const QuantPALayerParam &param, atb::Operation **operation);

static atb::Operation *CreateQuantPALayer(const nlohmann::json &paramJson)
{
    ATB_LOG(INFO) << GetFuncNameAndNameSpace(__PRETTY_FUNCTION__);
    atb::Operation *op;
    QuantPALayer(paramJson.get<QuantPALayerParam>(), &op);
    return op;
}

class QuantFlashAttentionHostBinder : public HostTensorBinder {
public:
    QuantFlashAttentionHostBinder();

    virtual ~QuantFlashAttentionHostBinder();

    void ParseParam(const nlohmann::json &paramJson) override;

    void BindTensor(atb::VariantPack &variantPack) override;

private:
    atb::SVector<int32_t> seqLen_;
};

} // namespace llama_pa
} // namespace atb_speed
#endif