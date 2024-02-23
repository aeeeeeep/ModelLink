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
#ifndef VLMO_ENCODER_VL_LAYER_H
#define VLMO_ENCODER_VL_LAYER_H

#include <atb/atb_infer.h>
#include <atb/svector.h>

#include "atb_speed/base/hosttensor_binder.h"
#include "atb_speed/log.h"
#include "atb_speed/utils/str_split.h"

namespace atb_speed {
namespace vlmo {
struct EncoderVllayerParam {
    float layerNormEps = 0;
    int headNum = 0;
    int dk = 0;
    int rank = 0;
    int rankSize = 1;
    int maxTextLen = 40;
    std::string backend = "hccl";
    std::string model = "vlmo";
};



static void from_json(const nlohmann::json &paramJson, EncoderVllayerParam &param)
{
    paramJson.at("layerNormEps").get_to(param.layerNormEps);
    paramJson.at("headNum").get_to(param.headNum);
    paramJson.at("dk").get_to(param.dk);
    if (paramJson.contains("rank")) {
        paramJson.at("rank").get_to(param.rank);
    }
    if (paramJson.contains("rankSize")) {
        paramJson.at("rankSize").get_to(param.rankSize);
    }
    if (paramJson.contains("backend")) {
        paramJson.at("backend").get_to(param.backend);
    }
    if (paramJson.contains("maxTextLen")) {
        paramJson.at("maxTextLen").get_to(param.maxTextLen);
    }
}



atb::Status EncoderVlLayer(const EncoderVllayerParam &param, atb::Operation **operation);

static atb::Operation *CreateEncoderVlLayer(const nlohmann::json &paramJson)
{
    ATB_LOG(INFO) << GetFuncNameAndNameSpace(__PRETTY_FUNCTION__);
    atb::Operation *op;
    atb_speed::vlmo::EncoderVlLayer(paramJson.get<EncoderVllayerParam>(), &op);
    return op;
}


class EncoderVlLayerBinder : public HostTensorBinder {
public:
    EncoderVlLayerBinder();
    ~EncoderVlLayerBinder() override;
    void ParseParam(const nlohmann::json &paramJson) override;
    void BindTensor(atb::VariantPack &variantPack) override;

private:
    std::vector<int32_t> tokenOffset_;
    std::vector<int32_t> seqLen_;
};
} // namespace vlmo
} // namespace atb_speed
#endif