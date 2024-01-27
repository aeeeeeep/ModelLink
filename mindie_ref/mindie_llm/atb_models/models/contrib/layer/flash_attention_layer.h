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
#ifndef CONTRIB_FLASH_ATTENTION_LAYER_H
#define CONTRIB_FLASH_ATTENTION_LAYER_H

#include <vector>
#include "atb/atb_infer.h"
#include "atb_speed/base/hosttensor_binder.h"
#include "atb_speed/log.h"
#include "nlohmann/json.hpp"

namespace atb_speed {
namespace contrib {
struct FlashAttentionLayerParam {
    float rmsNormEps = 0;
    int headNum = 0;
    int dk = 0;
    int rank = 0;
    int rankSize = 1;
    std::string model = "";
    std::string backend = "hccl";
    int kvHeadNum = 0;
    std::vector<int> seqLen;
    std::vector<int> tokenOffset;
    int rotaryCoeff = 2;
    int layerId = 1;
};

atb::Status FlashAttentionLayer(const FlashAttentionLayerParam &param, atb::Operation **operation);

static atb::Operation *CreateFlashAttentionLayer(const nlohmann::json &paramJson)
{
    FlashAttentionLayerParam param;
    if (paramJson.find("rmsNormEps") != paramJson.end()) {
        param.rmsNormEps = paramJson["rmsNormEps"].get<float>();
    }
    if (paramJson.find("headNum") != paramJson.end()) {
        param.headNum = paramJson["headNum"].get<int>();
    }
    if (paramJson.find("dk") != paramJson.end()) {
        param.dk = paramJson["dk"].get<int>();
    }
    if (paramJson.find("rank") != paramJson.end()) {
        param.rank = paramJson["rank"].get<int>();
    }
    if (paramJson.find("rankSize") != paramJson.end()) {
        param.rankSize = paramJson["rankSize"].get<int>();
    }
    if (paramJson.find("backend") != paramJson.end()) {
        param.backend = paramJson["backend"].get<std::string>();
    }
    if (paramJson.find("model") != paramJson.end()) {
        param.model = paramJson["model"].get<std::string>();
    }
    if (paramJson.contains("layerId")) {
        param.layerId = paramJson["layerId"].get<int>();
    }
    if (paramJson.contains("kvHeadNum")) {
        param.kvHeadNum = paramJson["kvHeadNum"].get<int>();
    }
    if (paramJson.contains("rotaryCoeff")) {
        param.rotaryCoeff = paramJson["rotaryCoeff"].get<int>();
    }

    ATB_LOG(INFO) << "FusionLayerParam params headNum:" << param.headNum << ", rmsNormEps:" << param.rmsNormEps
                  << ", dk:" << param.dk << ", model:" << param.model << ", rank:" << param.rank
                  << ", rankSize:" << param.rankSize << ", backend:" << param.backend
                  << ", kvHeadNum:" << param.kvHeadNum;
    atb::Operation *op;
    FlashAttentionLayer(param, &op);
    return op;
}

class FlashAttentionLayerBinder : public HostTensorBinder {
public:
    FlashAttentionLayerBinder();
    virtual ~FlashAttentionLayerBinder();
    void ParseParam(const nlohmann::json &paramJson) override;
    void BindTensor(atb::VariantPack &variantPack) override;

private:
    std::vector<int> tokenOffset_;
    std::vector<int> seqLen_;
    int32_t layerId_ = 0;
};

} // namespace contrib
} // namespace atb_speed
#endif