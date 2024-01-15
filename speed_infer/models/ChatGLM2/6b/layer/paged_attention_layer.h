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
#include "nlohmann/json.hpp"
#include "atb_speed/base/hosttensor_binder.h"

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

static atb::Operation *CreateDecoderPALayer(const nlohmann::json &paramJson)
{
    LayerParamPa param;
    if (paramJson.contains("rmsNormEps")) {
        param.rmsNormEps = paramJson["rmsNormEps"].get<double>();
    }
    if (paramJson.contains("residualAddScale")) {
        param.residualAddScale = paramJson["residualAddScale"].get<float>();
    }
    if (paramJson.contains("transKey")) {
        param.transKey = paramJson["transKey"].get<bool>();
    }
    if (paramJson.contains("layerId")) {
        param.layerId = paramJson["layerId"].get<int>();
    }
    if (paramJson.contains("preScale")) {
        param.preScale = paramJson["preScale"].get<float>();
    }
    if (paramJson.contains("postScale")) {
        param.postScale = paramJson["postScale"].get<float>();
    }
    if (paramJson.contains("numHeadsPerPartition")) {
        param.numHeadsPerPartition = paramJson["numHeadsPerPartition"].get<int>();
    }
    if (paramJson.contains("hiddenSizePerHead")) {
        param.hiddenSizePerHead = paramJson["hiddenSizePerHead"].get<int>();
    }
    if (paramJson.contains("numGroupsPerPartition")) {
        param.numGroupsPerPartition = paramJson["numGroupsPerPartition"].get<int>();
    }
    if (paramJson.contains("isPrefill")) {
        param.isPrefill = paramJson["isPrefill"].get<bool>();
    }
    ATB_LOG(INFO) << "ChatGLM2_6b_Decoder_Layer transKey:" << param.transKey << ", rmsNormEps:" << param.rmsNormEps
                  << ", layerId:" << param.layerId << ", preScale" << param.preScale << ", postScale" << param.postScale
                  << ", numHeadsPerPartion" << param.numHeadsPerPartition << ", hiddenSizePerHead"
                  << param.hiddenSizePerHead << ", numGroupsPerPartition" << param.numGroupsPerPartition
                  << ", residualAddScale" << param.residualAddScale
                  << ", isPrefill" << param.isPrefill;
    atb::Operation *op;
    DecoderPALayer(param, &op);
    return op;
}

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