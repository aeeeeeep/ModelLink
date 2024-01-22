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
#ifndef ATB_SPEED_MODELS_STAR_CODER_PARALLEL_FA_QUANT_LAYER_H
#define ATB_SPEED_MODELS_STAR_CODER_PARALLEL_FA_QUANT_LAYER_H

#include <nlohmann/json.hpp>
#include <atb/atb_infer.h>
#include "atb_speed/log.h"
#include "atb_speed/base/hosttensor_binder.h"

namespace atb_speed {
namespace star_coder {
struct FlashAttentionQuantLayerParam {
    std::string model = "star_coder";
    float layerNormEps = 0;
    int headNum = 0;
    int dk = 0;
    float qScale = 1.0;
    int rank = 0;
    int rankSize = 1;
    int kvHead = 1;
    bool isEncoder = false;
    // 量化参数
    bool quantmodel = false;
    float qkvInputScale = 1;
    int qkvInputOffset = 0;
    float denseInputScale = 1;
    int denseInputOffset = 0;
    float selfLnInputScale = 1;
    int selfLnInputOffset = 0;
    float mlpOutInputScale = 1;
    int mlpOutInputOffset = 0;
    bool isSparse = false;
};

atb::Status FlashAttentionQuantLayer(const FlashAttentionQuantLayerParam &param, atb::Operation **operation);

static atb::Operation *CreateFlashAttentionQuantLayer(const nlohmann::json &paramJson)
{
    FlashAttentionQuantLayerParam param;
    param.layerNormEps = paramJson["layerNormEps"].get<float>();
    param.headNum = paramJson["headNum"].get<int>();
    param.dk = paramJson["dk"].get<int>();
    param.model = paramJson["model"].get<std::string>();
    param.isEncoder = paramJson["isEncoder"].get<bool>();
    if (paramJson.contains("qScale")) {
        param.qScale = paramJson["qScale"].get<float>();
    }
    if (paramJson.contains("rank")) {
        param.rank = paramJson["rank"].get<int>();
    }
    if (paramJson.contains("rankSize")) {
        param.rankSize = paramJson["rankSize"].get<int>();
    }
    // 量化参数
    param.qkvInputScale = paramJson["qkvInputScale"].get<float>();
    param.qkvInputOffset = paramJson["qkvInputOffset"].get<int>();
    param.denseInputScale = paramJson["denseInputScale"].get<float>();
    param.denseInputOffset = paramJson["denseInputOffset"].get<int>();
    param.selfLnInputScale = paramJson["selfLnInputScale"].get<float>();
    param.selfLnInputOffset = paramJson["selfLnInputOffset"].get<float>();
    param.mlpOutInputScale = paramJson["mlpOutInputScale"].get<float>();
    param.mlpOutInputOffset = paramJson["mlpOutInputOffset"].get<int>();

    ATB_LOG(INFO) << __func__ << " layerNormEps:" << param.layerNormEps << ", headNum:" << param.headNum << ", dk:" <<
        param.dk << ", model:" << param.model;
    atb::Operation *op;
    FlashAttentionQuantLayer(param, &op);
    return op;
}

class FlashAttentionQuantHostBinder : public HostTensorBinder {
public:
    FlashAttentionQuantHostBinder();

    virtual ~FlashAttentionQuantHostBinder();

    void ParseParam(const nlohmann::json &paramJson) override;

    void BindTensor(atb::VariantPack &variantPack) override;

private:
    std::vector<int32_t> tokenOffset_;
    std::vector<int32_t> seqLen_;
};
} // namespace star_coder
} // namespace atb_speed
#endif