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
#ifndef ATB_SPEED_MODELS_LLAMA_FLASHATTENTION_LAYER_OPERATION_H
#define ATB_SPEED_MODELS_LLAMA_FLASHATTENTION_LAYER_OPERATION_H

#include <atb/atb_infer.h>
#include <atb/svector.h>
#include "nlohmann/json.hpp"
#include "atb_speed/base/hosttensor_binder.h"

namespace atb_speed {
namespace llama {
struct FlashAttentionLayerParam {
    float rmsNormEps = 0;
    int headNum = 0;
    int dk = 0; // headDim
    int rank = 0;
    int rankSize = 1;
    std::string model = "llama_13b";
    float qScale = 1.0;
    bool quantModel = false;
    bool sparseModel = false;
    bool isEncoder = false;
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

atb::Status FlashAttentionLayer(const FlashAttentionLayerParam &param, atb::Operation **operation);

static atb::Operation *CreateFlashAttentionLayer(const nlohmann::json &paramJson)
{
    FlashAttentionLayerParam param;
    param.rmsNormEps = paramJson["rmsNormEps"].get<float>();
    param.headNum = paramJson["headNum"].get<int>();
    param.dk = paramJson["dk"].get<int>();
    param.rank = paramJson["rank"].get<int>();
    param.rankSize = paramJson["rankSize"].get<int>();
    param.model = paramJson["model"].get<std::string>();
    param.quantModel = paramJson["quantModel"].get<bool>();
    param.sparseModel = paramJson["sparseModel"].get<bool>();
    param.isEncoder = paramJson["isEncoder"].get<bool>();
    // 量化参数
    param.qkvInputScale = paramJson["qkvInputScale"].get<float>();
    param.qkvInputOffset = paramJson["qkvInputOffset"].get<int>();
    param.denseInputScale = paramJson["denseInputScale"].get<float>();
    param.denseInputOffset = paramJson["denseInputOffset"].get<int>();
    param.selfLnInputScale = paramJson["selfLnInputScale"].get<float>();
    param.selfLnInputOffset = paramJson["selfLnInputOffset"].get<int>();
    param.ffnOutInputScale = paramJson["ffnOutInputScale"].get<float>();
    param.ffnOutInputOffset = paramJson["ffnOutInputOffset"].get<int>();

    ATB_LOG(INFO) << "LLaMA FlashAttentionLayer headNum:" << param.headNum << ", rmsNormEps:" <<
        param.rmsNormEps << ", dk:" << param.dk << ", model:" << param.model << ", rank:" << param.rank <<
        ", rankSize:" << param.rankSize;
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
    std::vector<int32_t> tokenOffset_;
    std::vector<int32_t> seqLen_;
};
} // namespace llama
} // namespace atb_speed
#endif