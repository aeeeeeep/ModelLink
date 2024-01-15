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
#ifndef OPS_CHATGML2_6B_COMMON_FA_OPERATION_H
#define OPS_CHATGML2_6B_COMMON_FA_OPERATION_H
#include "atb/atb_infer.h"
#include "atb_speed/log.h"
#include "nlohmann/json.hpp"
#include "atb_speed/base/hosttensor_binder.h"

namespace atb_speed {
namespace chatglm2_6b {

struct CommonLayerParamFa {
    int numHeadsPerPartition = 0;
    int numGroupsPerPartition = 1;
    int hiddenSizePerHead = 1;
    int layerId = 0;
    float rmsNormEps = 0;
    float residualAddScale = 0;
    float preScale = 0;
    float postScale = 0;
    bool transKey = false;
    bool quantmodel = false;
    bool isSparse = false;
    std::string model = "chatglm2_6b_parallel";
    float qkvInputScale = 1;
    int qkvInputOffset = 0;
    float denseInputScale = 1;
    int denseInputOffset = 0;
    float selfLnInputScale = 1;
    int selfLnInputOffset = 0;
    float ffnOutInputScale = 1;
    int ffnOutInputOffset = 0;
    int rank = 0;
    int rankSize = 1;
    std::string backend = "hccl";
    bool isEncoder = false;
};

atb::Status CommonLayerFa(const CommonLayerParamFa &param, atb::Operation **operation);

class CommonLayerFaBinder : public HostTensorBinder {
public:
    CommonLayerFaBinder();

    virtual ~CommonLayerFaBinder();

    void ParseParam(const nlohmann::json &paramJson) override;

    void BindTensor(atb::VariantPack &variantPack) override;

private:
    std::vector<int32_t> tokenOffset_;
    std::vector<int32_t> seqLen_;
};

static atb::Operation *CreateCommonLayerFa(const nlohmann::json &paramJson)
{
    CommonLayerParamFa param;
    if (paramJson.contains("qkvInputScale")) {
        param.qkvInputScale = paramJson["qkvInputScale"].get<float>();
    }
    if (paramJson.contains("qkvInputOffset")) {
        param.qkvInputOffset = paramJson["qkvInputOffset"].get<int>();
    }
    if (paramJson.contains("numHeadsPerPartition")) {
        param.numHeadsPerPartition = paramJson["numHeadsPerPartition"].get<int>();
    }
    if (paramJson.contains("hiddenSizePerHead")) {
        param.hiddenSizePerHead = paramJson["hiddenSizePerHead"].get<int>();
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
    if (paramJson.contains("denseInputScale")) {
        param.denseInputScale = paramJson["denseInputScale"].get<float>();
    }
    if (paramJson.contains("denseInputOffset")) {
        param.denseInputOffset = paramJson["denseInputOffset"].get<int>();
    }
    if (paramJson.contains("rank")) {
        param.rank = paramJson["rank"].get<int>();
    }
    if (paramJson.contains("rankSize")) {
        param.rankSize = paramJson["rankSize"].get<int>();
    }
    if (paramJson.contains("backend")) {
        param.backend = paramJson["backend"].get<std::string>();
    }
    if (paramJson.contains("selfLnInputScale")) {
        param.selfLnInputScale = paramJson["selfLnInputScale"].get<float>();
    }
    if (paramJson.contains("selfLnInputOffset")) {
        param.selfLnInputOffset = paramJson["selfLnInputOffset"].get<int>();
    }
    if (paramJson.contains("ffnOutInputScale")) {
        param.ffnOutInputScale = paramJson["ffnOutInputScale"].get<float>();
    }
    if (paramJson.contains("ffnOutInputOffset")) {
        param.ffnOutInputOffset = paramJson["ffnOutInputOffset"].get<int>();
    }
    if (paramJson.contains("isEncoder")) {
        param.isEncoder = paramJson["isEncoder"].get<bool>();
    }
    ATB_LOG(INFO) << "ChatGLM2_6b_SelfAttentionKvCacheParam preScale" << param.preScale << ", postScale"
                  << param.postScale << ", numHeadsPerPartion" << param.numHeadsPerPartition << ", hiddenSizePerHead"
                  << param.hiddenSizePerHead << ", numGroupsPerPartition" << param.numGroupsPerPartition
                  << ", isEncoder" << param.isEncoder;
    atb::Operation *op;
    CommonLayerFa(param, &op);
    return op;
}
}  // namespace chatglm2_6b
}  // namespace atb_speed
#endif
