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
}  // namespace chatglm2_6b
}  // namespace atb_speed
#endif
