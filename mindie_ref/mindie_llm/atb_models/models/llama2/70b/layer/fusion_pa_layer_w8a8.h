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
#ifndef ATB_SPEED_MODELS_LLAMA2_70B_FUSION_PA_LAYER_W8A8_H
#define ATB_SPEED_MODELS_LLAMA2_70B_FUSION_PA_LAYER_W8A8_H

#include <vector>
#include "atb/atb_infer.h"
#include "atb_speed/base/hosttensor_binder.h"
#include "atb_speed/log.h"
#include "nlohmann/json.hpp"

namespace atb_speed {
namespace llama2_70b {
struct FusionPALayerW8A8Param {
    float rmsNormEps = 0;
    int headNum = 0;
    int dk = 0;
    int rank = 0;
    int rankSize = 1;
    std::string model = "llama2_70b";
    std::string backend = "hccl";
    int numHeadsPerPartition = 0;
    std::vector<int> seqLen;
    std::vector<int> tokenOffset;
    int rotaryCoeff = 2;
    bool transposedWeight = false;
    bool isPrefill = false;
};

enum FusionPALayerW8A8TensorId : int {
    IN_HIDDENSTATES = 0,
    IN_NORMWEIGHT,
    IN_QWEIGHT,
    IN_QSCALE,
    IN_QOFFSET,
    IN_QDESCALE,
    IN_KWEIGHT,
    IN_KSCALE,
    IN_KOFFSET,
    IN_KDESCALE,
    IN_VWEIGHT,
    IN_VSCALE,
    IN_VOFFSET,
    IN_VDESCALE,
    IN_OWEIGHT,
    IN_OSCALE,
    IN_OOFFSET,
    IN_ODESCALE,
    IN_SELFOUTNORMWEIGHT,
    IN_MLPGATEWEIGHT,
    IN_MLPGATESCALE,
    IN_MLPGATEOFFSET,
    IN_MLPGATEDESCALE,
    IN_MLPUPWEIGHT,
    IN_MLPUPSCALE,
    IN_MLPUPOFFSET,
    IN_MLPUPDESCALE,
    IN_MLPDOWNWEIGHT,
    IN_MLPDOWNSCALE,
    IN_MLPDOWNOFFSET,
    IN_MLPDOWNDESCALE,
    IN_COSTABLE,
    IN_SINTABLE,
    IN_ATTENTIONMASK,
    IN_K_CACHE,
    IN_V_CACHE,
    IN_BLOCK_TABLES,
    IN_SLOTS,
    IN_SEQLEN,

    OUT_LLAMA70BLAYEROUT,

    INTERMIDATE_INPUTNORMOUT,
    INTERMIDATE_Q,
    INTERMIDATE_K,
    INTERMIDATE_V,
    INTERMIDATE_POSITIONEMBEDQ,
    INTERMIDATE_POSITIONEMBEDK,
    INTERMIDATE_ATTENTIONOUT,
    INTERMIDATE_SELFLINEAROUT,
    INTERMIDATE_SELFRESIDUALADDOUT,
    INTERMIDATE_SELFNORMOUT,
    INTERMIDATE_MLPOUT,
};

atb::Status FusionPALayerW8A8(const FusionPALayerW8A8Param &param, atb::Operation **operation);

class FusionPALayerW8A8Binder : public HostTensorBinder {
public:
    FusionPALayerW8A8Binder();
    virtual ~FusionPALayerW8A8Binder();
    void ParseParam(const nlohmann::json &paramJson) override;
    void BindTensor(atb::VariantPack &variantPack) override;

private:
    std::vector<int> tokenOffset_;
    std::vector<int> seqLen_;
    int32_t layerId_ = 0;
};

} // namespace llama2_70b
} // namespace atb_speed
#endif