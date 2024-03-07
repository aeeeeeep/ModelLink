/*
* Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
*
*  * Licensed under the Apache License, Version 2.0 (the "License")                                                                                                                                                                                                                                                                                                    ;
*  * you may not use this file except in compliance with the License.
*  * You may obtain a copy of the License at
*  *
*  * http://www.apache.org/licenses/LICENSE-2.0
*  *
*  * Unless required by applicable law or agreed to in writing, software
*  * distributed under the License is distributed on an "AS IS" BASIS,
*  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  * See the License for the specific language governing permissions and
*  * limitations under the License.
*  */
#ifndef ATB_SPEED_MODELS_MIXTRAL_DENSE_FUSION_LAYER_OPERATION_H
#define ATB_SPEED_MODELS_MIXTRAL_DENSE_FUSION_LAYER_OPERATION_H
#include <atb/atb_infer.h>
#include <atb/svector.h>
#include "atb_speed/base/hosttensor_binder.h"

namespace atb_speed {
namespace mixtralDense {
enum MixtralDenseFusionTensorId {
    IN_HIDDENSTATES = 0,
    IN_NORMWEIGHT,
    IN_QKVMIXDWEIGHT,
    IN_SELFOUTLINEARWEIGHT,
    IN_SELFOUTNORMWEIGHT,
    IN_BLOCK_SPARSE_MOE_GATE_WEIGHT,
    IN_MLPGATEUPWEIGHT_EXPERT_ONE,
    IN_MLPDOWNWEIGHT_EXPERT_ONE,
    IN_MLPGATEUPWEIGHT_EXPERT_TWO,
    IN_MLPDOWNWEIGHT_EXPERT_TWO,
    IN_MLPGATEUPWEIGHT_EXPERT_THREE,
    IN_MLPDOWNWEIGHT_EXPERT_THREE,
    IN_MLPGATEUPWEIGHT_EXPERT_FOUR,
    IN_MLPDOWNWEIGHT_EXPERT_FOUR,
    IN_MLPGATEUPWEIGHT_EXPERT_FIVE,
    IN_MLPDOWNWEIGHT_EXPERT_FIVE,
    IN_MLPGATEUPWEIGHT_EXPERT_SIX,
    IN_MLPDOWNWEIGHT_EXPERT_SIX,
    IN_MLPGATEUPWEIGHT_EXPERT_SEVEN,
    IN_MLPDOWNWEIGHT_EXPERT_SEVEN,
    IN_MLPGATEUPWEIGHT_EXPERT_EIGHT,
    IN_MLPDOWNWEIGHT_EXPERT_EIGHT,
    IN_COSTABLE,
    IN_SINTABLE,
    IN_FINAL_HIDDEN_STATE,
    IN_ONE_HOT_ONE,
    IN_ONE_HOT_ZERO,
    IN_ATTENTIONMASK,
    IN_CACHEK,
    IN_CACHEV,
    IN_TOKENOFFSET,
    IN_SEQLEN,
    IN_LAYERID,
    OUT_MIXTRAL_DENSE_LAYEROUT,
    INTERMIDATE_INPUTNORMOUT,
    INTERMIDATE_MIXEDQ,
    INTERMIDATE_MIXEDK,
    INTERMIDATE_MIXEDV,
    INTERMIDATE_POSITIONEMBEDQ,
    INTERMIDATE_POSITIONEMBEDK,
    INTERMIDATE_SELFOUT,
    INTERMIDATE_SELFLINEAROUT,
    INTERMIDATE_SELFRESIDUALADDOUT,
    INTERMIDATE_SELFNORMOUT,
    INTERMIDATE_MOEOUT,
    INTERMIDATE_MOELINEARPARALLELOUT,
};
struct MixtralDenseLayerFusionParam {
    float rmsNormEps = 0;
    int headNum = 0;
    int dk = 0;
    int rank = 0;
    int rankSize = 1;
    int rankRoot = 0;
    int coderType = 0;
    int isTriMask = 0;
    int kvHeadNum = 0;
    std::string backend = "lccl";
    std::string rankTableFile = "";
    HcclComm hcclComm = nullptr; // only effect when hcclComm is not null
    int layerId = 0;
    float qkScale = 1;
    int rotaryCoeff = 2;
    bool transpose = true;
    std::vector<int> tokenOffset;
    std::vector<int> seqLen;
};

atb::Status MixtralDenseLayerFusionOperation(const MixtralDenseLayerFusionParam &param,
                                             atb::Operation **operation);

class MixtralDenseLayerFusionBinder : public HostTensorBinder {
public:
    MixtralDenseLayerFusionBinder();
    virtual ~MixtralDenseLayerFusionBinder();
    void ParseParam(const nlohmann::json &paramJson) override;
    void BindTensor(atb::VariantPack &variantPack) override;

private:
    std::vector<int> tokenOffset_;
    std::vector<int> seqLen_;
    int32_t layerId_ = 0;
};
}
} // namespace atb_speed
#endif