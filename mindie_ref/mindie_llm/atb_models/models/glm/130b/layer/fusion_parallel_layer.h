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
#ifndef ATB_SPEED_MODELS_GLM130B_FUSION_PARALLEL_LAYER_H
#define ATB_SPEED_MODELS_GLM130B_FUSION_PARALLEL_LAYER_H
#include <atb/atb_infer.h>
#include <atb/svector.h>
#include "atb_speed/log.h"
#include "atb_speed/base/hosttensor_binder.h"

namespace atb_speed {
namespace glm130b {
enum FusionParallelLayerTensorId : int {
    IN_HIDDENSTATES_ID = 0,
    IN_NORMWEIGHT_ID,
    IN_NORMBIAS_ID,
    IN_QKVMIXEDWEIGHT_ID,
    IN_QKVMIXEDBIAS_ID,
    IN_SELFOUTLINEARWEIGHT_ID,
    IN_SELFOUTLINEARBIAS_ID,
    IN_SELFOUTNORMWEIGHT_ID,
    IN_SELFOUTNORMBIAS_ID,
    IN_MLPLINEARWEIGHT_ID,
    IN_MLPLINEARBIAS_ID,
    IN_MLPOUTLINEARWEIGHT_ID,
    IN_MLPOUTLINEARBIAS_ID,
    IN_COS_ID,
    IN_SIN_ID,
    IN_ATTENTIONMASK_ID,
    IN_CACHEK_ID,
    IN_CACHEV_ID,
    IN_SEQLEN_ID,
    IN_TOKENOFFSET_ID,
    IN_LAYERID_ID,
    OUT_LAYEROUT_ID,
    INTERMEDIATE_INPUTNORMOUT_ID,
    INTERMEDIATE_MIXEDLINEAROUTQKV_ID,
    INTERMEDIATE_POSITIONEMBEDQ_ID,
    INTERMEDIATE_POSITIONEMBEDK_ID,
    INTERMEDIATE_VALUE_ID,
    INTERMEDIATE_SELFOUT_ID,
    INTERMEDIATE_SELFLINEAROUT_ID,
    INTERMEDIATE_SELFRESIDUALOUT_ID,
    INTERMEDIATE_SELFADDOUT_ID,
    INTERMEDIATE_SELFNORMOUT_ID,
    INTERMEDIATE_MLPOUT_ID,
    INTERMEDIATE_MLPRESIDUALOUT_ID,
};

struct FusionParallelLayerParam {
    int headNum = 0;
    int headDim = 0;
    int rank = 0;
    int rankSize = 1;
    int coderType = 0;
    int rankRoot = 0;
    float qScale = 1;
    float qkScale = 1;
    float residualAddScale = 0;
    double layerNormEps = 0;
    std::string backend = "hccl";
};

atb::Status CreateFusionParallelLayer(const FusionParallelLayerParam &param, atb::Operation **operation);

class FusionParallelLayer : public HostTensorBinder {
public:
    FusionParallelLayer();
    virtual ~FusionParallelLayer();
    void ParseParam(const nlohmann::json &paramJson) override;
    void BindTensor(atb::VariantPack &variantPack) override;

private:
    atb::SVector<int32_t> tokenOffset_;
    atb::SVector<int32_t> seqLen_;
};
} // namespace glm130b
} // namespace atb_speed
#endif