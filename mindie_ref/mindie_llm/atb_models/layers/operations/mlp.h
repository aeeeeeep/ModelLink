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
#ifndef ATB_SPEED_MODELS_COMMON_MLP_OPERATION_H
#define ATB_SPEED_MODELS_COMMON_MLP_OPERATION_H
#include <atb/atb_infer.h>
#include "layers/operations/linear.h"
#include "layers/operations/linear_parallel.h"
#include "layers/operations/norm_linear.h"

namespace atb_speed {
namespace common {

enum MlpPackType : unsigned int {
    GATE_UP_WEIGHT_PACK = 0,
    GATE_UP_WEIGHT_NO_PACK = 1,
    UP_WEIGHT_ONLY = 2,
};

template <typename NormParamType>
struct MlpParam {
    bool isBF16 = false;
    bool gateUpHasBias = false;
    bool downHasBias = false;
    bool supportLcoc = false;
    bool skipNorm = false;
    bool normHasBias = false;
    bool useFusionNorm
    MlpPackType mlpPackType = GATE_UP_WEIGHT_PACK;
    std::vector<int> layerLinearQuantType;
    int packQuantType = atb_speed::common::PackQuantType::ALL_FP;
    NormParamType normParamType;
    NormParamType normQuantParamType;
    atb::infer::ActivationParam activationParam;
    atb_speed::common::TensorParallelInfo downLinearTensorParallelInfo;
};

template <typename NormParamType>
atb::Status Mlp(const MlpParam<NormParamType> &param, atb::Operation **operation);

} // namespace common
} // namespace atb_speed
#endif