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

#ifndef ASCEND_SPEED_INFERENCE_COMMON_NORM_LINEAR_H
#define ASCEND_SPEED_INFERENCE_COMMON_NORM_LINEAR_H

#include <atb/atb_infer.h>
#include "layers/operations/add_norm.h"
#include "layers/operations/linear.h"

namespace atb_speed {
namespace common {

enum PackQuantType : unsigned int {
    ALL_FP = 1,
    ALL_W8A8 = 2,
    ALL_W8A8_ANTI = 3,
    MIX_W8A8 = 4,
    MIX_W8A8_ANTI = 5,
    ALL_W8A16 = 6,
    ALL_W8A8SC = 7,
    MIX_W8A8SC = 8,
};

enum NextResidualAddInType : unsigned int {
    NORM_OUT = 0,
    ADD_OUT,
};

template <typename NormParamType>
struct NormLinearParam {
    NextResidualAddInType nextResidualAddIn = ADD_OUT;
    atb_speed::common::AddNormParam<NormParamType> addNormParam;
    atb_speed::common::FusionLinearParam fusionLinearParam;
};

NormQuantType GetNormQuantType(const int &packQuantType);
LinearQuantType GetLinearQuantType(const int &packQuantType, const int &linearType, bool hasNorm);
template <typename NormParamType>
atb::Status NormLinear(const NormLinearParam<NormParamType> &param, atb::Operation **operation);

} // namespace common
} // namespace atb_speed

#endif
