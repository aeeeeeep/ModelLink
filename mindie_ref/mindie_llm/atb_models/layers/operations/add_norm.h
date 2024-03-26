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

#ifndef ASCEND_SPEED_INFERENCE_COMMON_ADD_NORM_H
#define ASCEND_SPEED_INFERENCE_COMMON_ADD_NORM_H

#include <atb/atb_infer.h>
#include "layers/operations/empty_operation.h"

namespace atb_speed {
namespace common {

enum AddNormType : unsigned int {
    FUSION_ADD_NORM = 0,
    ADD_NORM,
    NORM_ONLY,
};

enum NormQuantType : unsigned int {
    NORM_NO_QUANT = 0,
    NORM_QUANT,
    NORM_ANTI_OUTLIER_QUANT,
};

template <typename NormParamType>
struct AddNormParam {
    bool normHasBias = false;
    AddNormType addNormType = ADD_NORM;
    NormQuantType normQuantType = NORM_NO_QUANT;
    NormParamType normParamType;
    NormParamType normQuantParamType;
};

template <typename NormParamType>
atb::Status AddNorm(const AddNormParam<NormParamType> &param, atb::Operation **operation);

} // namespace common
} // namespace atb_speed

#endif
