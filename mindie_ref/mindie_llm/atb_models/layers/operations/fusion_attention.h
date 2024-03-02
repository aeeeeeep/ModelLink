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

#ifndef ATB_SPEED_MODELS_COMMON_ATTENTION_H
#define ATB_SPEED_MODELS_COMMON_ATTENTION_H

#include <vector>
#include <atb/atb_infer.h>
#include "atb_speed/log.h"
#include "layers/operations/linear.h"
#include "layers/operations/linear_parallel.h"
#include "layers/operations/norm_linear.h"

namespace atb_speed {
namespace common {

template <typename NormParamType>
struct FusionAttentionParam {
    // QKV linear param
    bool isPack = false;
    int isGroupedQueryAttention = false;
    bool isBF16 = false;
    bool isAnti = false;
    bool hasBias = false;
    int packQuantType = atb_speed::common::PackQuantType::ALL_FP;
    std::vector<int> layerLinearQuantType;
    NormParamType normParamType;
    NormParamType normQuantParamType;
    // rope param
    int rotaryCoeff = 2;
    // self attention param
    bool isFA = true;
    bool isPrefill = false;
    int headDim = 0;
    atb::infer::SelfAttentionParam selfAttentionParam;
    atb::infer::PagedAttentionParam pageAttentionParam;
    // self out linear param
    atb_speed::common::TensorParallelInfo selfOutLinearTensorParallelInfo;
};

template <typename NormParamType>
atb::Status Attention(const FusionAttentionParam<NormParamType> &param, atb::Operation **operation);
template <typename NormParamType>
atb::Status QKVLinearSplit(const FusionAttentionParam<NormParamType> &param, atb::Operation **operation);
template <typename NormParamType>
atb::Status SelfAttention(const FusionAttentionParam<NormParamType> &param, atb::Operation **operation);
} // namespace common
} // namespace atb_speed
#endif