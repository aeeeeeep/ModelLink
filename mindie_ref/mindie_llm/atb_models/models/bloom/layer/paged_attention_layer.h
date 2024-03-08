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
#ifndef OPS_PAGED_BLOOM7BLAYER_OPERATION_H
#define OPS_PAGED_BLOOM7BLAYER_OPERATION_H
#include <atb/atb_infer.h>
#include "atb_speed/base/hosttensor_binder.h"

namespace atb_speed {
namespace bloom_7b {
struct Bloom7bPagedLayerParam {
    double layerNormEps = 0;
    int headNum = 0;
    int dk = 0;
    float invNormFactorvarAttr = 0;
    std::string model = "bloom_7b";
    std::string backend = "lccl";
    int rank = 0;
    int rankSize = 1;
    bool quantmodel = false;
    bool isPrefill = false;
    int quantMode = -1;   // 0:not quant, 1:w8a8, 2:w8a16
    float qkvInputScale = 1;
    int qkvInputOffset = 0;
    float denseInputScale = 1;
    int denseInputOffset = 0;
    float selfLnInputScale = 1;
    int selfLnInputOffset = 0;
    float ffnOutInputScale = 1;
    int ffnOutInputOffset = 0;
};

atb::Status PagedLayer(const Bloom7bPagedLayerParam &param, atb::Operation **operation);

void SqueezeThirdDim(const atb::Dims &oldShape, atb::Dims &newShape);

void MulFirstSecondDim(const atb::Dims &oldShape, atb::Dims &newShape);

} // namespace bloom_7b
} // namespace atb_speed
#endif