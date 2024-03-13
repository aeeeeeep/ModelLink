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

#ifndef ATB_SPEED_MODELS_TELECHAT_QUANT_LAYER_H
#define ATB_SPEED_MODELS_TELECHAT_QUANT_LAYER_H

#include "atb/atb_infer.h"
#include "atb_speed/log.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtype-limits"
#include "nlohmann/json.hpp"
#pragma GCC diagnostic pop

namespace atb_speed {
namespace telechat {
struct QuantFALayerParam {
    double rmsNormEps = 0;
    int headNum = 0;
    int dk = 0;
    bool isFloatQueryLayer = false;
    bool isFloatKVLayer = false;
    bool isFloatDownLayer = false;
    float inputScale_qkv = 1;
    int inputOffset_qkv = 0;
    float inputScale_dense = 1;
    int inputOffset_dense = 0;
    float inputScale_gate_up = 1;
    int inputOffset_gate_up = 0;
    float inputScale_down_proj = 1;
    int inputOffset_down_proj = 0;
    int rank = 0;
    int rankSize = 1;
};

atb::Status QuantFALayer(const QuantFALayerParam &param, atb::Operation **operation);

}  // namespace telechat
}  // namespace atb_speed

#endif