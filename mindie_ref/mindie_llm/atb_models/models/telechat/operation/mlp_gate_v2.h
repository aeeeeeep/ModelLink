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

#ifndef ATB_SPEED_TELECHAT_MLP_GATE_V2_H
#define ATB_SPEED_TELECHAT_MLP_GATE_V2_H

#include "atb_speed/log.h"
#include "nlohmann/json.hpp"
#include "parallel_layer_v2.h"
#include <atb/atb_infer.h>

namespace atb_speed {
namespace telechat {
struct MlpGateParamV2 {
    atb::infer::ActivationType activationType;
    bool transposeB = false;
    bool isBias = false;
    bool isPack = false;
    bool isUpQuant = false;
    bool isGateQuant = false;
    bool isDownQuant = false;
    bool isSparse = false;
    bool noGate = false;
    CommParam commDownParam;
    QuantParam quantUpParam;
    QuantParam quantGateParam;
    QuantParam quantDownParam;
};

atb::Status MlpGateLayerV2(const MlpGateParamV2 &param, atb::Operation **operation);

} // namespace telechat
} // namespace atb_speed
#endif
