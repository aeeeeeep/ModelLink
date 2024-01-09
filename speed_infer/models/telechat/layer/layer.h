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

#ifndef ATB_SPEED_MODELS_TELECHAT_QUANT_FA_LAYER_H
#define ATB_SPEED_MODELS_TELECHAT_QUANT_FA_LAYER_H

#include "atb/atb_infer.h"
#include "atb_speed/log.h"
#include "nlohmann/json.hpp"

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
};

atb::Status QuantFALayer(const QuantFALayerParam &param, atb::Operation **operation);
static atb::Operation *CreateQuantFALayer(const nlohmann::json &paramJson)
{
    QuantFALayerParam param;
    param.rmsNormEps = paramJson["rmsNormEps"].get<double>();
    param.headNum = paramJson["headNum"].get<int>();
    param.dk = paramJson["dk"].get<int>();
    param.isFloatQueryLayer = false;
    param.isFloatKVLayer = false;
    param.isFloatDownLayer = false;
    if (paramJson.contains("inputScale_qkv")){
        param.inputScale_qkv = paramJson["inputScale_qkv"].get<float>();
    } // quant
    if (paramJson.contains("inputOffset_qkv")){
        param.inputOffset_qkv = paramJson["inputOffset_qkv"].get<int>();
    }
    if (paramJson.contains("inputScale_dense")){
        param.inputScale_dense = paramJson["inputScale_dense"].get<float>();
    }
    if (paramJson.contains("inputOffset_dense")){
        param.inputOffset_dense = paramJson["inputOffset_dense"].get<int>();
    }
    if (paramJson.contains("inputScale_gate_up")){
        param.inputScale_dense = paramJson["inputScale_gate_up"].get<float>();
    }
    if (paramJson.contains("inputOffset_gate_up")){
        param.inputOffset_gate_up = paramJson["inputOffset_gate_up"].get<int>();
    }
    if (paramJson.contains("inputOffset_down_proj")){
        param.inputOffset_gate_up = paramJson["inputOffset_down_proj"].get<int>();
    }
    atb::Operation *op;
    QuantFALayer(param, &op);
    return op;
}
}
}
#endif