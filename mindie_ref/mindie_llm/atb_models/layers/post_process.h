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

#ifndef ATB_SPEED_LAYERS_POST_PROCESS_LAYER_H
#define ATB_SPEED_LAYERS_POST_PROCESS_LAYER_H

#include "atb_speed/log.h"
#include "atb_speed/utils/operation_factory.h"
#include "common.h"
#include "nlohmann/json.hpp"
#include <atb/atb_infer.h>

namespace atb_speed {
namespace common {

struct PostProcessParam {
    double temperature = 1.0;
    int topK = 0;
    int randSeed = 0;
};

atb::Status Sample(const PostProcessParam &param, atb::Operation **operation);

static atb::Operation *SampleLayerCreate(const nlohmann::json &paramJson)
{
    atb_speed::common::PostProcessParam param;
    param.temperature = paramJson["temperature"].get<double>();
    param.topK = paramJson["topK"].get<uint32_t>();
    param.randSeed = paramJson["randSeed"].get<uint32_t>();
    ATB_LOG(INFO) << "SampleLayerCreate: temperature:" << param.temperature << ", topK:" << param.topK
                  << ", randSeed:" << param.randSeed;
    atb::Operation *op;
    Sample(param, &op);
    return op;
}

REGISTER_OPERATION(common, SampleLayerCreate);

} // namespace common
} // namespace atb_speed
#endif
