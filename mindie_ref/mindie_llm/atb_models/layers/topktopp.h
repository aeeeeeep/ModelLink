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
#ifndef ATB_SPEED_LAYERS_TOPKTOPP_H
#define ATB_SPEED_LAYERS_TOPKTOPP_H

#include "atb/atb_infer.h"
#include "atb_speed/log.h"
#include "nlohmann/json.hpp"
  
namespace atb_speed {
namespace layers {
struct TopktoppParam {
    int64_t axes = 0;
    int32_t headNum = 0;
    int32_t topk = 3;
    int32_t min_tokens_to_keep = 2;
    int32_t vocsize = 32000;
    int32_t row = 1;
    int32_t randseed = 0;
    float filter_value=-INFINITY;
  
};

atb::Status Topktopp(const TopktoppParam &param, atb::Operation **operation);

}
} 
#endif
