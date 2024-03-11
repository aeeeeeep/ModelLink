/*
* Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
*  * you may not use this file except in compliance with the License.
*  * You may obtain a copy of the License at
*  *
*  * http://www.apache.org/licenses/LICENSE-2.0
*  *
*  * Unless required by applicable law or agreed to in writing, software
*  * distributed under the License is distributed on an "AS IS" BASIS,
*  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  * See the License for the specific language governing permissions and
*  * limitations under the License.
*  */
#ifndef ATB_SPEED_MODELS_DEEPSEEK_DENSE_MASK_SPLIT_OPERATION_H
#define ATB_SPEED_MODELS_DEEPSEEK_DENSE_MASK_SPLIT_OPERATION_H
#include <atb/atb_infer.h>

namespace atb_speed {
namespace deepseekDense {
struct DeepseekDenseMaskSplitParam {
    int splitDim = 0;
    int splitSize = 2;
};

atb::Status CreateDeepseekDenseMaskSplitOperation(
    const DeepseekDenseMaskSplitParam &param, atb::Operation **operation);
}
} // namespace atb_speed
#endif