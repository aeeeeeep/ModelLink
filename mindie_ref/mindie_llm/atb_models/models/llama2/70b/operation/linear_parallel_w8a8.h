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

#ifndef ASCEND_SPEED_INFERENCE_LINEAR_PARALLEL_W8A8_H
#define ASCEND_SPEED_INFERENCE_LINEAR_PARALLEL_W8A8_H

#include <atb/atb_infer.h>

namespace atb_speed {
namespace llama2_70b {
struct LinearParallelW8A8Param {
    bool transWeight = false;
    int rank = 0;
    int rankSize = 0;
    int rankRoot = 0;
    std::string bias = "";
    std::string parallelType = "RowParallel";
    std::string backend = "hccl";
    HcclComm hcclComm = nullptr; // only effect when hcclComm is not null
};

atb::Status CreateLinearParallelW8A8(const LinearParallelW8A8Param &param, atb::Operation **operation);
} // namespace llama2_70b
} // namespace atb_speed

#endif
