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
#ifndef ATB_SPEED_MODELS_LLAMA_MLP_W8A8_OPERATION_H
#define ATB_SPEED_MODELS_LLAMA_MLP_W8A8_OPERATION_H
#include <atb/atb_infer.h>

namespace atb_speed {
namespace llama2_70b {
struct MlpW8A8Param {
    int rank = 0;
    int rankSize = 1;
    int rankRoot = 0;
    void *hcclComm = nullptr;
    atb::infer::ActivationType activationType;
    bool transposeB = false;
    bool isBias = false;
    bool isPack = false;
    std::string backend = "hccl";
};

atb::Status CreateMlpW8A8Operation(const MlpW8A8Param &param, atb::Operation **operation);
} // namespace llama2_70b
} // namespace atb_speed
#endif