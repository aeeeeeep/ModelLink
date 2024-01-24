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
#ifndef ATB_SPEED_MODELS_LLAMA_PARALLEL_LMHEAD_H
#define ATB_SPEED_MODELS_LLAMA_PARALLEL_LMHEAD_H

#include <atb/atb_infer.h>
#include "models/llama_parallel/operation/linear.h"
#include "models/llama_parallel/operation/linear_parallel.h"

namespace atb_speed {
namespace llama_parallel {
struct LmHeadParam {
    bool gatherAhead = false;  // Prefill阶段使用gatherAhead，只获取最后最后一个token，以此减少显存占用
    bool unpadInputs = false;
    int hiddenSizePerAttentionHead = 0;  // 当Parallel的类型为ROW PARALLEL时，需要此参数切分Gather算子的输出结果
    atb_speed::llama_parallel::LinearParallelParam linearParallelParam;
};

atb::Status LmHead(const LmHeadParam &param, atb::Operation **operation);
} // namespace llama_parallel
} // namespace atb_speed
#endif
