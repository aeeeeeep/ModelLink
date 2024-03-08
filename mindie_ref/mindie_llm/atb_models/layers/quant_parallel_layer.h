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

#ifndef ATB_SPEED_LAYERS_QUANT_PARALLEL_LAYER_H
#define ATB_SPEED_LAYERS_QUANT_PARALLEL_LAYER_H
#include <atb/atb_infer.h>
#include "atb_speed/log.h"
#include "nlohmann/json.hpp"
#include "common.h"

namespace atb_speed {
namespace common {
struct QuantParallelParam {
    int rank = 0;
    int rankSize = 1;
    int rankRoot = 0;
    void *hcclComm = nullptr;
    bool isBias = false;
    bool transposeA = false;
    bool transposeB = false;
    std::string backend = "hccl";
};

class QuantLinearWithBiasAndParallel : public CommonOpBase {
public:
    using CommonOpBase::CommonOpBase;

    enum LinearWithBiasAndParallelId : unsigned int {
        IN_INPUT = 0,
        IN_WEIGHT,
        IN_DEQUANT_SCALE,
        IN_DEQUANT_OFFSET,
        IN_BIAS,
        OUT_LINEAROUT,
        INTERMIDATE_MATMULOUT,
        INTERMIDATE_ALLREDUCEOUT,
    };
};

class QuantLinearWithParallel : public CommonOpBase {
public:
    using CommonOpBase::CommonOpBase;

    enum LinearWithParallelId : unsigned int {
        IN_INPUT = 0,
        IN_WEIGHT,
        IN_DEQUANT_SCALE,
        IN_DEQUANT_OFFSET,
        INTERMIDATE_ALLREDUCEOUT,
        INTERMIDATE_MATMULOUT,
        IN_BIAS,
        OUT_LINEAROUT,
    };
};

class QuantLinearWithBias : public CommonOpBase {
public:
    using CommonOpBase::CommonOpBase;

    enum LinearWithBiasId : unsigned int {
        IN_INPUT = 0,
        IN_WEIGHT,
        IN_DEQUANT_SCALE,
        IN_DEQUANT_OFFSET,
        IN_BIAS,
        OUT_LINEAROUT,
        INTERMIDATE_MATMULOUT,
        INTERMIDATE_ALLREDUCEOUT,
    };
};

class QuantLinearOnly : public CommonOpBase {
public:
    using CommonOpBase::CommonOpBase;

    enum LinearOnlyId : unsigned int {
        IN_INPUT = 0,
        IN_WEIGHT,
        IN_DEQUANT_SCALE,
        IN_DEQUANT_OFFSET,
        INTERMIDATE_MATMULOUT,
        IN_BIAS,
        OUT_LINEAROUT,
        INTERMIDATE_ALLREDUCEOUT,
    };
};

atb::Status QuantRowParallelLinear(const QuantParallelParam &param, atb::Operation **operation);
atb::Status QuantColumnParallelLinear(const QuantParallelParam &param, atb::Operation **operation);
atb::Status QuantVocabParallelEmbedding(const QuantParallelParam &param, atb::Operation **operation);
} // namespace common
} // namespace atb_speed

#endif
