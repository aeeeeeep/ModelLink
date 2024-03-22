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

#ifndef ATB_SPEED_ATTENTION_LAYERS_H
#define ATB_SPEED_ATTENTION_LAYERS_H

#include <atb/atb_infer.h>
#include "atb_speed/log.h"
#include "atb_speed/utils/operation_util.h"
#include "layers/parallel_layer_v2.h"

namespace atb_speed {
namespace common {

enum FlashAttentionWithROPEId : int {
    IN_HIDDENSTATES = 0,             // [batch, seqLen, hiddenSize]
    IN_WEIGHT_MIXEDQKV,              // [hiddenSize, 3*headNum*headDim]
    IN_WEIGHT_SELFOUT,
    IN_BIAS_MIXEDQKV,               // optional
    IN_BIAS_SELFOUT,                // optional
    IN_DEQSCALE_MIXEDQKV,           // 量化独有权重
    IN_DEQSCALE_SELFOUT,            // 量化独有权重

    IN_QKVMIXDWEIGHT_INDEX,         // 稀疏独有权重
    IN_QKVMIXDOFFSETX,
    IN_QKVMIXDWEIGHT_COMPRESSINFO,
    IN_SELFOUTLINEARWEIGHT_INDEX,
    IN_SELFOUTLINEAROFFSETX,
    IN_SELFOUTLINEARWEIGHT_COMPRESSINFO,

    IN_ROPE_COS,
    IN_ROPE_SIN,
    IN_SEQLEN,
    IN_CACHED_K,
    IN_CACHED_V,
    IN_ATTENTION_MASK,
    IN_TOKEN_OFFSET,
    IN_LAYER_ID,
    IN_HOLDER,

    OUT_RESULT_ID,                      // [batch, seqLen, hiddenSize]

    INTERMEDIATE_MIXED_QKV,               // [batch, seqLen, 3*hiddenSize]
    INTERMEDIATE_QUERY,
    INTERMEDIATE_KEY,
    INTERMEDIATE_VALUE,
    INTERMEDIATE_POSITIONEMBED_Q,
    INTERMEDIATE_POSITIONEMBED_K,
    INTERMEDIATE_SELFOUT,            // [batch, seqLen, 3*hiddenSize]
    INTERMEDIATE_KV,
};

enum class PositionEmbeddingTensorId : int {
    IN_QUERY = 0,
    IN_KEY,
    IN_ROPE_COS,
    IN_ROPE_SIN,
    IN_SEQLEN,

    OUT_QUERY,
    OUT_KEY,

    INTERMEDIATE_QCHUNK0,
    INTERMEDIATE_QCHUNK1,
    INTERMEDIATE_KCHUNK0,
    INTERMEDIATE_KCHUNK1,
    INTERMEDIATE_QOUT,
    INTERMEDIATE_KOUT,
};

struct FTWithROPEParam {
    // QKV linear param
    bool isCrossedWeight = false;
    atb_speed::common::ParallelParamV2 mixdQkvLinearParam;
    // self attention param
    int isGroupedQueryAttention = false;
    int faHeadDim = 0;
    atb::infer::SelfAttentionParam selfAttentionKvCacheParam;
    // rotary coeff
    int rotaryCoeff = 2;
    bool isHalfRotary = false;
    // self out linear param
    atb_speed::common::ParallelParamV2 selfOutLinearParam;
};

class FlashAttentionWithPosEmbedding {
public:
    // Currently supported: RoPE, Alibi
    virtual atb::Status FlashAttentionWithPositionEmbeddingLayer(const FTWithROPEParam &param,
                                                                 atb::Operation **operation) final;

private:
    atb::Status PositionEmbedding(const FTWithROPEParam &param, atb::Operation **operation);
};
} // namespace common
} // namespace atb_speed
#endif
