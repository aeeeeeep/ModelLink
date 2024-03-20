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
#ifndef ATB_SPEED_MODELS_STAR_CODER_PA_QUANT_LAYER_H
#define ATB_SPEED_MODELS_STAR_CODER_PA_QUANT_LAYER_H

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtype-limits"
#include <nlohmann/json.hpp>
#pragma GCC diagnostic pop
#include <atb/atb_infer.h>
#include "atb_speed/log.h"
#include "atb_speed/base/hosttensor_binder.h"

namespace atb_speed {
namespace star_coder {
struct PAQuantLayerParam {
    float layerNormEps = 0;
    int headNum = 0;
    int dk = 0;
    int kvHead = 1;
    int rank = 0;
    int worldSize = 1;
    std::string backend = "hccl";
    std::string model = "star_coder";
    int numAttentionHeadsPerRank = 0;
    int hiddenSizePerAttentionHead = 0;
    int numKeyValueHeadsPerRank = 0;
    int layerId = 0;
    bool isFA = true;
    bool isPrefill = false;
    bool isBF16 = false;
    bool isPack = true;
    bool supportSwiGLU = false;
    int quantType = 0;
    bool quantmodel = false;
    std::vector<int> seqLen;
    std::vector<int> packQuantType = {};  // 两个元素，第一个元素代表QKV pack的量化类型，第二个元素代表MLP pack的量化类型
    // 七个元素，分别代表q，k，v，self attention out，gate，up，down linear的类型
    std::vector<int> linearQuantType = {};
};

enum PAQuantLayerTensorId : uint32_t {
    IN_RESIDUAL_ADD_OUT = 0,
    IN_HIDDEN_STATES,                   // shape: FA: [batchSize, seqLen, maxPositionEmbeddings] PA: [seqLen, hiddenSize]
    IN_INPUT_NORM_WEIGHT,               // 6144
    IN_INPUT_NORM_BIAS,                 // 6144
    IN_INPUT_NORM_NEW_WEIGHT,
    IN_INPUT_NORM_NEW_BIAS,
    IN_QKV_WEIGHT_0,                    // [1024, 6144]
                                        // No pack:
    IN_QKV_DEOFFSET_0,                  // [1024]
    IN_QKV_DESCALE_0,                   // Quant所需权重
    IN_QKV_OFFSET_0,                    // Quant所需权重
    IN_QKV_SCALE_0,                     // Quant所需权重
    IN_QKV_WEIGHT_1,                    // Pack: no usage; No pack: (K) shape: [numKeyValueHeadsPerRank * hiddenSizePerAttentionHead, hiddenSize]
    IN_QKV_DEOFFSET_1,                  // Quant所需权重
    IN_QKV_DESCALE_1,                   // Quant所需权重
    IN_QKV_OFFSET_1,                    // Quant所需权重
    IN_QKV_SCALE_1,                     // Quant所需权重
    IN_QKV_WEIGHT_2,                    // Pack: no usage; No pack: (V) shape: [numKeyValueHeadsPerRank * hiddenSizePerAttentionHead, hiddenSize]
    IN_QKV_DEOFFSET_2,                  // Quant所需权重
    IN_QKV_DESCALE_2,                   // Quant所需权重
    IN_QKV_OFFSET_2,                    // Quant所需权重
    IN_QKV_SCALE_2,                     // Quant所需权重
    IN_ATTENTION_OUT_WEIGHT,            // [6144, 768] shape: [hiddenSize, numAttentionHeadsPerRank * hiddenSizePerAttentionHead]
    IN_ATTENTION_OUT_DEOFFSET,          // [6144] [hiddenSize]
    IN_ATTENTION_OUT_DESCALE,           // Quant所需权重
    IN_ATTENTION_OUT_OFFSET,            // Quant所需权重
    IN_ATTENTION_OUT_SCALE,             // Quant所需权重
    IN_ATTENTION_NORM_WEIGHT,           // [6144] [hiddenSize]
    IN_ATTENTION_NORM_BIAS,             // [6144] [hiddenSize]
    IN_ATTENTION_NORM_NEW_WEIGHT,
    IN_ATTENTION_NORM_NEW_BIAS,
    IN_MLP_WEIGHT_0,                    // [3072, 6144] Pack: shape: [2 * intermediateSizePerRank, hiddenSize]
                                        // No pack: (Gate) shape: [intermediateSizePerRank, hiddenSize]
    IN_MLP_DEOFFSET_0,                  // [3072]
    IN_MLP_DESCALE_0,                   // Quant所需权重
    IN_MLP_OFFSET_0,                    // Quant所需权重
    IN_MLP_SCALE_0,                     // Quant所需权重
    IN_MLP_WEIGHT_1,                    // Pack: no usage; No pack: (Up) shape: [intermediateSizePerRank, hiddenSize]
    IN_MLP_DEOFFSET_1,                  // Quant所需权重
    IN_MLP_DESCALE_1,                   // Quant所需权重
    IN_MLP_OFFSET_1,                    // Quant所需权重
    IN_MLP_SCALE_1,                     // Quant所需权重
    IN_MLP_DOWN_WEIGHT,                 // [6144, 3072]
    IN_MLP_DOWN_DEOFFSET,               // [6144]
    IN_MLP_DOWN_DESCALE,                // Quant所需权重
    IN_MLP_DOWN_OFFSET,                 // Quant所需权重
    IN_MLP_DOWN_SCALE,                  // Quant所需权重
    IN_ATTENTION_MASK,                  // shape: FA: [batchSize, maxPositionEmbeddings, maxPositionEmbeddings] PA: [seqLen, seqLen]
    IN_BLOCK_TABLES,                    // shape: [seqLen, seqLen]; PA所需参数
    IN_SLOTS,                           // shape: [seqLen]; PA所需参数
    IN_SEQ_LEN,                         // shape: [batchSize]
    IN_PLACE_HOLDER,                    // shape: [1]
    IN_K_CACHE,                         // shape: FA: [batchSize, maxPositionEmbeddings, numKeyValueHeadsPerRank * hiddenSizePerAttentionHead] PA: [2622, hiddenSizePerAttentionHead, numAttentionHeadsPerRank, hiddenSizePerAttentionHead]
    IN_V_CACHE,                         // shape: FA: [batchSize, maxPositionEmbeddings, numKeyValueHeadsPerRank * hiddenSizePerAttentionHead] PA: [2622, hidd
    
    OUT_ATTENTION_RESIDUAL_ADD,         // shape: FA: [batchSize, seqLen, maxPositionEmbeddings] PA: [seqLen, hiddenSize]
    OUT_MLP,

    INTERMEDIATE_ATTENTION_OUT,         // shape: PA: [seqLen, hiddenSize]
};

atb::Status PAQuantLayer(const PAQuantLayerParam &param, atb::Operation **operation);

class StarCoderPAQuantHostBinder : public HostTensorBinder {
public:
    StarCoderPAQuantHostBinder();

    virtual ~StarCoderPAQuantHostBinder();

    void ParseParam(const nlohmann::json &paramJson) override;

    void BindTensor(atb::VariantPack &variantPack) override;

private:
    std::vector<int32_t> seqLen_;
};
} // namespace star_coder
} // namespace atb_speed
#endif