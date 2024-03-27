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
#ifndef ATB_SPEED_MODELS_BAICHUAN2_13B_PA_QUANT_LAYER_H
#define ATB_SPEED_MODELS_BAICHUAN2_13B_PA_QUANT_LAYER_H

#include <atb/atb_infer.h>
#include <nlohmann/json.hpp>

#include <vector>
#include "atb_speed/base/hosttensor_binder.h"
#include "atb_speed/log.h"
#include "atb_speed/utils/str_split.h"

namespace atb_speed {
namespace baichuan2_13b {
struct PAQuantLayerParam {
    int dk = 0;
    int headNum = 0;
    int rankSize = 1;
    bool isFA = true;
    bool isPrefill = false;
    bool isBF16 = false;
    bool isPack = true;
    bool supportSwiGLU = true;  // baichuan2-13b默认为true
    int quantType = 0;
    float rmsNormEps = 0;
    int numAttentionHeadsPerRank = 0;
    int hiddenSizePerAttentionHead = 0;
    int numKeyValueHeadsPerRank = 0;
    bool supportLcoc = false;
    int layerId = 0;
    bool transposedWeight = true;
    std::string backend = "hccl";
    std::string model = "baichuan2_13b";

    int rank = 0;
    int worldSize = 1;
    std::vector<int> seqLen;
    std::vector<int> packQuantType = {};  // 两个元素，第一个元素代表QKV pack的量化类型，第二个元素代表MLP pack的量化类型
    // 七个元素，分别代表q，k，v，self attention out，gate，up，down linear的类型
    std::vector<int> linearQuantType = {};
};

enum LayerQuantPATensorId : uint32_t {
    IN_RESIDUAL_ADD_OUT = 0,    
    IN_HIDDEN_STATES,       
    IN_INPUT_NORM_WEIGHT,               // shape: [hiddenSize]
    IN_INPUT_NORM_BIAS,
    IN_INPUT_NORM_NEW_WEIGHT,
    IN_INPUT_NORM_NEW_BIAS,
    IN_QKV_WEIGHT_0,                    // Pack: shape: MHA [3 * numAttentionHeadsPerRank * hiddenSizePerAttentionHead, hiddenSize] GQA [(numAttentionHeadsPerRank + 2 * numKeyValueHeadsPerRank) * hiddenSizePerAttentionHead, hiddenSize]
    IN_QKV_DEOFFSET_0,                  // Quant所需权重
    IN_QKV_DESCALE_0,                   // Quant所需权重
    IN_QKV_OFFSET_0,                    // Quant所需权重
    IN_QKV_SCALE_0,                     // Quant所需权重
    IN_QKV_COMPRESS_IDX_0,              // Quant所需权重
    IN_QKV_WEIGHT_1,                    // Pack: no usage; No pack: (K) shape: [numKeyValueHeadsPerRank * hiddenSizePerAttentionHead, hiddenSize]
    IN_QKV_DEOFFSET_1,                  // Quant所需权重
    IN_QKV_DESCALE_1,                   // Quant所需权重
    IN_QKV_OFFSET_1,                    // Quant所需权重
    IN_QKV_SCALE_1,                     // Quant所需权重
    IN_QKV_COMPRESS_IDX_1,              // Quant所需权重
    IN_QKV_WEIGHT_2,                    // Pack: no usage; No pack: (V) shape: [numKeyValueHeadsPerRank * hiddenSizePerAttentionHead, hiddenSize]
    IN_QKV_DEOFFSET_2,                  // Quant所需权重
    IN_QKV_DESCALE_2,                   // Quant所需权重
    IN_QKV_OFFSET_2,                    // Quant所需权重
    IN_QKV_SCALE_2,                     // Quant所需权重
    IN_QKV_COMPRESS_IDX_2,              // Quant所需权重
    IN_ATTENTION_OUT_WEIGHT,            // shape: [hiddenSize, numAttentionHeadsPerRank * hiddenSizePerAttentionHead]
    IN_ATTENTION_OUT_DEOFFSET,          // Quant所需权重
    IN_ATTENTION_OUT_DESCALE,           // Quant所需权重
    IN_ATTENTION_OUT_OFFSET,            // Quant所需权重
    IN_ATTENTION_OUT_SCALE,             // Quant所需权重
    IN_ATTENTION_OUT_COMPRESS_IDX,    // Quant所需权重
    IN_ATTENTION_NORM_WEIGHT,           // shape: [hiddenSize]
    IN_ATTENTION_NORM_BIAS,
    IN_ATTENTION_NORM_NEW_WEIGHT,
    IN_ATTENTION_NORM_NEW_BIAS,
    IN_MLP_WEIGHT_0,                    // Pack: shape: [2 * intermediateSizePerRank, hiddenSize]
    IN_MLP_DEOFFSET_0,                  // Quant所需权重
    IN_MLP_DESCALE_0,                   // Quant所需权重
    IN_MLP_OFFSET_0,                    // Quant所需权重
    IN_MLP_SCALE_0,                     // Quant所需权重
    IN_MLP_COMPRESS_IDX_0,              // Quant所需权重
    IN_MLP_WEIGHT_1,                    // Pack: no usage; No pack: (Up) shape: [intermediateSizePerRank, hiddenSize]
    IN_MLP_DEOFFSET_1,                  // Quant所需权重
    IN_MLP_DESCALE_1,                   // Quant所需权重
    IN_MLP_OFFSET_1,                    // Quant所需权重
    IN_MLP_SCALE_1,                     // Quant所需权重
    IN_MLP_COMPRESS_IDX_1,              // Quant所需权重
    IN_MLP_DOWN_WEIGHT,                 // shape: [hiddenSize, intermediateSizePerRank]
    IN_MLP_DOWN_DEOFFSET,               // Quant所需权重
    IN_MLP_DOWN_DESCALE,                // Quant所需权重
    IN_MLP_DOWN_OFFSET,                 // Quant所需权重
    IN_MLP_DOWN_SCALE,                  // Quant所需权重
    IN_MLP_DOWN_COMPRESS_IDX,         // Quant所需权重
    IN_ATTENTION_MASK,                  // shape: FA: [batchSize, maxPositionEmbeddings, maxPositionEmbeddings] PA: [seqLen, seqLen]
    IN_K_CACHE,                         // shape: FA: [batchSize, maxPositionEmbeddings, numKeyValueHeadsPerRank * hiddenSizePerAttentionHead] PA: [2622, hiddenSizePerAttentionHead, numAttentionHeadsPerRank, hiddenSizePerAttentionHead]
    IN_V_CACHE,                         // shape: FA: [batchSize, maxPositionEmbeddings, numKeyValueHeadsPerRank * hiddenSizePerAttentionHead] PA: [2622, hiddenSizePerAttentionHead, numAttentionHeadsPerRank, hiddenSizePerAttentionHead]
    IN_INPUT_LENGTHS,
    IN_BLOCK_TABLES,                    // shape: [seqLen, seqLen]; PA所需参数
    IN_SLOTS,  
    IN_PLACE_HOLDER,                    // shape: [1]
    OUT_ATTENTION_RESIDUAL_ADD,  // 当前layer的attention out输出
    OUT_MLP,  // 当前layer的mlp输出
    INTERMEDIATE_ATTENTION_OUT,         // shape: PA: [seqLen, hiddenSize]
};


atb::Status PAQuantLayer(const PAQuantLayerParam &param, atb::Operation **operation);


class PAQuantLayerHostBinder : public HostTensorBinder {
public:
    PAQuantLayerHostBinder();
    virtual ~PAQuantLayerHostBinder();
    void ParseParam(const nlohmann::json &paramJson) override;
    void BindTensor(atb::VariantPack &variantPack) override;

private:
    std::vector<int> seqLen_;
    int32_t layerId_ = 0;
};
} // namespace baichuan2_13b
} // namespace atb_speed
#endif