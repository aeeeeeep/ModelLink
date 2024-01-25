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
#ifndef ATB_SPEED_MODELS_LLAMA_PARALLEL_DECODER_LAYER_H
#define ATB_SPEED_MODELS_LLAMA_PARALLEL_DECODER_LAYER_H

#include "nlohmann/json.hpp"

#include "atb/atb_infer.h"
#include "atb_speed/base/hosttensor_binder.h"
#include "atb_speed/log.h"

namespace atb_speed {
namespace llama_parallel {
struct DecoderLayerParam {
    bool isFA = true;
    bool isPrefill = false;
    bool isBF16 = false;
    bool isPack = true;
    int quantType = 0;
    float rmsNormEps = 0;
    int numAttentionHeadsPerRank = 0;
    int hiddenSizePerAttentionHead = 0;
    int numKeyValueHeadsPerRank = 0;
    int rank = 0;
    int worldSize = 1;
    std::string backend = "hccl";
    std::vector<int> seqLen;
    std::vector<int> tokenOffset;
};

enum DecoderLayerTensorIdx : uint32_t {
    IN_HIDDEN_STATES = 0,               // shape: FA: [batchSize, seqLen, maxPositionEmbeddings] PA: [seqLen, hiddenSize]
    IN_INPUT_NORM_WEIGHT,               // shape: [hiddenSize]
    IN_QKV_WEIGHT_0,                    // Pack: shape: MHA [3 * numAttentionHeadsPerRank * hiddenSizePerAttentionHead, hiddenSize] GQA [(numAttentionHeadsPerRank + 2 * numKeyValueHeadsPerRank) * hiddenSizePerAttentionHead, hiddenSize]
                                        // No pack: (Q) shape: [numAttentionHeadsPerRank * hiddenSizePerAttentionHead, hiddenSize]
    IN_QKV_SCALE_0,                     // Quant所需权重
    IN_QKV_OFFSET_0,                    // Quant所需权重
    IN_QKV_DESCALE_0,                   // Quant所需权重
    IN_QKV_WEIGHT_1,                    // Pack: no usage; No pack: (K) shape: [numKeyValueHeadsPerRank * hiddenSizePerAttentionHead, hiddenSize]
    IN_QKV_SCALE_1,                     // Quant所需权重
    IN_QKV_OFFSET_1,                    // Quant所需权重
    IN_QKV_DESCALE_1,                   // Quant所需权重
    IN_QKV_WEIGHT_2,                    // Pack: no usage; No pack: (V) shape: [numKeyValueHeadsPerRank * hiddenSizePerAttentionHead, hiddenSize]
    IN_QKV_SCALE_2,                     // Quant所需权重
    IN_QKV_OFFSET_2,                    // Quant所需权重
    IN_QKV_DESCALE_2,                   // Quant所需权重
    IN_ATTENTION_OUT_WEIGHT,            // shape: [hiddenSize, numAttentionHeadsPerRank * hiddenSizePerAttentionHead]
    IN_ATTENTION_OUT_SCALE,             // Quant所需权重
    IN_ATTENTION_OUT_OFFSET,            // Quant所需权重
    IN_ATTENTION_OUT_DESCALE,           // Quant所需权重
    IN_ATTENTION_NORM_WEIGHT,           // shape: [hiddenSize]
    IN_MLP_WEIGHT_0,                    // Pack: shape: [2 * intermediateSizePerRank, hiddenSize]
                                        // No pack: (Gate) shape: [intermediateSizePerRank, hiddenSize]
    IN_MLP_SCALE_0,                     // Quant所需权重
    IN_MLP_OFFSET_0,                    // Quant所需权重
    IN_MLP_DESCALE_0,                   // Quant所需权重
    IN_MLP_WEIGHT_1,                    // Pack: no usage; No pack: (Up) shape: [intermediateSizePerRank, hiddenSize]
    IN_MLP_SCALE_1,                     // Quant所需权重
    IN_MLP_OFFSET_1,                    // Quant所需权重
    IN_MLP_DESCALE_1,                   // Quant所需权重
    IN_MLP_DOWN_WEIGHT,                 // shape: [hiddenSize, intermediateSizePerRank]
    IN_MLP_DOWN_SCALE,                  // Quant所需权重
    IN_MLP_DOWN_OFFSET,                 // Quant所需权重
    IN_MLP_DOWN_DESCALE,                // Quant所需权重
    IN_COS_TABLE,                       // shape: FA: [batchSize * seqLen, hiddenSizePerAttentionHead] PA: [seqLen, hiddenSizePerAttentionHead]
    IN_SIN_TABLE,                       // shape: FA: [batchSize * seqLen, hiddenSizePerAttentionHead] PA: [seqLen, hiddenSizePerAttentionHead]
    IN_ATTENTION_MASK,                  // shape: FA: [batchSize, maxPositionEmbeddings, maxPositionEmbeddings] PA: [seqLen, seqLen]
    IN_K_CACHE,                         // shape: FA: [batchSize, maxPositionEmbeddings, numKeyValueHeadsPerRank * hiddenSizePerAttentionHead] PA: [2622, hiddenSizePerAttentionHead, numAttentionHeadsPerRank, hiddenSizePerAttentionHead]
    IN_V_CACHE,                         // shape: FA: [batchSize, maxPositionEmbeddings, numKeyValueHeadsPerRank * hiddenSizePerAttentionHead] PA: [2622, hiddenSizePerAttentionHead, numAttentionHeadsPerRank, hiddenSizePerAttentionHead]
    IN_SEQ_LEN,                         // shape: [batchSize]
    IN_PLACE_HOLDER,                    // shape: [1]
    IN_BETA,                            // shape: [hiddenSize]; Quant所需参数
    IN_TOKEN_OFFSET,                    // shape: [batchSize]; FA所需参数
    IN_LAYER_ID,                        // shape: [1]; FA所需参数
    IN_BLOCK_TABLES,                    // shape: [seqLen, seqLen]; PA所需参数
    IN_SLOTS,                           // shape: [seqLen]; PA所需参数

    OUT_DECODER_LAYER,                  // shape: FA: [batchSize, seqLen, maxPositionEmbeddings] PA: [seqLen, hiddenSize]

    INTERMEDIATE_INPUT_NORM_OUT,        // shape: PA: [seqLen, hiddenSize]
    INTERMEDIATE_ATTENTION_OUT,         // shape: PA: [seqLen, hiddenSize]
    INTERMEDIATE_RESIDUAL_ADD_OUT,      // shape: PA: [seqLen, hiddenSize]
    INTERMEDIATE_ATTENTION_NORM_OUT,    // shape: PA: [seqLen, hiddenSize]
    INTERMEDIATE_MLP_OUT,               // shape: PA: [seqLen, hiddenSize]
};

atb::Status DecoderLayer(const DecoderLayerParam &param, atb::Operation **operation);

class DecoderLayerBinder : public HostTensorBinder {
public:
    DecoderLayerBinder();
    virtual ~DecoderLayerBinder();
    void ParseParam(const nlohmann::json &paramJson) override;
    void BindTensor(atb::VariantPack &variantPack) override;

private:
    std::vector<int> tokenOffset_;
    std::vector<int> seqLen_;
    int32_t layerId_ = 0;
};

}  // namespace llama_parallel
}  // namespace atb_speed
#endif