/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
#ifndef ATB_SPEED_MODELS_DEEPSEEK_DENSE_DECODER_LAYER_H
#define ATB_SPEED_MODELS_DEEPSEEK_DENSE_DECODER_LAYER_H

#include <vector>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtype-limits"
#include "nlohmann/json.hpp"
#pragma GCC diagnostic pop

#include "atb/atb_infer.h"
#include "atb_speed/base/hosttensor_binder.h"
#include "atb_speed/log.h"

namespace atb_speed {
namespace deepseekDense {
struct DecoderLayerParam {
    bool isFA = true;
    bool isPrefill = false;
    bool isBF16 = false;
    bool isPack = true;
    bool supportSwiGLU = false;
    bool supportLcoc = false;
    int quantType = 0;
    float rmsNormEps = 0;
    bool transpose = true;
    int numOfExperts = 64;
    int layerId = 0;
    int numAttentionHeadsPerRank = 0;
    int hiddenSizePerAttentionHead = 0;
    int numKeyValueHeadsPerRank = 0;
    int rank = 0;
    int worldSize = 1;
    std::string backend = "hccl";
    std::vector<int> seqLen;
    std::vector<int> tokenOffset;
    std::vector<int> packQuantType = {};  // 两个元素，第一个元素代表QKV pack的量化类型，第二个元素代表MLP pack的量化类型
    // 七个元素，分别代表q，k，v，self attention out，gate，up，down linear的类型
    std::vector<int> linearQuantType = {};
};

enum DecoderLayerTensorIdx : uint32_t {
    IN_HIDDEN_STATES = 0,               // shape: FA: [batchSize, seqLen, maxPositionEmbeddings] PA: [seqLen, hiddenSize]
    IN_INPUT_NORM_WEIGHT,               // shape: [hiddenSize]
    IN_INPUT_NORM_BIAS,
    IN_INPUT_NORM_NEW_WEIGHT,
    IN_INPUT_NORM_NEW_BIAS,
    IN_QKV_WEIGHT_0,                    // Pack: shape: MHA [3 * numAttentionHeadsPerRank * hiddenSizePerAttentionHead, hiddenSize] GQA [(numAttentionHeadsPerRank + 2 * numKeyValueHeadsPerRank) * hiddenSizePerAttentionHead, hiddenSize]
                                        // No pack: (Q) shape: [numAttentionHeadsPerRank * hiddenSizePerAttentionHead, hiddenSize]
    IN_QKV_BIAS_0,                  // Quant所需权重
    IN_QKV_DESCALE_0,                   // Quant所需权重
    IN_QKV_OFFSET_0,                    // Quant所需权重
    IN_QKV_SCALE_0,                     // Quant所需权重
    IN_QKV_WEIGHT_1,                    // Pack: no usage; No pack: (K) shape: [numKeyValueHeadsPerRank * hiddenSizePerAttentionHead, hiddenSize]
    IN_QKV_BIAS_1,                  // Quant所需权重
    IN_QKV_DESCALE_1,                   // Quant所需权重
    IN_QKV_OFFSET_1,                    // Quant所需权重
    IN_QKV_SCALE_1,                     // Quant所需权重
    IN_QKV_WEIGHT_2,                    // Pack: no usage; No pack: (V) shape: [numKeyValueHeadsPerRank * hiddenSizePerAttentionHead, hiddenSize]
    IN_QKV_BIAS_2,                  // Quant所需权重
    IN_QKV_DESCALE_2,                   // Quant所需权重
    IN_QKV_OFFSET_2,                    // Quant所需权重
    IN_QKV_SCALE_2,                     // Quant所需权重
    IN_ATTENTION_OUT_WEIGHT,            // shape: [hiddenSize, numAttentionHeadsPerRank * hiddenSizePerAttentionHead]
    IN_ATTENTION_OUT_BIAS,          // Quant所需权重
    IN_ATTENTION_OUT_DESCALE,           // Quant所需权重
    IN_ATTENTION_OUT_OFFSET,            // Quant所需权重
    IN_ATTENTION_OUT_SCALE,             // Quant所需权重

    IN_SELFATTENTION_OUT_NORM_WEIGHT,
    IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT,
    IN_MLP_DOWN_WEIGHT_EXPERT_SHARED_EXPERT,
    IN_BLOCK_SPARSE_MOE_GATE_WEIGHT,
    IN_MLP_GATEUP_WEIGHT_EXPERT_ZERO,
    IN_MLP_DOWN_WEIGHT_EXPERT_ZERO,
    IN_MLP_GATEUP_WEIGHT_EXPERT_ONE,
    IN_MLP_DOWN_WEIGHT_EXPERT_ONE,
    IN_MLP_GATEUP_WEIGHT_EXPERT_TWO,
    IN_MLP_DOWN_WEIGHT_EXPERT_TWO,
    IN_MLP_GATEUP_WEIGHT_EXPERT_THREE,
    IN_MLP_DOWN_WEIGHT_EXPERT_THREE,
    IN_MLP_GATEUP_WEIGHT_EXPERT_FOUR,
    IN_MLP_DOWN_WEIGHT_EXPERT_FOUR,
    IN_MLP_GATEUP_WEIGHT_EXPERT_FIVE,
    IN_MLP_DOWN_WEIGHT_EXPERT_FIVE,
    IN_MLP_GATEUP_WEIGHT_EXPERT_SIX,
    IN_MLP_DOWN_WEIGHT_EXPERT_SIX,
    IN_MLP_GATEUP_WEIGHT_EXPERT_SEVEN,
    IN_MLP_DOWN_WEIGHT_EXPERT_SEVEN,
    IN_MLP_GATEUP_WEIGHT_EXPERT_EIGHT,
    IN_MLP_DOWN_WEIGHT_EXPERT_EIGHT,
    IN_MLP_GATEUP_WEIGHT_EXPERT_NINE,
    IN_MLP_DOWN_WEIGHT_EXPERT_NINE,
    IN_MLP_GATEUP_WEIGHT_EXPERT_TEN,
    IN_MLP_DOWN_WEIGHT_EXPERT_TEN,
    IN_MLP_GATEUP_WEIGHT_EXPERT_ELEVEN,
    IN_MLP_DOWN_WEIGHT_EXPERT_ELEVEN,
    IN_MLP_GATEUP_WEIGHT_EXPERT_TWELVE,
    IN_MLP_DOWN_WEIGHT_EXPERT_TWELVE,
    IN_MLP_GATEUP_WEIGHT_EXPERT_THIRTEEN,
    IN_MLP_DOWN_WEIGHT_EXPERT_THIRTEEN,
    IN_MLP_GATEUP_WEIGHT_EXPERT_FOURTEEN,
    IN_MLP_DOWN_WEIGHT_EXPERT_FOURTEEN,
    IN_MLP_GATEUP_WEIGHT_EXPERT_FIFTEENN,
    IN_MLP_DOWN_WEIGHT_EXPERT_FIFTEEN,
    IN_MLP_GATEUP_WEIGHT_EXPERT_SIXTEEN,
    IN_MLP_DOWN_WEIGHT_EXPERT_SIXTEEN,
    IN_MLP_GATEUP_WEIGHT_EXPERT_SEVENTEEN,
    IN_MLP_DOWN_WEIGHT_EXPERT_SEVENTEEN,
    IN_MLP_GATEUP_WEIGHT_EXPERT_EIGHTEEN,
    IN_MLP_DOWN_WEIGHT_EXPERT_EIGHTEEN,
    IN_MLP_GATEUP_WEIGHT_EXPERT_NINETEEN,
    IN_MLP_DOWN_WEIGHT_EXPERT_NINETEEN,
    IN_MLP_GATEUP_WEIGHT_EXPERT_TWENTY,
    IN_MLP_DOWN_WEIGHT_EXPERT_TWENTY,
    IN_MLP_GATEUP_WEIGHT_EXPERT_TWENTY_ONE,
    IN_MLP_DOWN_WEIGHT_EXPERT_TWENTY_ONE,
    IN_MLP_GATEUP_WEIGHT_EXPERT_TWENTY_TWO,
    IN_MLP_DOWN_WEIGHT_EXPERT_TWENTY_TWO,
    IN_MLP_GATEUP_WEIGHT_EXPERT_TWENTY_THREE,
    IN_MLP_DOWN_WEIGHT_EXPERT_TWENTY_THREE,
    IN_MLP_GATEUP_WEIGHT_EXPERT_TWENTY_FOUR,
    IN_MLP_DOWN_WEIGHT_EXPERT_TWENTY_FOUR,
    IN_MLP_GATEUP_WEIGHT_EXPERT_TWENTY_FIVE,
    IN_MLP_DOWN_WEIGHT_EXPERT_TWENTY_FIVE,
    IN_MLP_GATEUP_WEIGHT_EXPERT_TWENTY_SIX,
    IN_MLP_DOWN_WEIGHT_EXPERT_TWENTY_SIX,
    IN_MLP_GATEUP_WEIGHT_EXPERT_TWENTY_SEVEN,
    IN_MLP_DOWN_WEIGHT_EXPERT_TWENTY_SEVEN,
    IN_MLP_GATEUP_WEIGHT_EXPERT_TWENTY_EIGHT,
    IN_MLP_DOWN_WEIGHT_EXPERT_TWENTY_EIGHT,
    IN_MLP_GATEUP_WEIGHT_EXPERT_TWENTY_NINE,
    IN_MLP_DOWN_WEIGHT_EXPERT_TWENTY_NINE,
    IN_MLP_GATEUP_WEIGHT_EXPERT_THIRTY,
    IN_MLP_DOWN_WEIGHT_EXPERT_THIRTY,
    IN_MLP_GATEUP_WEIGHT_EXPERT_THIRTY_ONE,
    IN_MLP_DOWN_WEIGHT_EXPERT_THIRTY_ONE,
    IN_MLP_GATEUP_WEIGHT_EXPERT_THIRTY_TWO,
    IN_MLP_DOWN_WEIGHT_EXPERT_THIRTY_TWO,
    IN_MLP_GATEUP_WEIGHT_EXPERT_THIRTY_THREE,
    IN_MLP_DOWN_WEIGHT_EXPERT_THIRTY_THREE,
    IN_MLP_GATEUP_WEIGHT_EXPERT_THIRTY_FOUR,
    IN_MLP_DOWN_WEIGHT_EXPERT_THIRTY_FOUR,
    IN_MLP_GATEUP_WEIGHT_EXPERT_THIRTY_FIVE,
    IN_MLP_DOWN_WEIGHT_EXPERT_THIRTY_FIVE,
    IN_MLP_GATEUP_WEIGHT_EXPERT_THIRTY_SIX,
    IN_MLP_DOWN_WEIGHT_EXPERT_THIRTY_SIX,
    IN_MLP_GATEUP_WEIGHT_EXPERT_THIRTY_SEVEN,
    IN_MLP_DOWN_WEIGHT_EXPERT_THIRTY_SEVEN,
    IN_MLP_GATEUP_WEIGHT_EXPERT_THIRTY_EIGHT,
    IN_MLP_DOWN_WEIGHT_EXPERT_THIRTY_EIGHT,
    IN_MLP_GATEUP_WEIGHT_EXPERT_THIRTY_NINE,
    IN_MLP_DOWN_WEIGHT_EXPERT_THIRTY_NINE,
    IN_MLP_GATEUP_WEIGHT_EXPERT_FOURTY,
    IN_MLP_DOWN_WEIGHT_EXPERT_FOURTY,
    IN_MLP_GATEUP_WEIGHT_EXPERT_FOURTY_ONE,
    IN_MLP_DOWN_WEIGHT_EXPERT_FOURTY_ONE,
    IN_MLP_GATEUP_WEIGHT_EXPERT_FOURTY_TWO,
    IN_MLP_DOWN_WEIGHT_EXPERT_FOURTY_TWO,
    IN_MLP_GATEUP_WEIGHT_EXPERT_FOURTY_THREE,
    IN_MLP_DOWN_WEIGHT_EXPERT_FOURTY_THREE,
    IN_MLP_GATEUP_WEIGHT_EXPERT_FOURTY_FOUR,
    IN_MLP_DOWN_WEIGHT_EXPERT_FOURTY_FOUR,
    IN_MLP_GATEUP_WEIGHT_EXPERT_FOURTY_FIVE,
    IN_MLP_DOWN_WEIGHT_EXPERT_FOURTY_FIVE,
    IN_MLP_GATEUP_WEIGHT_EXPERT_FOURTY_SIX,
    IN_MLP_DOWN_WEIGHT_EXPERT_FOURTY_SIX,
    IN_MLP_GATEUP_WEIGHT_EXPERT_FOURTY_SEVEN,
    IN_MLP_DOWN_WEIGHT_EXPERT_FOURTY_SEVEN,
    IN_MLP_GATEUP_WEIGHT_EXPERT_FOURTY_EIGHT,
    IN_MLP_DOWN_WEIGHT_EXPERT_FOURTY_EIGHT,
    IN_MLP_GATEUP_WEIGHT_EXPERT_FOURTY_NINE,
    IN_MLP_DOWN_WEIGHT_EXPERT_FOURTY_NINE,
    IN_MLP_GATEUP_WEIGHT_EXPERT_FIFTY,
    IN_MLP_DOWN_WEIGHT_EXPERT_FIFTY,
    IN_MLP_GATEUP_WEIGHT_EXPERT_FIFTY_ONE,
    IN_MLP_DOWN_WEIGHT_EXPERT_FIFTY_ONE,
    IN_MLP_GATEUP_WEIGHT_EXPERT_FIFTY_TWO,
    IN_MLP_DOWN_WEIGHT_EXPERT_FIFTY_TWO,
    IN_MLP_GATEUP_WEIGHT_EXPERT_FIFTY_THREE,
    IN_MLP_DOWN_WEIGHT_EXPERT_FIFTY_THREE,
    IN_MLP_GATEUP_WEIGHT_EXPERT_FIFTY_FOUR,
    IN_MLP_DOWN_WEIGHT_EXPERT_FIFTY_FOUR,
    IN_MLP_GATEUP_WEIGHT_EXPERT_FIFTY_FIVE,
    IN_MLP_DOWN_WEIGHT_EXPERT_FIFTY_FIVE,
    IN_MLP_GATEUP_WEIGHT_EXPERT_FIFTY_SIX,
    IN_MLP_DOWN_WEIGHT_EXPERT_FIFTY_SIX,
    IN_MLP_GATEUP_WEIGHT_EXPERT_FIFTY_SEVEN,
    IN_MLP_DOWN_WEIGHT_EXPERT_FIFTY_SEVEN,
    IN_MLP_GATEUP_WEIGHT_EXPERT_FIFTY_EIGHT,
    IN_MLP_DOWN_WEIGHT_EXPERT_FIFTY_EIGHT,
    IN_MLP_GATEUP_WEIGHT_EXPERT_FIFTY_NINEE,
    IN_MLP_DOWN_WEIGHT_EXPERT_FIFTY_NINE,
    IN_MLP_GATEUP_WEIGHT_EXPERT_SIXTY,
    IN_MLP_DOWN_WEIGHT_EXPERT_SIXTY,
    IN_MLP_GATEUP_WEIGHT_EXPERT_SIXTY_ONE,
    IN_MLP_DOWN_WEIGHT_EXPERT_SIXTY_ONE,
    IN_MLP_GATEUP_WEIGHT_EXPERT_SIXTY_TWO,
    IN_MLP_DOWN_WEIGHT_EXPERT_SIXTY_TWO,
    IN_MLP_GATEUP_WEIGHT_EXPERT_SIXTY_THREE,
    IN_MLP_DOWN_WEIGHT_EXPERT_SIXTY_THREE,

    IN_ONE_HOT_ONE,
    IN_ONE_HOT_ZERO,
    IN_FINAL_HIDDEN_STATE,
    IN_COS_TABLE,                       // shape: FA: [batchSize * seqLen, hiddenSizePerAttentionHead] PA: [seqLen, hiddenSizePerAttentionHead]
    IN_SIN_TABLE,                       // shape: FA: [batchSize * seqLen, hiddenSizePerAttentionHead] PA: [seqLen, hiddenSizePerAttentionHead]
    IN_ATTENTION_MASK,                  // shape: FA: [batchSize, maxPositionEmbeddings, maxPositionEmbeddings] PA: [seqLen, seqLen]
    IN_K_CACHE,                         // shape: FA: [batchSize, maxPositionEmbeddings, numKeyValueHeadsPerRank * hiddenSizePerAttentionHead] PA: [2622, hiddenSizePerAttentionHead, numAttentionHeadsPerRank, hiddenSizePerAttentionHead]
    IN_V_CACHE,                         // shape: FA: [batchSize, maxPositionEmbeddings, numKeyValueHeadsPerRank * hiddenSizePerAttentionHead] PA: [2622, hiddenSizePerAttentionHead, numAttentionHeadsPerRank, hiddenSizePerAttentionHead]
    IN_SEQ_LEN,                         // shape: [batchSize]
    IN_PLACE_HOLDER,                    // shape: [1]
    IN_TOKEN_OFFSET,                    // shape: [batchSize]; FA所需参数
    IN_LAYER_ID,                        // shape: [1]; FA所需参数
    IN_BLOCK_TABLES,                    // shape: [seqLen, seqLen]; PA所需参数
    IN_SLOTS,                           // shape: [seqLen]; PA所需参数

    OUT_DECODER_LAYER,                  // shape: FA: [batchSize, seqLen, maxPositionEmbeddings] PA: [seqLen, hiddenSize]

    INTERMEDIATE_ATTENTION_OUT,         // shape: PA: [seqLen, hiddenSize]
    INTERMEDIATE_RESIDUAL_ADD_OUT,      // shape: PA: [seqLen, hiddenSize]
    INTERMEDIATE_MLP_OUT,               // shape: PA: [seqLen, hiddenSize]

    INTERMIDATE_SELFATTENTION_NORM_OUT,
    INTERMIDATE_MOE_OUT_ALL,
    INTERMIDATE_MOE_OUT,
    INTERMIDATE_HIDDEN_STATE_SHARED_EXPERTS,
};

atb::Status DecoderLayer(const DecoderLayerParam &param, atb::Operation **operation);

class DecoderLayerBinder : public HostTensorBinder {
public:
    DecoderLayerBinder();
    virtual ~DecoderLayerBinder();

private:
    std::vector<int> tokenOffset_;
    std::vector<int> seqLen_;
    int32_t layerId_ = 0;
};

}  // namespace deepseekDense
}  // namespace atb_speed
#endif