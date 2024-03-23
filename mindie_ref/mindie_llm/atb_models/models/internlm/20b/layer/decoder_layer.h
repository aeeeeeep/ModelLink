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
#ifndef ATB_SPEED_MODELS_INTERNLM_20B_PARALLEL_DECODER_LAYER_H
#define ATB_SPEED_MODELS_INTERNLM_20B_PARALLEL_DECODER_LAYER_H

#include <vector>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtype-limits"
#include "nlohmann/json.hpp"
#pragma GCC diagnostic pop

#include "atb/atb_infer.h"
#include "atb_speed/base/hosttensor_binder.h"
#include "atb_speed/log.h"

namespace atb_speed {
namespace internlm_20b_parallel {
struct DecoderLayerParam {
    bool isFA = true;
    bool isPrefill = false;
    bool isBF16 = false;
    bool isPack = true;
    bool supportSwiGLU = false;
    bool supportLcoc = false;
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
    std::vector<int> packQuantType = {};
    
    // 记录[Q,K,V,Out,Gate,Up,Down]的linear类型标识符。 0：浮点型 1: 量化型 -1：与前一个矩阵合并 
    std::vector<int> linearQuantType = {};
};

enum DecoderLayerTensorIdx : uint32_t {
    IN_HIDDEN_STATES = 0,               
    IN_INPUT_NORM_WEIGHT,               
    IN_INPUT_NORM_BIAS,
    IN_INPUT_NORM_NEW_WEIGHT,
    IN_INPUT_NORM_NEW_BIAS,

    IN_QKV_WEIGHT_0,                                     
    IN_QKV_BIAS_0,                  
    IN_QKV_DESCALE_0,                   
    IN_QKV_OFFSET_0,                    
    IN_QKV_SCALE_0, 
    IN_QKV_COMPRESS_IDX_0,
    
    IN_QKV_WEIGHT_1,                    
    IN_QKV_BIAS_1,                  
    IN_QKV_DESCALE_1,                   
    IN_QKV_OFFSET_1,                   
    IN_QKV_SCALE_1,
    IN_QKV_COMPRESS_IDX_1,
    
    IN_QKV_WEIGHT_2,                   
    IN_QKV_BIAS_2,                
    IN_QKV_DESCALE_2,          
    IN_QKV_OFFSET_2,             
    IN_QKV_SCALE_2,
    IN_QKV_COMPRESS_IDX_2,
    
    IN_ATTENTION_OUT_WEIGHT,            
    IN_ATTENTION_OUT_BIAS,       
    IN_ATTENTION_OUT_DESCALE,           
    IN_ATTENTION_OUT_OFFSET,           
    IN_ATTENTION_OUT_SCALE,
    IN_ATTENTION_OUT_COMPRESS_IDX,    

    IN_ATTENTION_NORM_WEIGHT,           
    IN_ATTENTION_NORM_BIAS,
    IN_ATTENTION_NORM_NEW_WEIGHT,
    IN_ATTENTION_NORM_NEW_BIAS,

    IN_MLP_WEIGHT_0,                    
    IN_MLP_BIAS_0,                  
    IN_MLP_DESCALE_0,                
    IN_MLP_OFFSET_0,                    
    IN_MLP_SCALE_0,
    IN_MLP_COMPRESS_IDX_0,

    IN_MLP_WEIGHT_1,                  
    IN_MLP_BIAS_1,                
    IN_MLP_DESCALE_1,                 
    IN_MLP_OFFSET_1,                 
    IN_MLP_SCALE_1,
    IN_MLP_COMPRESS_IDX_1,                    

    IN_MLP_DOWN_WEIGHT,              
    IN_MLP_DOWN_BIAS,               
    IN_MLP_DOWN_DESCALE,                
    IN_MLP_DOWN_OFFSET,                
    IN_MLP_DOWN_SCALE,
    IN_MLP_DOWN_COMPRESS_IDX,               

    IN_COS_TABLE,                  
    IN_SIN_TABLE,                      
    IN_ATTENTION_MASK,                  
    IN_K_CACHE,                      
    IN_V_CACHE,                        
    IN_SEQ_LEN,                     
    IN_PLACE_HOLDER,                  
    IN_TOKEN_OFFSET,                    
    IN_LAYER_ID,                       
    IN_BLOCK_TABLES,                   
    IN_SLOTS,                          

    OUT_DECODER_LAYER,                

    INTERMEDIATE_ATTENTION_OUT,       
    INTERMEDIATE_RESIDUAL_ADD_OUT,  
    INTERMEDIATE_MLP_OUT,               
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

}  // namespace internlm_20b_parallel
}  // namespace atb_speed
#endif