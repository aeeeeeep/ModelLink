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
#include "hosttensor_binder_creator.h"

#include "baichuan2/13b/layer/flash_attention_layer.h"
#include "baichuan2/13b/layer/flash_attention_quant_layer.h"
#include "baichuan2/13b/layer/flash_attention_quant_oper_layer.h"
#include "codellama/34b/layer/flash_attention_rope_layer.h"
#include "internlm/20b/layer/flash_attention_quant_layer.h"
#include "internlm/20b/layer/flash_attention_rope_antioutlier_layer.h"
#include "internlm/20b/layer/flash_attention_rope_layer.h"
#include "internlm/7b/layer/flash_attention_rope_layer.h"

atb_speed::HostTensorBinder *CreateHostTensorBinder(const std::string &opName)
{
    if (opName == "baichuan2_13b_flash_attention_layer") {
        return new atb_speed::baichuan2_13b::FlashAttentionLayerBinder();
    } else if (opName == "baichuan2_13b_flash_attention_quant_layer") {
        return new atb_speed::baichuan2_13b::FlashAttentionQuantLayerBinder();
    } else if (opName == "baichuan2_13b_flash_attention_oper_quant_layer") {
        return new atb_speed::baichuan2_13b::FlashAttentionQuantOperLayerBinder();
    } else if (opName == "internlm_7b_flash_attention_rope_layer") {
        return new atb_speed::internlm_7b::FlashAttentionRopeLayerBinder();
    } else if (opName == "internlm_20b_flash_attention_rope_layer") {
        return new atb_speed::internlm_20b::FlashAttentionRopeLayerBinder();
    } else if (opName == "internlm_20b_flash_attention_quant_layer") {
        return new atb_speed::internlm_20b::FlashAttentionQuantLayerBinder();
    } else if (opName == "internlm_20b_flash_attention_rope_antioutlier_layer") {
        return new atb_speed::internlm_20b::FlashAttentionRopeAntiOutlierLayerBinder();
    } else if (opName == "codellama_34b_flash_attention_rope_layer") {
        return new atb_speed::codellama_34b::FlashAttentionRopeLayerBinder();
    }
    return nullptr;
}