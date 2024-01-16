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
#include "chatglm6b/layer/chatglm6blayer_decoder_flashattention_operation.h"

atb_speed::HostTensorBinder *CreateHostTensorBinder(const std::string &opName)
{
    if (opName == "ChatGlm6BLayerDecoderFlashAttentionOperation") {
        return new atb_speed::ChatGlm6BLayerDecoderFlashAttentionBinder();
    }
    return nullptr;
}