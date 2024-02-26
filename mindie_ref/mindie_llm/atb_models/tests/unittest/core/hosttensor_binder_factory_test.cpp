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
#include "atb_speed/base/model.h"
#include "atb_speed/utils/hosttensor_binder_factory.h"
#include "chatglm2/6b/layer/paged_attention_layer.h"

#include <gtest/gtest.h>
#include <string>
#include <vector>

using namespace atb_speed;

TEST(HosttensorBinderFactory, RegisterShouldReturnTrueWhenGivenUniqueBinderName)
{
    bool firstTimeRegister = HosttensorBinderFactory::Register("FlashAttentionHostBinder_1", []() {
        return new atb_speed::chatglm2_6b::FlashAttentionHostBinder();
    });
    ASSERT_EQ(firstTimeRegister, true);

    bool duplicateRegister = HosttensorBinderFactory::Register("FlashAttentionHostBinder_1", []() {
        return new atb_speed::chatglm2_6b::FlashAttentionHostBinder();
    });
    ASSERT_EQ(duplicateRegister, false);
}

TEST(HosttensorBinderFactory, CreateHosttensorBinderByClassConstructorWouldNotGetNullptrWhenGivenCorrectParam)
{
    atb_speed::HostTensorBinder *binder_ = new atb_speed::chatglm2_6b::FlashAttentionHostBinder();
    ASSERT_NE(binder_, nullptr);
}

TEST(HosttensorBinderFactory, CreateModelByCreateInstanceWouldNotGetNullptrWhenGivenCorrectBinderName)
{
    atb_speed::HostTensorBinder *binder_ = HosttensorBinderFactory::CreateInstance("FlashAttentionHostBinder_1");
    ASSERT_NE(binder_, nullptr);
}