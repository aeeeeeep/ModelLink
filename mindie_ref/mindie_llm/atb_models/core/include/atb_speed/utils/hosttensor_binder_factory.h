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
#ifndef ATB_SPEED_UTILS_MODEL_FACTORY_H
#define ATB_SPEED_UTILS_MODEL_FACTORY_H

#include <functional>
#include <iostream>
#include <string>
#include <unordered_map>

#include "atb_speed/base/hosttensor_binder.h"
#include "atb_speed/log.h"

namespace atb_speed {
using CreateBinderFuncPtr = std::function<atb_speed::HostTensorBinder *()>;

class HosttensorBinderFactory {
public:
    static bool Register(const std::string &binderName, CreateBinderFuncPtr createBinder);
    static atb_speed::HostTensorBinder *CreateInstance(const std::string &binderName);
private:
    static std::unordered_map<std::string, CreateBinderFuncPtr> &GetRegistryMap();
};

#define BINDER_NAMESPACE_STRINGIFY(binderNameSpace) #binderNameSpace
#define REGISTER_BINDER(nameSpace, binderName)                                                               \
        struct Register##_##nameSpace##_##binderName {                                                       \
            inline Register##_##nameSpace##_##binderName()                                                   \
            {                                                                                                \
                ATB_LOG(INFO) << "register " << #nameSpace << "_" << #binderName;                            \
                HosttensorBinderFactory::Register(BINDER_NAMESPACE_STRINGIFY(nameSpace##_##binderName),      \
                    []() { return new binderName(); });                                                      \
            }                                                                                                \
        } static instance_##nameSpace##binderName;
} // namespace atb_speed
#endif