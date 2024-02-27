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
#include "atb_speed/utils/hosttensor_binder_factory.h"

#include "atb_speed/log.h"

namespace atb_speed {
bool HosttensorBinderFactory::Register(const std::string &binderName, CreateBinderFuncPtr createBinder)
{
    auto it = HosttensorBinderFactory::GetRegistryMap().find(binderName);
    if (it != HosttensorBinderFactory::GetRegistryMap().end()) {
        ATB_LOG(WARN) << binderName << " hosttensor binder already exists, but the duplication doesn't matter.";
        return false;
    }
    HosttensorBinderFactory::GetRegistryMap()[binderName] = createBinder;
    return true;
}

atb_speed::HostTensorBinder *HosttensorBinderFactory::CreateInstance(const std::string &binderName)
{
    auto it = HosttensorBinderFactory::GetRegistryMap().find(binderName);
    if (it != HosttensorBinderFactory::GetRegistryMap().end()) {
        ATB_LOG(INFO) << "find hosttensor binder: " << binderName;
        return it->second();
    }
    ATB_LOG(WARN) << "HosttensorBinderName: " << binderName << " not find in model factory map";
    return nullptr;
}

std::unordered_map<std::string, CreateBinderFuncPtr> &HosttensorBinderFactory::GetRegistryMap()
{
    static std::unordered_map<std::string, CreateBinderFuncPtr> binderRegistryMap;
    return binderRegistryMap;
}
} // namespace atb_speed
