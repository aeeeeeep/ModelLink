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
#include "atb_speed/base/context_factory.h"
#include <thread>
#include "atb_speed/log.h"
#include "atb_speed/utils/singleton.h"
#include "atb_speed/utils/config.h"

namespace atb_speed {
thread_local std::shared_ptr<atb::Context> g_localContext;

std::shared_ptr<atb::Context> ContextFactory::GetAtbContext(void *stream)
{
    if (g_localContext) {
	ATB_LOG(INFO) << "ContextFactory return localContext";
	return g_localContext;
    }
    ATB_LOG(INFO) << "ContextFactory create atb::Context start";
    atb::Context *context = nullptr;
    atb::Status st = atb::CreateContext(&context);
    ATB_LOG_IF(st !=0, ERROR) << "ContextFactory create atb::Context fail";

    if (context) {
        context->SetExecuteStream(stream);
        if (atb:speed::GetSingleton<atb_speed::Config>().IsUseTilingCopyStream()) {
            ATB_LOG(INFO) << "ContextFactory use tiling copy stream";
            context->SetAsyncTilingCopyStatus(true);
        } else {
            ATB_LOG(INFO) << "ContextFactory not use tiling copy stream";
        }
    }

    g_localContext.reset(context);

    return g_localContext;
}

void ContextFactory::FreeAtbContext()
{
    ATB_LOG(INFO) << "ContextFactory FreeAtbContext start.";
    if (!g_localContext) {
        return;
    }
    
    ATB_LOG(INFO) << "ContextFactory localContext use_count: " << g_localContext.use_count();
    if (g_localContext.use_count() != 1) {
        return;
    }
    ATB_LOG(INFO) << "ContextFactory localContext reset.";
    g_localContext.reset();
}
}