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
#include "pytorch/adapter/utils/utils.h"
#include "atb_speed/log.h"
#include "atb_speed/utils/singleton.h"
#include "atb_speed/utils/config.h"

namespace atb_speed {
thread_local std::shared_ptr<atb::Context> localContext;

std::shared_ptr<atb::Context> ContextFactory::GetAtbContext()
{
    if (localContext) {
	ATB_LOG(INFO) << "ContextFactory return localContext";
	return localContext;
    }
    ATB_LOG(INFO) << "ContextFactory create AtbContext";
    uint64_t tilingBufferNumMask = 0x0000000000000070;
    uint64_t tilingBufferSizeMask = 0x0000000000000002;
    uint64_t flag = tilingBufferNumMask | tilingBufferSizeMask;
    if (atb_speed::GetSingleton<atb_speed::Config>().IsUseTilingCopyStream()) {
	flag = flag | atb::MULTI_STREAM_MASK;
    }
    atb::Context *contextPtr = nullptr;
    atb::CreateContext(&contextPtr, flag);
    localContext.reset(contextPtr);
    localContext->SetExecuteStream(Utils::GetCurrentStream());
    return localContext;
}

void ContextFactory::FreeAtbContext()
{
    ATB_LOG(INFO) << "ContextFactory FreeAtbContext start.";
    if (!localContext) {
        return;
    }
    
    ATB_LOG(INFO) << "ContextFactory localContext use_count: " << localContext.use_count();
    if (localContext.use_count() != 1) {
        return;
    }
    ATB_LOG(INFO) << "ContextFactory localContext reset.";
    localContext.reset();
}
}