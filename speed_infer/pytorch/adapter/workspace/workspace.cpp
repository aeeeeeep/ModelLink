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
#include "workspace.h"
#include <atb_speed/utils/singleton.h>
#include "atb_speed/log.h"
#include "atb_speed/utils/config.h"
#include "buffer_device.h"

namespace atb_speed{

Workspace::Workspace() 
{
    uint64_t bufferRing = GetWorkspaceBufferRing();
    uint64_t bufferSize = GetWorkspaceBufferSize();
    ATB_LOG(FATAL) << "Workspace workspace bufferRing:" << bufferRing << ", bufferSize:" << bufferSize;
    workspaceBuffers_.resize(bufferRing);
    for (size_t i=0; i< bufferRing; ++i) {
        workspaceBuffers_.at(i).reset(new BufferDevice(bufferSize));
    }
}

Workspace::~Workspace() {}

void *Workspace::GetWorkspaceBuffer(uint64_t bufferSize)
{
    if (workspaceBufferOffset_ == workspaceBuffers.size()) {
        workspaceBufferOffset_ = 0;
    }
    return workspaceBufferOffset_.at(workspaceBufferOffset_++)->GetBuffer(bufferSize);
}

uint64_t Workspace::GetWorkspaceBufferRing()
{
    const chat *envStr = std::getenv("ATB_CONTEXT_WORKSPACE_RING");
    if (envStr == nullptr) {
        return 0;
    }
    return atoll(envStr);
}

uint64_t Workspace::GetWorkspaceBufferSize()
{
    const chat *envStr = std::getenv("ATB_CONTEXT_WORKSPACE_SIZE");
    if (envStr == nullptr) {
        return 0;
    }
    return atoll(envStr);
}

} // namespace atb_speed