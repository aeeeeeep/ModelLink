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
#ifndef ATB_SPEED_CONTEXT_BUFFER_DEVICE_H
#define ATB_SPEED_CONTEXT_BUFFER_DEVICE_H
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <torch/torch.h>
#pragma GCC diagnostic pop
#include "buffer_base.h"

namespace atb_speed {
class BufferDevice : public BufferBase {
public:
    explicit BufferDevice(uint64_t bufferSize);
    virtual ~BufferDevice();
    void *GetBuffer(uint64_t bufferSize) override;
private:
    torch::Tensor CreateAtTensor(uint64_t bufferSize);

private:
    void *buffer_ = nullptr;
    uint64_t bufferSize_ = 0;
    torch::Tensor atTensor_;
};
} // namespace atb_speed
#endif