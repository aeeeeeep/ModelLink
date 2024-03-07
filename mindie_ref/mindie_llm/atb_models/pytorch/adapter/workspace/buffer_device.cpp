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
#include "buffer_device.h"
#include <acl/acl.h>
#include <atb_speed/utils/timer.h>
#include <atb_speed/utils/singleton.h>
#include <atb/types.h>
#include "atb_speed/log.h"
#include "atb_speed/utils/statistic.h"
#include "pytorch/adapter/utils/utils.h"

namespace atb_speed {
constexpr int KB_1 = 1024;
constexpr int MB_1 = 1024 * 1024;
constexpr int GB_1 = 1024 * 1024 * 1024;

BufferDevice::BufferDevice(uint64_t bufferSize) : bufferSize_(bufferSize)
{
    ATB_LOG(INFO) << "BufferDevice::BufferDevice called, bufferSize:" << bufferSize;
    bufferSize_ = bufferSize;
    if (bufferSize_ > 0) {
        ATB_LOG(FATAL) << "BufferDevice::GetBuffer bufferSize:" << bufferSize_;
        atTensor_ = CreateAtTensor(bufferSize_);
        buffer_ = atTensor_.data_ptr();
    }
}

BufferDevice::~BufferDevice() {}

void *BufferDevice::GetBuffer(uint64_t bufferSize)
{
    if (bufferSize <= bufferSize_) {
        ATB_LOG(INFO) << "BufferDevice::GetBuffer bufferSize:" << bufferSize << "<= bufferSize_:" << bufferSize_
                        << ", not new device mem.";
        return atTensor_.data_ptr();
    }
    
    torch::Tensor newAtTensor = CreateAtTensor(bufferSize);
    bufferSize_ = newAtTensor.numel();
    atTensor_ = newAtTensor;
    ATB_LOG(INFO) << "BufferDevice::GetBuffer new bufferSize:" << bufferSize;
    buffer_ = atTensor_.data_ptr();
    return atTensor_.data_ptr();
}

torch::Tensor BufferDevice::CreateAtTensor(uint64_t bufferSize)
{
    atb::TensorDesc tensorDesc;
    tensorDesc.dtype = ACL_UINT8;
    tensorDesc.format = ACL_FORMAT_ND;
    if (bufferSize <= KB_1) {
        tensorDesc.shape.dimNum = 1;
        tensorDesc.shape.dims[0] = bufferSize;
    } else if (bufferSize <= MB_1) {
        tensorDesc.shape.dimNum = 2;
        tensorDesc.shape.dims[0] = KB_1;
        tensorDesc.shape.dims[1] = bufferSize / KB_1 + 1;
    } else if (bufferSize <= GB_1) {
        tensorDesc.shape.dimNum = 3;
        tensorDesc.shape.dims[0] = KB_1;
        tensorDesc.shape.dims[1] = KB_1;     
        tensorDesc.shape.dims[2] = bufferSize / MB_1 + 1;        
    } else {
        tensorDesc.shape.dimNum = 4;
        tensorDesc.shape.dims[0] = KB_1;
        tensorDesc.shape.dims[1] = KB_1;  
        tensorDesc.shape.dims[2] = KB_1;    
        tensorDesc.shape.dims[3] = bufferSize / GB_1 + 1;   
    }
    
    return Utils::CreateAtTensorFromTensorDesc(tensorDesc);
}
} // namespace atb_speed