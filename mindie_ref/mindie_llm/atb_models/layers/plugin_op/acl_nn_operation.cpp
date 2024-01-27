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
#include "acl_nn_operation.h"
#include "atb_speed/log.h"
#include "atb_speed/utils/timer.h"
#include "atb_speed/utils/singleton.h"
#include "atb_speed/utils/statistic.h"

namespace atb_speed {
namespace common {
const int ACLNN_WORKSPACE_SIZE = 800 * 1024 * 1024;

int AclNnTensor::CreateTensor(const std::string &opName)
{
    atb::SVector<int64_t> tmpStrides(atbTensor.desc.shape.dimNum, 1);
    for (int64_t i = atbTensor.desc.shape.dimNum - 2; i >= 0; i--) {
        tmpStrides[i] = atbTensor.desc.shape.dims[i + 1] * tmpStrides[i + 1];
    }
    strides = tmpStrides;

    ATB_LOG(INFO) << opName << " aclCreateTensor start, tensor.deviceData:" << atbTensor.deviceData;
    tensor = aclCreateTensor(atbTensor.desc.shape.dims, atbTensor.desc.shape.dimNum, atbTensor.desc.dtype,
                             strides.data(), 0, atbTensor.desc.format, atbTensor.desc.shape.dims,
                             atbTensor.desc.shape.dimNum, atbTensor.deviceData);
    if (tensor) {
        ATB_LOG(INFO) << opName << " aclCreateTensor success, tensor:" << tensor;
        return atb::NO_ERROR;
    }

    ATB_LOG(ERROR) << opName << " aclCreateTensor fail";
    return atb::ERROR_INTERNAL_ERROR;
}

int AclNnTensor::InitTensor(const std::string &opName)
{
    if (!tensor) {
        ATB_LOG(ERROR) << opName << " acl tensor is null, not call aclInitTensor";
        return atb::ERROR_INTERNAL_ERROR;
    }

    ATB_LOG(INFO) << opName << " aclInitTensor start, tensor:" << tensor
                  << ", tensor.deviceData:" << atbTensor.deviceData;
    int ret = aclInitTensor(tensor, atbTensor.desc.shape.dims, atbTensor.desc.shape.dimNum,
                            atbTensor.desc.dtype, strides.data(), 0, atbTensor.desc.format, atbTensor.desc.shape.dims,
                            atbTensor.desc.shape.dimNum, atbTensor.deviceData);
    ATB_LOG_IF(ret != 0, ERROR) << opName << " aclInitTensor fail, error:" << ret;
    return ret;
}

void AclNnTask::Destory()
{
    for (size_t i = 0; i < aclInTensors.size(); ++i) {
        aclDestroyTensor(aclInTensors[i].tensor);
    }
    aclInTensors.clear();

    for (size_t i = 0; i < aclOutTensors.size(); ++i) {
        aclDestroyTensor(aclOutTensors[i].tensor);
    }
    aclOutTensors.clear();

    for (size_t i = 0; i < aclInIntArrays.size(); ++i) {
        aclDestroyIntArray(aclInIntArrays[i]);
    }
    aclInIntArrays.clear();
}

AclNnOperation::AclNnOperation(const std::string &opName) : opName_(opName) {}

AclNnOperation::~AclNnOperation() {}

std::string AclNnOperation::GetName() const { return opName_;}

atb::Status AclNnOperation::Setup(const atb::VariantPack &variantPack, uint64_t &workspaceSize, atb::Context *context)
{
    GetSingleton<Statistic>().pluginOpCallCount++;

    Timer setupTimer;
    ATB_LOG(INFO) << opName_ << " setup start";

    workspaceSize = ACLNN_WORKSPACE_SIZE;

    ATB_LOG(INFO) << opName_ << " setup end";
    GetSingleton<Statistic>().pluginOpSetupTime += setupTimer.ElapsedMicroSecond();
    return atb::NO_ERROR;
}

atb::Status AclNnOperation::Execute(const atb::VariantPack &variantPack, uint8_t *workspace, uint64_t workspaceSize,
                                    atb::Context *context)
{
    Timer executeTimer;
    ATB_LOG(INFO) << opName_ << " execute start";
    if (!context) {
        ATB_LOG(ERROR) << opName_ << " execute fail, context param is null";
        return atb::ERROR_INVALID_PARAM;
    }

    aclrtStream stream = context->GetExecuteStream();
    if (!stream) {
        ATB_LOG(ERROR) << opName_ << " execute fail, execute stream in context is null";
        return atb::ERROR_INVALID_PARAM;
    }

    int ret = CreateAclTensors(variantPack, aclNnTask_);
    if (ret != 0) {
        ATB_LOG(ERROR) << opName_ << " call acl create tensor fail, error:" << ret;
        return atb::ERROR_CANN_ERROR;
    }
    for (size_t i = 0; i < aclNnTask_.aclInTensors.size(); ++i) {
        int ret = aclNnTask_.aclInTensors.at(i).CreateTensor(opName_);
        if (ret != 0) {
            return atb::ERROR_INTERNAL_ERROR;
        }
    }
    for (size_t i = 0; i < aclNnTask_.aclOutTensors.size(); ++i) {
        int ret = aclNnTask_.aclOutTensors.at(i).CreateTensor(opName_);
        if (ret != 0) {
            return atb::ERROR_INTERNAL_ERROR;
        }
    }

    Timer executeGetWorkspaceSizeTimer;
    int ret2 = CallAclGetWorkspace(variantPack, aclNnTask_, workspaceSize);
    if (ret2 != 0) {
        ATB_LOG(ERROR) << opName_ << " call acl get workspace fail, error:" << ret;
        return atb::ERROR_CANN_ERROR;
    }
    GetSingleton<Statistic>().pluginOpExecuteGetWorkspaceTime += executeGetWorkspaceSizeTimer.ElapsedMicroSecond();

    Timer executeLaunchTimer;
    ret = CallAclExecute(workspace, workspaceSize, aclNnTask_.aclExecutor, stream);
    if (ret != 0) {
        ATB_LOG(ERROR) << opName_ << " call acl execute fail, error:" << ret;
        aclNnTask_.Destory();
        return atb::ERROR_CANN_ERROR;
    }
    GetSingleton<Statistic>().pluginOpExecuteLaunchTime += executeLaunchTimer.ElapsedMicroSecond();

    aclNnTask_.Destory();

    ATB_LOG(INFO) << opName_ << " execute end";
    GetSingleton<Statistic>().pluginOpExecuteTime += executeTimer.ElapsedMicroSecond();

    return atb::NO_ERROR;
}

atb::Status AclNnOperation::UpdateAclTensorDataPtr(const atb::VariantPack &variantPack)
{
    for (size_t i = 0; i < aclNnTask_.aclInTensors.size(); ++i) {
        AclNnTensor &aclNnTensor = aclNnTask_.aclInTensors[i];
        if (aclNnTensor.needUpdateTensorDataPtr) {
            aclNnTensor.atbTensor.deviceData = variantPack.inTensors.at(i).deviceData;
            int ret = aclNnTensor.InitTensor(opName_);
            if (ret != 0) {
                ATB_LOG(ERROR) << opName_ << " call InitTensor fail, error:" << ret;
                return atb::ERROR_CANN_ERROR;
            }
        }
    }

    for (size_t i = 0; i < aclNnTask_.aclOutTensors.size(); ++i) {
        AclNnTensor &aclNnTensor = aclNnTask_.aclOutTensors[i];
        if (aclNnTensor.needUpdateTensorDataPtr) {
            aclNnTensor.atbTensor.deviceData = variantPack.outTensors.at(i).deviceData;
            int ret = aclNnTensor.InitTensor(opName_);
            if (ret != 0) {
                ATB_LOG(ERROR) << opName_ << " call InitTensor fail, error:" << ret;
                return atb::ERROR_CANN_ERROR;
            }
        }
    }

    return atb::NO_ERROR;
}
} // namespace common
} // namespace atb_speed