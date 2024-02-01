
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
#include "adapter.h"
#include <iostream>
#include <sys/stat.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/DeviceUtils.h>
#include <torch_npu/csrc/framework/OpCommand.h>

using namespace std;

at::Tensor CreateAtTensorFromTensorDesc(const atb::TensorDesc &tensorDesc)
{
    static std::map<aclDataType, at::ScalarType> dtypeMap = {
        {ACL_BOOL, at::ScalarType::Bool},   {ACL_UINT8, at::ScalarType::Byte},
        {ACL_INT8, at::ScalarType::Char},   {ACL_FLOAT16, at::ScalarType::Half},
        {ACL_FLOAT, at::ScalarType::Float}, {ACL_INT32, at::ScalarType::Int},
        {ACL_INT64, at::ScalarType::Long},
    };
    at::TensorOptions options = at::TensorOptions(torch_npu::utils::get_npu_device_type());
    auto it = dtypeMap.find(tensorDesc.dtype);
    TORCH_CHECK(it != dtypeMap.end(), "not support dtype:");
    options = options.dtype(it->second);
    options = options.layout(torch::kStrided).requires_grad(false);
    at::Tensor newTensor = at::zeros(at::IntArrayRef(tensorDesc.shape.dims, tensorDesc.shape.dimNum), options);
    if (!newTensor.is_contiguous()) {
        newTensor = newTensor.contiguous();
    }

    return newTensor;
}

atb::Tensor AtTensor2Tensor(const at::Tensor atTensor)
{
    static std::map<at::ScalarType, aclDataType> dtypeMap = {
        {at::ScalarType::Bool, ACL_BOOL},   {at::ScalarType::Byte, ACL_UINT8},
        {at::ScalarType::Char, ACL_INT8},   {at::ScalarType::Half, ACL_FLOAT16},
        {at::ScalarType::Float, ACL_FLOAT}, {at::ScalarType::Int, ACL_INT32},
        {at::ScalarType::Long, ACL_INT64},
    };

    TORCH_CHECK(atTensor.is_contiguous(), "atTensor is not contiguous");
    atb::Tensor tensor;
    tensor.desc.format = ACL_FORMAT_ND;
    tensor.deviceData = atTensor.data_ptr();

    tensor.desc.shape.dimNum = atTensor.sizes().size();
    for (uint64_t i = 0; i < atTensor.sizes().size(); i++) {
        tensor.desc.shape.dims[i] = atTensor.sizes()[i];
    }

    auto it = dtypeMap.find(atTensor.scalar_type());
    TORCH_CHECK(it != dtypeMap.end(), "not support dtype:");
    tensor.desc.dtype = it->second;

    tensor.dataSize = atb::Utils::GetTensorSize(tensor);

    return tensor;
}

at::Tensor FormatTrans(const at::Tensor &at_tensor)
{
    if (at_tensor.defined()) {
        TORCH_CHECK(torch_npu::utils::is_npu(at_tensor), "only npu tensor is supported");
        return at_npu::native::NPUNativeFunctions::npu_format_cast(at_tensor, ACL_FORMAT_ND);
    }
    return at_tensor;
}

atb::Tensor Input(const at::Tensor &tensor)
{
    at::Tensor newTensor = FormatTrans(tensor);
    if (!newTensor.is_contiguous()) {
        newTensor = newTensor.contiguous();
    }
    return AtTensor2Tensor(newTensor);
}

atb::Tensor Input(const c10::optional<at::Tensor> &tensor)
{
    if (!tensor.has_value()) {
        return atb::Tensor();
    }
    return Input(tensor.value());
}


void BuildVariantPack(std::vector<atb::Tensor> inTensors, std::vector<at::Tensor> &outTensors, atb::VariantPack &variantPack, atb::Operation *operation)
{
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    for (size_t i = 0; i < inTensors.size(); ++i) {
        atb::Tensor inTensor = inTensors.at(i);
        inTensorDescs.push_back(inTensor.desc);
        variantPack.inTensors.push_back(inTensors[i]);
    }
    atb::Status status = operation->InferShape(inTensorDescs, outTensorDescs);
    TORCH_CHECK(status == 0, "infershape failed!");

    for (size_t i = 0; i < outTensorDescs.size(); ++i) {
        at::Tensor newTensor = CreateAtTensorFromTensorDesc(outTensorDescs.at(i));
        outTensors.push_back(newTensor);
        atb::Tensor atbOutTensor = AtTensor2Tensor(newTensor);
        atbOutTensor.desc.format = ACL_FORMAT_ND;
        variantPack.outTensors.push_back(atbOutTensor);
    }
}

void RunAtbOps(atb::VariantPack &variantPack, const char* name, atb::Operation *operation)
{
    atb::Context *context = GetContext();
    TORCH_CHECK(context, "execute failed");

    uint8_t *workspace = nullptr;
    uint64_t workspaceSize = 0;
    atb::Status status = operation->Setup(variantPack, workspaceSize, context);
    TORCH_CHECK(status == 0, "setup failed!");
    if (workspaceSize > 0) {
        at::TensorOptions options = at::TensorOptions(torch_npu::utils::get_npu_device_type());
        auto workspaceTensor = at::empty({workspaceSize}, options.dtype(at::kByte));
        workspace = (uint8_t *)workspaceTensor.storage().data();
    }

    auto aclCall = [operation, variantPack, workspace, workspaceSize, context]() -> int {
        atb::Status status = operation->Execute(variantPack, workspace, workspaceSize, context);
        TORCH_CHECK(status == 0, "execute failed");
        atb::DestroyOperation(operation);
        return 0;
    };
    at_npu::native::OpCommand cmd;
    cmd.Name(name);
    cmd.SetCustomHandler(aclCall);
    cmd.Run();
}