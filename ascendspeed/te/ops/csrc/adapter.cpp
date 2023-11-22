
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
#include "atb/utils.h"

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
    at::Tensor newTensor = at::empty(at::IntArrayRef(tensorDesc.shape.dims, tensorDesc.shape.dimNum), options);
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

TECommand::TECommand() {}

TECommand& TECommand::SetOperation(atb::Operation **operation)
{
    this->operation = *operation;
    return *this;
}

TECommand& TECommand::Name(std::string name)
{
    this->name = name;
    return *this;
}

TECommand& TECommand::Input(const at::Tensor &tensor)
{
    at::Tensor newTensor = FormatTrans(tensor);
    if(!newTensor.is_contiguous()) {
        newTensor = newTensor.contiguous();
    }
    inTensors.push_back(AtTensor2Tensor(newTensor));
    return *this;
}

TECommand& TECommand::Input(const at::Tensor &tensor, bool isNone)
{
    if (isNone) {
        inTensors.push_back(atb::Tensor());
        return *this;
    }
    return Input(tensor);
}

TECommand& TECommand::Input(const c10::optional<at::Tensor> &tensor)
{
    if (!tensor.has_value()) {
        inTensors.push_back(atb::Tensor());
        return *this;
    }
    return Input(tensor.value());
}

void TECommand::Output(std::vector<at::Tensor> &output)
{
    auto variantPack = BuildVariantPack(output);

    uint64_t workspaceSize = 0;
    auto status = operation->Setup(variantPack, workspaceSize);
    TORCH_CHECK(status == 0, "setup failed!");
    TORCH_CHECK(workspaceSize > 0, "get workspace size failed!");

    at::TensorOptions options = at::TensorOptions(torch_npu::utils::get_npu_device_type());
    auto workspaceTensor = at::empty({workspaceSize}, options.dtype(at::kByte));

    int32_t devId = 0;
    aclrtGetDevice(&devId);
    aclrtStream stream = c10_npu::getCurrentNPUStream(devId).stream();
    TORCH_CHECK(stream != nullptr, "get current stream failed");

    atb::Context* context = nullptr;
    status = atb::CreateContext(&context);
    TORCH_CHECK(status == 0, "create context failed!");

    context->SetExecuteStream(stream);

    auto op = this->operation;
    output.clear();
    auto te_call = [&op, variantPack, &workspaceTensor, workspaceSize, &context, &output]() -> int {
        auto api_ret = op->Execute(variantPack, (uint8_t *)workspaceTensor.storage().data(), workspaceSize, context);
        TORCH_CHECK(api_ret == 0, "execute failed");
        // atb::DestroyContext(context);
        return api_ret;
    };
    at_npu::native::OpCommand cmd;
    cmd.Name(this->name);
    cmd.SetCustomHandler(te_call);
    cmd.Run();
}

atb::VariantPack TECommand::BuildVariantPack(std::vector<at::Tensor> &atOutTensors)
{
    atb::VariantPack variantPack;

    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    for (size_t i = 0; i < inTensors.size(); ++i) {
        atb::Tensor inTensor = inTensors.at(i);
        if (inTensor.desc.format == ACL_FORMAT_NCHW) {
            inTensor.desc.format = ACL_FORMAT_ND;
        }
        inTensorDescs.push_back(inTensor.desc);
    }
    atb::Status status = operation->InferShape(inTensorDescs, outTensorDescs);
    TORCH_CHECK(status == 0, "infershape failed!");

    variantPack.inTensors.resize(inTensorDescs.size());
    for (size_t i = 0; i < inTensors.size(); ++i) {
        variantPack.inTensors.at(i) = inTensors.at(i);
        if (variantPack.inTensors.at(i).desc.format == ACL_FORMAT_NCHW) {
            variantPack.inTensors.at(i).desc.format = ACL_FORMAT_ND;
        }
    }

    for (size_t i = 0; i < outTensorDescs.size(); ++i) {
        at::Tensor newTensor = CreateAtTensorFromTensorDesc(outTensorDescs.at(i));
        atOutTensors.push_back(newTensor);
    }

    variantPack.outTensors.resize(atOutTensors.size());
    for (size_t i = 0; i < atOutTensors.size(); ++i) {
        variantPack.outTensors.at(i) = AtTensor2Tensor(atOutTensors.at(i));
        if (variantPack.outTensors.at(i).desc.format == ACL_FORMAT_NCHW) {
            variantPack.outTensors.at(i).desc.format = ACL_FORMAT_ND;
        }
    }
    return variantPack;
}
