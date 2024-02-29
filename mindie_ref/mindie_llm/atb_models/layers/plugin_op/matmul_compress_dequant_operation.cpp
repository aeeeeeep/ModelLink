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
#include "matmul_compress_dequant_operation.h"
#include <iostream>
#include <sstream>
#include <unistd.h>
#include <cstring>
#include <syscall.h>
#include <securec.h>

#include "acl/acl.h"
#include "aclnnop/aclnn_matmul_compress_dequant.h"
#include "atb_speed/log.h"

namespace atb_speed {
namespace common {
const size_t HOST_TILING_BUFFER_DEFAULT_SIZE = 10240;
constexpr uint32_t MAX_PROFILING_FUNC_NAME = 2;
static constexpr int32_t DIM_1 = 1;
static constexpr int32_t DIM_2 = 2;
static constexpr int32_t DIM_3 = 3;
static constexpr int32_t DIM_4 = 4;
static constexpr int32_t DIM_7 = 7;

MatMulCompressDequantOperation::MatMulCompressDequantOperation(const std::string &name) : name_(name) {}

MatMulCompressDequantOperation::~MatMulCompressDequantOperation() {}

std::string MatMulCompressDequantOperation::GetName() const
{
    return name_;
}

atb::Status MatMulCompressDequantOperation::InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
    atb::SVector<atb::TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).dtype = aclDataType::ACL_FLOAT16;

    if (inTensorDescs.at(0).shape.dimNum == 3) {
        outTensorDescs.at(0).shape.dimNum = DIM_3;
        outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
        outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1];
        outTensorDescs.at(0).shape.dims[2] = inTensorDescs.at(3).shape.dims[0];
    } else {
        outTensorDescs.at(0).shape.dimNum = DIM_2;
        outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
        outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(3).shape.dims[0];
    }
    return 0;
}

uint32_t MatMulCompressDequantOperation::GetInputNum() const
{
    return DIM_7;
}

uint32_t MatMulCompressDequantOperation::GetOutputNum() const
{
    return DIM_1;
}

aclTensor *CreateTensor(const atb::VariantPack &variantPack, int64_t index, bool is_input)
{
    atb::Tensor tensor;
    if (is_input) {
        tensor = variantPack.inTensors.at(index);
    } else {
        tensor = variantPack.outTensors.at(index);
    }

    atb::TensorDesc tensorDesc = tensor.desc;
    const int64_t *dims = tensorDesc.shape.dims;
    const uint64_t dimNum = tensorDesc.shape.dimNum;

    aclDataType dtype = tensorDesc.dtype;
    if (index == DIM_4) {
        dtype = aclDataType::ACL_UINT64;
        tensor.desc.dtype = aclDataType::ACL_UINT64;
    }

    if (!is_input) {
        dtype = aclDataType::ACL_FLOAT16;
        tensor.desc.dtype = aclDataType::ACL_FLOAT16;
    }

    aclFormat format = tensorDesc.format;

    if (dimNum == DIM_3) {
        int newDimNum = 2;
        int64_t newDims[2] = {dims[0] * dims[1], dims[2]};
        atb::SVector<int64_t> strides(dimNum, 1);
        for (int64_t i = newDimNum - 2; i >= 0; i--) {
            strides[i] = newDims[i + 1] * strides[i + 1];
        }
        auto ret = aclCreateTensor(newDims, newDimNum, dtype, strides.data(), 0, format, newDims, newDimNum,
            tensor.deviceData);
        return ret;
    } else {
        atb::SVector<int64_t> strides(dimNum, 1);
        for (int64_t i = dimNum - 2; i >= 0; i--) {
            strides[i] = dims[i + 1] * strides[i + 1];
        }
        auto ret = aclCreateTensor(dims, dimNum, dtype, strides.data(), 0, format, dims, dimNum, tensor.deviceData);
        return ret;
    }
}

atb::Status MatMulCompressDequantOperation::Setup(uint64_t &workspaceSize)
{
    workspaceSize = 0;
    return 0;
}

atb::Status MatMulCompressDequantOperation::Execute(const atb::VariantPack &variantPack, uint8_t *workspace,
    uint64_t workspaceSize, atb::Context *context)
{
    if (x1_ == nullptr) {
        x1_ = CreateTensor(variantPack, 0, true);
    } else {
        aclDestroyTensor(x1_);
        x1_ = CreateTensor(variantPack, 0, true);
    }

    if (x2_ == nullptr) {
        x2_ = CreateTensor(variantPack, 1, true);
    }

    if (compressIndex_ == nullptr) {
        compressIndex_ = CreateTensor(variantPack, 2, true);
    }

    if (bias_ == nullptr) {
        bias_ = CreateTensor(variantPack, 3, true);
    }

    if (deqScale_ == nullptr) {
        deqScale_ = CreateTensor(variantPack, 4, true);
    }

    int *offsetX = static_cast<int *>(variantPack.inTensors.at(5).hostData);
    int64_t *compressInfoData = static_cast<int64_t *>(variantPack.inTensors.at(6).hostData);

    aclIntArray *compressInfo = aclCreateIntArray(compressInfoData, aclDataType::ACL_INT64);

    aclTensor *output = CreateTensor(variantPack, 0, false);

    int ret1 = aclnnMatmulCompressDequantGetWorkspaceSize(x1_, x2_, compressIndex_, bias_, deqScale_, nullptr, *offsetX,
        compressInfo, output, &workspaceSize, &m_executor);

    ATB_LOG(INFO) << "start to execute aclnnMatmulCompressDequant.., ret1: " << ret1;

    int ret2 = aclnnMatmulCompressDequant(workspace, workspaceSize, m_executor, context->GetExecuteStream());

    ATB_LOG(INFO) << "ret2: " << ret2;

    return 0;
}

aclError MatMulCompressDequantOperation::CheckAcl(aclError ret)
{
    if (ret != ACL_ERROR_NONE) {
        std::cerr << __FILE__ << ":" << __LINE__ << "aclError:" << ret << std::endl;
    }

    return ret;
}
} // namespace common
} // namespace atb_speed
