// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <string>
#include <vector>

#include "acl/acl.h"
#include "atb/operation.h"
#include "atb/train_op_params.h"
#include "atb/types.h"
#include "atb/utils.h"

#include "inc/adapter.h"
#include "../ops_interface.h"
#include "nlohmann/json.hpp"

static atb::Operation *FastSoftMaxOperationCreate(const string paramString)
{
    nlohmann::json paramJson = nlohmann::json::parse(paramString);
    atb::train::FastSoftMaxParam param;
    param.headNum = paramJson["headNum"].get<int32_t>();
    for (auto item : paramJson["seqLen"]) {
        param.qSeqLen.push_back(item.get<int32_t>());
    }
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}

static atb::Operation *FastSoftMaxGradOperationCreate(const string paramString)
{
    nlohmann::json paramJson = nlohmann::json::parse(paramString);
    atb::train::FastSoftMaxGradParam param;
    param.headNum = paramJson["headNum"].get<int32_t>();
    for (auto item : paramJson["seqLen"]) {
        param.qSeqLen.push_back(item.get<int32_t>());
    }
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}

at::Tensor fastsoftmax(const at::Tensor &dataInput, const std::vector<int32_t> &seqLen, int32_t headNum)
{
    std::string param = "{\"seqLen\": [";
    for (int i = 0; i < seqLen.size() - 1; ++i) {
        param.append(std::to_string(seqLen[i])).append(",");
    }
    param.append(std::to_string(seqLen[seqLen.size() - 1])).append("], \"headNum\": ");
    param.append(std::to_string(headNum)).append("}");
    atb::Operation *operation = FastSoftMaxOperationCreate(param);
    TORCH_CHECK(operation, "execute failed");

    atb::Tensor atbDataInput = AtTensor2Tensor(dataInput.contiguous());
    atbDataInput.desc.format = ACL_FORMAT_ND;

    std::vector<atb::Tensor> inTensors;
    inTensors.push_back(atbDataInput);
    std::vector<at::Tensor> outTensors;
    atb::VariantPack variantPack;
    BuildVariantPack(inTensors, outTensors, variantPack, operation);
    RunAtbOps(variantPack, "FastSoftMaxOperation", operation);
    return outTensors[0];
}

at::Tensor fastsoftmaxgrad(const at::Tensor &yInput, const at::Tensor &yGrad,const std::vector<int32_t> &seqLen,
    int32_t headNum)
{
    std::string param = "{\"seqLen\": [";
    for (int i = 0; i < seqLen.size() - 1; ++i) {
        param.append(std::to_string(seqLen[i])).append(",");
    }
    param.append(std::to_string(seqLen[seqLen.size() - 1])).append("], \"headNum\": ");
    param.append(std::to_string(headNum)).append("}");
    atb::Operation *operation = FastSoftMaxGradOperationCreate(param);
    TORCH_CHECK(operation, "execute failed");

    atb::Tensor atbYInput = AtTensor2Tensor(yInput.contiguous());
    atb::Tensor atbYGrad = AtTensor2Tensor(yGrad.contiguous());
    atbYInput.desc.format = ACL_FORMAT_ND;
    atbYGrad.desc.format = ACL_FORMAT_ND;

    std::vector<atb::Tensor> inTensors;
    inTensors.push_back(atbYInput);
    inTensors.push_back(atbYGrad);
    std::vector<at::Tensor> outTensors;
    atb::VariantPack variantPack;
    BuildVariantPack(inTensors, outTensors, variantPack, operation);
    RunAtbOps(variantPack, "FastSoftMaxGradOperation", operation);
    return outTensors[0];
}
