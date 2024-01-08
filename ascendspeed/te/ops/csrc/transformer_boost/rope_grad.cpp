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

#include "nlohmann/json.hpp"
#include "inc/adapter.h"
#include "../ops_interface.h"
#include <acl/acl.h>
#include <acl/acl_rt.h>
#include "atb/context.h"
#include "atb/operation.h"
#include "atb/train_op_params.h"
using namespace std;

static atb::Operation *RopeGradOperationCreate(const string param1)
{
    nlohmann::json paramJson;
    try {
        paramJson = nlohmann::json::parse(param1);
    } catch (const std::exception &e) {
        cout << "=======================parse json fail, error:" << endl;
        cout << e.what() << endl;
    }
    atb::train::RopeGradParam param;
    for(auto item : paramJson["qSeqLen"]){
        param.qSeqLen.push_back(item.get<int>());
    }
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
} 

std::tuple<at::Tensor, at::Tensor> rope_grad(const torch::Tensor &input1, const torch::Tensor &input2,
    const torch::Tensor &input3, const torch::Tensor &input4, const std::vector<int> qSeqLen)
{
    string param = "{\"qSeqLen\": [";
    for (int i = 0; i < qSeqLen.size()-1; ++i) {
        param.append(to_string(qSeqLen[i])).append(",");
    }
    param.append(to_string(qSeqLen[qSeqLen.size()-1])).append("] }");

    atb::Operation* operation = RopeGradOperationCreate(param);
    TORCH_CHECK(operation, "execute failed");

    atb::Tensor atbInTensor1 = AtTensor2Tensor(input1);
    atb::Tensor atbInTensor2 = AtTensor2Tensor(input2);
    atb::Tensor atbInTensor3 = AtTensor2Tensor(input3);
    atb::Tensor atbInTensor4 = AtTensor2Tensor(input4);

    atbInTensor1.desc.format = ACL_FORMAT_ND;
    atbInTensor2.desc.format = ACL_FORMAT_ND;
    atbInTensor3.desc.format = ACL_FORMAT_ND;
    atbInTensor4.desc.format = ACL_FORMAT_ND;
    
    std::vector<atb::Tensor> inTensors;
    inTensors.push_back(atbInTensor1);
    inTensors.push_back(atbInTensor2);
    inTensors.push_back(atbInTensor3);
    inTensors.push_back(atbInTensor4);
    std::vector<at::Tensor> outTensors;
    atb::VariantPack variantPack;
    BuildVariantPack(inTensors, outTensors, variantPack, operation);
    RunAtbOps(variantPack, "RopeGradOperation", operation);
    return std::make_tuple(outTensors[0], outTensors[1]);
}