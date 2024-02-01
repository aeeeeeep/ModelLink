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

#include <sstream>
#include <string>
#include <vector>

#include "inc/adapter.h"
#include "../ops_interface.h"
#include "atb/operation.h"
#include "atb/train_op_params.h"
#include "nlohmann/json.hpp"
using namespace std;

static atb::Operation *PadOperationCreate(const string param1)
{
    nlohmann::json paramJson;
    try {
        paramJson = nlohmann::json::parse(param1);
    } catch (const std::exception &e) {
        cout << "=======================parse json fail, error:" << endl;
        cout << e.what() << endl;
    }
    atb::train::PadWithHiddenStateParam param;
    if (paramJson.contains("maxSeqLen")) {
        param.maxSeqLen = paramJson["maxSeqLen"].get<int32_t>();
    }
    for (auto item : paramJson["seqLen"]) {
        param.qSeqLen.push_back(item.get<int>());
    }
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}

at::Tensor pad(const torch::Tensor &input, const std::vector<int> seqLen, int maxSeqLen)
{
    string param = "{\"seqLen\": [";   // {"headNum": 0, "maxSeqLen": None}
    for (int i = 0; i < seqLen.size() - 1; ++i) {
        param.append(to_string(seqLen[i])).append(",");
    }
    param.append(to_string(seqLen[seqLen.size() - 1])).append("], \"maxSeqLen\": ");
    param.append(to_string(maxSeqLen)).append("}");

    atb::Operation *operation = PadOperationCreate(param);
    TORCH_CHECK(operation, "execute failed");

    atb::Tensor atbInTensor = AtTensor2Tensor(input);
    atbInTensor.desc.format = ACL_FORMAT_ND;
    std::vector<atb::Tensor> inTensors;
    inTensors.push_back(atbInTensor);
    std::vector<at::Tensor> outTensors;
    atb::VariantPack variantPack;
    BuildVariantPack(inTensors, outTensors, variantPack, operation);
    RunAtbOps(variantPack, "PadWithHiddenStateOperation", operation);
    return outTensors[0];

}
