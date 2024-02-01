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

#include "acl/acl.h"
#include "atb/operation.h"
#include "atb/train_op_params.h"
#include "atb/types.h"
#include "atb/utils.h"

#include "inc/adapter.h"
#include "../ops_interface.h"
#include "nlohmann/json.hpp"

static atb::Operation *StridedBatchMatmulOperationCreate(const string param1)
{
    nlohmann::json paramJson;
    try {
        paramJson = nlohmann::json::parse(param1);
    } catch (const std::exception &e) {
        std::cout << "=======================parse json fail, error:" << std::endl;
        std::cout << e.what() << std::endl;
    }
    atb::train::StridedBatchMatmulParam param;
    if (paramJson.contains("transA")) {
        param.transposeA = paramJson["transA"].get<int32_t>();
    }
    if (paramJson.contains("transB")) {
        param.transposeB = paramJson["transB"].get<int32_t>();
    }
    for (auto item : paramJson["m"]) {
        param.m.push_back(item.get<int>());
    }
    for (auto item : paramJson["k"]) {
        param.k.push_back(item.get<int>());
    }
    for (auto item : paramJson["n"]) {
        param.n.push_back(item.get<int>());
    }
    for (auto item : paramJson["lda"]) {
        param.lda.push_back(item.get<int>());
    }
    for (auto item : paramJson["ldb"]) {
        param.ldb.push_back(item.get<int>());
    }
    for (auto item : paramJson["ldc"]) {
        param.ldc.push_back(item.get<int>());
    }
    for (auto item : paramJson["strideA"]) {
        param.strideA.push_back(item.get<int>());
    }
    for (auto item : paramJson["strideB"]) {
        param.strideB.push_back(item.get<int>());
    }
    for (auto item : paramJson["strideC"]) {
        param.strideC.push_back(item.get<int>());
    }
    if (paramJson.contains("batch")) {
        param.batch = paramJson["batch"].get<int32_t>();
    }
    if (paramJson.contains("headNum")) {
        param.headNum = paramJson["headNum"].get<int32_t>();
    }
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}

at::Tensor stridedbatchmatmul(const torch::Tensor &input1, const torch::Tensor &input2, int32_t transA, int32_t transB, const std::vector<int32_t> m, const std::vector<int32_t> k, const std::vector<int32_t> n, const std::vector<int32_t> lda, const std::vector<int32_t> ldb, const std::vector<int32_t> ldc, const std::vector<int32_t> strideA, const std::vector<int32_t> strideB, const std::vector<int32_t> strideC, int32_t batch, int32_t headNum) 
{
    std::stringstream ss;

    ss << "{\"transA\": " << transA << ", \"transB\": " << transB;
    ss << ", \"m\": [" << m[0];
    for (size_t i = 1; i < m.size(); ++i) {
        ss << ", " << m[i];
    }
    ss << "], \"k\": [" << k[0];
    for (size_t i = 1; i < k.size(); ++i) {
        ss << ", " << k[i];
    }
    ss << "], \"n\": [" << n[0];
    for (size_t i = 1; i < n.size(); ++i) {
        ss << ", " << n[i];
    }
    ss << "], \"lda\": [" << lda[0];
    for (size_t i = 1; i < lda.size(); ++i) {
        ss << ", " << lda[i];
    }
    ss << "], \"ldb\": [" << ldb[0];
    for (size_t i = 1; i < ldb.size(); ++i) {
        ss << ", " << ldb[i];
    }
    ss << "], \"ldc\": [" << ldc[0];
    for (size_t i = 1; i < ldc.size(); ++i) {
        ss << ", " << ldc[i];
    }
    ss << "], \"strideA\": [" << strideA[0];
    for (size_t i = 1; i < strideA.size(); ++i) {
        ss << ", " << strideA[i];
    }
    ss << "], \"strideB\": [" << strideB[0];
    for (size_t i = 1; i < strideB.size(); ++i) {
        ss << ", " << strideB[i];
    }
    ss << "], \"strideC\": [" << strideC[0];
    for (size_t i = 1; i < strideC.size(); ++i) {
        ss << ", " << strideC[i];
    }
    ss << "], \"batch\": " << batch << ", \"headNum\": " << headNum;
    ss << "}";
    std::string param = ss.str();
    atb::Operation *operation = StridedBatchMatmulOperationCreate(param);
    TORCH_CHECK(operation, "execute failed");

    atb::Tensor atbInTensor1 = AtTensor2Tensor(input1.contiguous());
    atb::Tensor atbInTensor2 = AtTensor2Tensor(input2.contiguous());
    atbInTensor1.desc.format = ACL_FORMAT_ND;
    atbInTensor2.desc.format = ACL_FORMAT_ND;
    
    std::vector<atb::Tensor> inTensors;
    inTensors.push_back(atbInTensor1);
    inTensors.push_back(atbInTensor2);
    std::vector<at::Tensor> outTensors;
    atb::VariantPack variantPack;
    BuildVariantPack(inTensors, outTensors, variantPack, operation);
    RunAtbOps(variantPack, "StridedBatchMatmulOperation", operation);
    return outTensors[0];
 }