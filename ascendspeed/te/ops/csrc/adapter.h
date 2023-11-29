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
#ifndef ADAPTER_H
#define ADAPTER_H
#include <vector>
#include <string>
#include <torch/script.h>
#include <torch/custom_class.h>
#include "atb/types.h"
#include "atb/operation.h"
#include "atb/utils.h"

#define OP_SETPARAM(OpParam)                                                      \
    void SetParam(const OpParam &opParam, TECommand &command)                     \
    {                                                                             \
        atb::Operation* operation = nullptr;                                      \
        atb::CreateOperation(opParam, &operation);                                \
        command.SetOperation(&operation);                                         \
    }

atb::Tensor Input(const at::Tensor &tensor);
atb::Tensor Input(const c10::optional<at::Tensor> &tensor);
at::Tensor CreateAtTensorFromTensorDesc(const atb::TensorDesc &tensorDesc);
atb::Tensor AtTensor2Tensor(const at::Tensor atTensor);
atb::Context* GetContext();

class TECommand {
public:
    TECommand();
    ~TECommand()
    {
        std::cout<<"TECommand quit!"<<std::endl;
        operation = nullptr;
    }
    TECommand& SetOperation(atb::Operation **operation);
    TECommand& Name(std::string name);
    TECommand& Input(const at::Tensor &tensor);
    TECommand& Input(const at::Tensor &tensor, bool isNone);
    TECommand& Input(const c10::optional<at::Tensor> &tensor);
    void Output(std::vector<at::Tensor> &output);

private:
    void BuildVariantPack(std::vector<at::Tensor> &output);

private:
    std::string name;
    atb::Operation* operation = nullptr;
    atb::VariantPack variantPack;
    std::vector<atb::Tensor> inTensors;
};

#endif
