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
#ifndef ATB_ADAPTER_H
#define ATB_ADAPTER_H
#include <vector>
#include <string>
#include <torch/script.h>
#include <torch/custom_class.h>
#include "atb/types.h"
#include "atb/operation.h"
#include "atb/utils.h"

static atb::Context* msContext = nullptr;
atb::Tensor AtTensor2Tensor(const at::Tensor atTensor);
atb::Context* GetContext();
at::Tensor GetWorkspaceTensor(uint64_t workspaceSize, atb::Operation *operation);
uint64_t OperationSetup(atb::VariantPack variantPack, atb::Operation *operation, atb::Context* contextPtr);
#define RUN_TE_CMD(param, paramsetter, name)                                                                 \
    atb::Operation* op = nullptr;                                                                            \
    auto contextPtr = GetContext();                                                                          \
    atb::CreateOperation(param, &op);                                                                        \
    TORCH_CHECK(op != nullptr, "get op failed!");                                                            \
    uint64_t workspaceSize = OperationSetup(paramsetter.variantPack, op, contextPtr);                        \
    auto workspaceTensor = GetWorkspaceTensor(workspaceSize, op);                                            \
    void *workspacePtr = nullptr;                                                                            \
    workspacePtr = workspaceTensor.storage().data();                                                         \
    auto acl_call = [op, contextPtr, paramsetter, workspacePtr, workspaceSize]() -> int {                    \
        auto st = op->Execute(paramsetter.variantPack, (uint8_t *)workspacePtr, workspaceSize, contextPtr);  \
        DestroyOperation(op);                                                                                \
        return 0;                                                                                            \
    };                                                                                                       \
    at_npu::native::OpCommand cmd;                                                                           \
    cmd.Name(name);                                                                                          \
    cmd.SetCustomHandler(acl_call);                                                                          \
    cmd.Run();                                                                                               \

class ParamSetter {
public:
    ParamSetter& Input(const at::Tensor &tensor);
    ParamSetter& Input(const c10::optional<at::Tensor> &tensor);
    ParamSetter& Output(at::Tensor &tensor);
    atb::VariantPack variantPack;
};

#endif
