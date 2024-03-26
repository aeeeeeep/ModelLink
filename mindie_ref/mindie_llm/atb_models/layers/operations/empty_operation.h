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

#ifndef EMPTY_OPERATION_H
#define EMPTY_OPERATION_H

#include "atb/operation.h"

class EmptyOperation : public atb::Operation {
public:
    EmptyOperation() {}
    ~EmptyOperation() {}
    std::string GetName() const {return "EmptyOperation";}
    atb::Status InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs, atb::SVector<atb::TensorDesc> &outTensorDescs) const {
        ATB_LOG(DEBUG) << "EmptyOperation inTensorDescs size " << inTensorDescs.size();
        ATB_LOG(DEBUG) << "EmptyOperation outTensorDescs size " << outTensorDescs.size();
        return 0;
    }
    uint32_t GetInputNum() const {return 0;}
    uint32_t GetOutputNum() const {return 1;}
    atb::Status Setup(const atb::VariantPack &variantPack, uint64_t &workspaceSize, atb::Context *context) {
        ATB_LOG(DEBUG) << "EmptyOperation variantPack outTensor size " << variantPack.outTensors.size();
        ATB_LOG(DEBUG) << "EmptyOperation workspaceSize " << workspaceSize;
        ATB_LOG(DEBUG) << "EmptyOperation context " << context;
        return 0;
    }
    atb::Status Execute(const atb::VariantPack &variantPack, uint8_t *workspace, uint64_t workspaceSize,
        atb::Context *context)  {
        ATB_LOG(DEBUG) << "EmptyOperation variantPack outTensor size " << variantPack.outTensors.size();
        ATB_LOG(DEBUG) << "EmptyOperation workspace " << workspace;
        ATB_LOG(DEBUG) << "EmptyOperation workspaceSize " << workspaceSize;
        ATB_LOG(DEBUG) << "EmptyOperation context " << context;
        return 0;
    }
};
#endif