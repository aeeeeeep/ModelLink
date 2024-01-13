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
#ifndef ATB_SPEED_PLUGIN_ACLNN_MATMUL_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_MATMUL_OPERATION_H
#include <vector>
#include <string>
#include "atb/operation.h"
#include "acl/acl.h"
#include "aclnn/acl_meta.h"

namespace atb_speed
{
    namespace common
    {
        class MatMulCompressDequantOperation : public atb::Operation
        {
        public:
            explicit MatMulCompressDequantOperation(const std::string &name);
            ~MatMulCompressDequantOperation() override;
            std::string GetName() const override;
            atb::Status InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                   atb::SVector<atb::TensorDesc> &outTensorDescs) const override;
            uint32_t GetInputNum() const override;
            uint32_t GetOutputNum() const override;
            atb::Status Setup(const atb::VariantPack &variantPack, uint64_t &workspaceSize, atb::Context *context) override;
            atb::Status Execute(const atb::VariantPack &variantPack, uint8_t *workspace, uint64_t workspaceSize,
                                atb::Context *context) override;

        private:
            aclError CheckAcl(aclError x);
            std::string name_;

            aclOpExecutor *m_executor = nullptr;

            aclTensor *x1_ = nullptr;
            aclTensor *x2_ = nullptr;
            aclTensor *compressIndex_ = nullptr;
            aclTensor *bias_ = nullptr;
            aclTensor *deqScale_ = nullptr;
        };
    }
} // namespace atb_speed
#endif