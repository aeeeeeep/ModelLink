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
#ifndef ATB_SPEED_MODELS_INTERNLM_20B_PAGED_ATTENTION_MODEL_H
#define ATB_SPEED_MODELS_INTERNLM_20B_PAGED_ATTENTION_MODEL_H
#include "atb_speed/base/model.h"
#include "atb_speed/utils/model_factory.h"

namespace atb_speed {
namespace internlm_20b {
class PagedAttentionModel : public Model {
public:
    struct Param {
        double rmsNormEps = 0;
        int headNum = 0;
        int dk = 0;
        int layerNum = 0;
        float qkScale = 1.0;
        int rank = 0;
        int rankSize = 1;
        bool isLmHeadParallel = true;
        bool isPrefill = false;
        std::string backend = "hccl";
        void FromString(const std::string &param);
    };

    explicit PagedAttentionModel(const std::string &param);

    ~PagedAttentionModel();

    uint32_t GetInputNum() const override;

    uint32_t GetOutputNum() const override;

    atb::Status InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
                           std::vector<atb::TensorDesc> &outTensorDescs) override;

private:
    int64_t BuildGraph() override;

    Param param_;

    atb::Status ParseParam(const std::string &param) override;

    atb::Status BindParamHostTensor(uint32_t nodeId) override;

    std::vector<int32_t> seqLen_;
};

REGISTER_MODEL(internlm_20b, PagedAttentionModel);

} // namespace internlm_20b
} // namespace atb_speed
#endif