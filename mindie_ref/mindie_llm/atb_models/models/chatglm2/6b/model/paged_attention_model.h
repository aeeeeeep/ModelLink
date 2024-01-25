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
#ifndef ATB_SPEED_MODELS_CHATGLM2_6B_DECODER_PA_MODEL_H
#define ATB_SPEED_MODELS_CHATGLM2_6B_DECODER_PA_MODEL_H

#include "atb_speed/base/model.h"

namespace atb_speed {
namespace chatglm2_6b {
class PagedAttentionModel : public Model {
public:
    struct Param {
        float rmsNormEps = 0;
        int headNum = 0;
        int dk = 0;
        bool isPrefill = true;
        int numHeadsPerPartition = 0;
        int hiddenSizePerHead = 1;
        int numGroupsPerPartition = 1;
        bool transKey = false;
        int layerNum = 0;
        float residualAddScale = 0;
        int rank = 0;
        int rankSize = 1;
        bool isLmHeadParallel = true;
        
        void FromString(const std::string &param);
    };

    explicit PagedAttentionModel(const std::string &param);

    ~PagedAttentionModel();

    uint32_t GetInputNum() const override;

    uint32_t GetOutputNum() const override;

    atb::Status InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
                           std::vector<atb::TensorDesc> &outTensorDescs) override;

private:
    virtual int64_t BuildGraph() override;

    atb::Status ParseParam(const std::string &param) override;

    atb::Status BindParamHostTensor(uint32_t nodeId) override;

private:
    Param param_;
    std::vector<int32_t> seqLen_;
};
} // namespace chatglm2_6b
} // namespace atb_speed
#endif
