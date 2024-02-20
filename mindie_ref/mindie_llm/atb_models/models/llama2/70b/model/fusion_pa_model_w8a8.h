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
#ifndef ATB_SPEED_MODELS_LLAMA2_70B_FUSION_PA_MODEL_W8A8_H
#define ATB_SPEED_MODELS_LLAMA2_70B_FUSION_PA_MODEL_W8A8_H

#include <vector>
#include "atb_speed/base/model.h"

namespace atb_speed {
namespace llama2_70b {
class FusionPAModelW8A8 : public Model {
public:
    struct Param {
        double rmsNormEps = 0;
        int headNum = 0;
        int dk = 0;
        int layerNum = 0;
        int rank = 0;
        int rankSize = 1;
        int numHeadsPerPartition = 0;
        float qkScale = 1.0;
        int rotaryCoeff = 2;
        bool transposedWeight = false;
        bool isPrefill = false;
        std::string backend = "hccl";
        std::vector<int> tokenOffset = {};
        std::vector<int> seqLen = {};

        void FromString(const std::string &param);
    };

    explicit FusionPAModelW8A8(const std::string &param);

    ~FusionPAModelW8A8();

    uint32_t GetInputNum() const override;

    uint32_t GetOutputNum() const override;

    atb::Status InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
                           std::vector<atb::TensorDesc> &outTensorDescs) override;

private:
    int64_t BuildGraph() override;
    Param param_;
    atb::Status ParseParam(const std::string &param) override;
    atb::Status BindParamHostTensor(uint32_t nodeId) override;
    std::vector<int> tokenOffset_;
    std::vector<int> seqLen_;
    int32_t layerId_ = 0;
};
} // namespace llama2_70b
} // namespace atb_speed
#endif
