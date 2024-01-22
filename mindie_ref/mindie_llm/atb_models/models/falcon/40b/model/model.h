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
#ifndef ATB_SPEED_MODELS_FALCON_40B_FUSION_MODEL_H
#define ATB_SPEED_MODELS_FALCON_40B_FUSION_MODEL_H

#include <vector>
#include "atb_speed/base/model.h"

namespace atb_speed {
namespace falcon_40b {
class FusionModel : public Model {
public:
    struct Param {
        int headNum = 0;
        int hiddenSize = 0;
        int kvHeadNum = 0;
        int layerNum = 0;
        int headDim = 0;
        int rank = 0;
        int rankSize = 1;
        int axis = 0;
        int rotaryCoeff = 2;
        float qScale = 1.0;
        float qkScale = 1.0;
        double layerNormEps = 0.0;
        std::string model = "falcon_40b";
        void FromString(const std::string &param);
    };

    explicit FusionModel(const std::string &param);

    ~FusionModel();

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
} // namespace falcon_40b
} // namespace atb_speed
#endif
