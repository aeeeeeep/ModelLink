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
#ifndef ATB_SPEED_MODELS_VISUALGLM_6B_ENCODER_MODEL_H
#define ATB_SPEED_MODELS_VISUALGLM_6B_ENCODER_MODEL_H
#include <atb/svector.h>
#include <vector>
#include "atb_speed/base/model.h"
#include "atb_speed/utils/model_factory.h"

namespace atb_speed {
namespace visualglm_6b {
class FlashAttentionModel : public Model {
public:
    struct Param {
        double layerNormEps = 0;
        int headNum = 0;
        int headDim = 0;
        int layerNum = 0;
        bool isEncoder = true;
        int operationCountBeforeLayers = 0;
        std::vector<float> qScale;
        std::vector<float> qkScale;
        float residualAddScale = 0;
        int beginNormAxis = 2;
        void FromString(const std::string &param);
    };

    explicit FlashAttentionModel(const std::string &param);
    ~FlashAttentionModel();
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;
    atb::Status InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
                           std::vector<atb::TensorDesc> &outTensorDescs) override;

private:
    int64_t BuildGraph() override;
    atb::Status ParseParam(const std::string &param) override;
    atb::Status BindParamHostTensor(uint32_t nodeId) override;

private:
    Param param_;
    std::vector<int> tokenOffset_;
    std::vector<int> seqLen_;
};

REGISTER_MODEL(visualglm_6b, FlashAttentionModel);

} // namespace visualglm_6b
} // namespace atb_speed
#endif