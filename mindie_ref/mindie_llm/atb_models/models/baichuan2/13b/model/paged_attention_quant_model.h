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
#ifndef ATB_SPEED_MODELS_BAICHUAN2_13B_PA_QUANT_MODEL_H
#define ATB_SPEED_MODELS_BAICHUAN2_13B_PA_QUANT_MODEL_H

#include "atb_speed/base/model.h"

namespace atb_speed {
namespace baichuan2_13b {
class PagedAttentionQuantModel : public Model {
public:
    struct Param {
        float rmsNormEps = 0;
        int headNum = 0;
        int dk = 0;
        int layerNum = 0;
        bool transposedWeight = true;
        bool isPrefill = false;
        int rank = 0;
        int rankSize = 1;
        bool isLmHeadParallel = true;
        bool isOpera = false;
        std::string backend = "hccl";
        std::vector<float> wPackInputScale;
        std::vector<int> wPackInputOffset;
        std::vector<float> oProjInputScale;
        std::vector<int> oProjInputOffset;
        std::vector<float> gateProjInputScale;
        std::vector<int> gateProjInputOffset;
        std::vector<float> downProjInputScale;
        std::vector<int> downProjInputOffset;
        std::vector<float> upProjInputScale;
        std::vector<int> upProjInputOffset;
        std::vector<int> rollBackLayer;

        void FromString(const std::string &param);
    };

    explicit PagedAttentionQuantModel(const std::string &param);

    ~PagedAttentionQuantModel();

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
    std::vector<int32_t> seqLen_;
};

} // namespace baichuan2_13b
} // namespace atb_speed
#endif
