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
#ifndef ATB_SPEED_MODELS_BAICHUAN2_7B_FLASH_ATTENTION_QUANT_MODEL_H
#define ATB_SPEED_MODELS_BAICHUAN2_7B_FLASH_ATTENTION_QUANT_MODEL_H
#include "atb_speed/base/model.h"
#include "atb_speed/utils/model_factory.h"

namespace atb_speed {
namespace baichuan2_7b {
class FlashAttentionQuantModel : public Model {
public:
    struct Param {
        double rmsNormEps = 0;
        int headNum = 0;
        int dk = 0;
        int layerNum = 0;
        int rank = 0;
        int rankSize = 1;
        std::string backend = "hccl";
        std::vector<float> w_packInputScale;
        std::vector<int> w_packInputOffset;
        std::vector<float> o_projInputScale;
        std::vector<int> o_projInputOffset;
        std::vector<float>gate_projInputScale;
        std::vector<int> gate_projInputOffset;
        std::vector<float> down_projInputScale;
        std::vector<int> down_projInputOffset;
        std::vector<float> up_projInputScale;
        std::vector<int> up_projInputOffset;
        std::vector<int> roll_back_layer;
        void FromString(const std::string &param);
    };

    explicit FlashAttentionQuantModel(const std::string &param);
    ~FlashAttentionQuantModel();
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;
    atb::Status InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
                           std::vector<atb::TensorDesc> &outTensorDescs) override;

private:
    virtual int64_t BuildGraph() override;
    Param param_;
    atb::Status ParseParam(const std::string &param) override;
    atb::Status BindParamHostTensor(uint32_t nodeId) override;
    std::vector<int32_t> tokenOffset_;
    std::vector<int32_t> seqLen_;
};

REGISTER_MODEL(baichuan2_7b, FlashAttentionQuantModel);

} // namespace baichuan2_13b
} // namespace atb_speed
#endif
