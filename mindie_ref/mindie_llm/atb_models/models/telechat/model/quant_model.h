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
#ifndef ATB_SPEED_MODELS_TELECHAT_ALL_QUANT_FA_MODEL_H
#define ATB_SPEED_MODELS_TELECHAT_ALL_QUANT_FA_MODEL_H
#include <atb/svector.h>
#include "atb_speed/base/model.h"

namespace atb_speed {
namespace telechat {
class QuantFAModel : public Model {
public:
    struct QuantFAParam {
        bool transKey = false;
        int layerNum = 0;
        int headNum = 0;
        int dk = 0;
        double rmsNormEps = 0;
        int rank = 0;
        int rankSize = 1;
        void FromString(const std::string &param);

        // quant+float
        std::vector<float> float_query_layers;
        std::vector<float> float_kv_layers;
        std::vector<float> float_down_layers;

        // quant
        std::vector<float> inputScale_qkv;
        std::vector<int> inputOffset_qkv;
        std::vector<float> inputScale_dense;
        std::vector<int> inputOffset_dense;
        std::vector<float> inputScale_gate_up;
        std::vector<int> inputOffset_gate_up;
        std::vector<float> inputScale_down_proj;
        std::vector<int> inputOffset_down_proj;
    };

    explicit QuantFAModel(const std::string &param);
    ~QuantFAModel();
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;
    atb::Status InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
                           std::vector<atb::TensorDesc> &outTensorDescs) override;

private:
    void BuildGraph() override;
    atb::Status ParseParam(const std::string &param) override;
    atb::Status BindParamHostTensor(uint32_t nodeId) override;
    QuantFAParam param_;
    atb::SVector<int32_t> tokenOffset_;
    atb::SVector<int32_t> seqLen_;
};
}  // namespace telechat
}  // namespace atb_speed
#endif