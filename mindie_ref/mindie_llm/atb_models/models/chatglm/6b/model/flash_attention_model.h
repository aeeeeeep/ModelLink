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
#ifndef ATB_SPEED_MODELS_CHATGML_6B_COMMON_MODEL_FA_H
#define ATB_SPEED_MODELS_CHATGML_6B_COMMON_MODEL_FA_H

#include "atb_speed/base/model.h"
#include "atb_speed/utils/model_factory.h"

namespace atb_speed {
namespace chatglm_6b {
class ChatGlmCommonModelFa : public Model {
public:
    struct Param {
        float rmsNormEps = 0;
        int numHeadsPerPartition = 0;
        int hiddenSizePerHead = 0;
        bool transKey = true;
        bool quantmodel = false;
        bool isSparse = false;
        int layerNum = 0;
        int correctNodeId = -1;
        float residualAddScale = 0;
        std::vector<float> qkvInputScale;
        std::vector<int> qkvInputOffset;
        std::vector<float> denseInputScale;
        std::vector<int> denseInputOffset;
        std::vector<float> selfLnInputScale;
        std::vector<int> selfLnInputOffset;
        std::vector<float> ffnOutInputScale;
        std::vector<int> ffnOutInputOffset;
        std::vector<float> preScale;
        std::vector<float> postScale;
        void FromString(const std::string &param);
        int rank = 0;
        int rankSize = 1;
        std::string backend = "hccl";
        bool isEncoder = false;
        std::vector<int64_t> offsetX;
        std::vector<std::vector<int64_t>> compressInfo;
    };

    explicit ChatGlmCommonModelFa(const std::string &param);

    ~ChatGlmCommonModelFa();

    uint32_t GetInputNum() const override;

    uint32_t GetOutputNum() const override;

    atb::Status InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
                           std::vector<atb::TensorDesc> &outTensorDescs) override;

private:
    virtual int64_t BuildGraph() override;

    atb::Status ParseParam(const std::string &param) override;

    atb::Status BindParamHostTensor(uint32_t nodeId) override;

    Param param_;
    std::vector<int32_t> tokenOffset_;
    std::vector<int32_t> seqLen_;
};

REGISTER_MODEL(chatglm_6b, ChatGlmCommonModelFa);

} // namespace chatglm_6b
} // namespace atb_speed
#endif
