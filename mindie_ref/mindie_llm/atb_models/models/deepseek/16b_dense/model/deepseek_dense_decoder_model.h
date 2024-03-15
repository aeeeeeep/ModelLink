/*
* Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
*
*  * Licensed under the Apache License, Version 2.0 (the "License");
*  * you may not use this file except in compliance with the License.
*  * You may obtain a copy of the License at
*  *
*  * http://www.apache.org/licenses/LICENSE-2.0
*  *
*  * Unless required by applicable law or agreed to in writing, software
*  * distributed under the License is distributed on an "AS IS" BASIS,
*  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  * See the License for the specific language governing permissions and
*  * limitations under the License.
*  */
#ifndef ATB_SPEED_MODELS_DEEPSEEK_DENSE_DECODER_MODEL_H
#define ATB_SPEED_MODELS_DEEPSEEK_DENSE_DECODER_MODEL_H
#include <vector>
#include "atb_speed/base/model.h"
#include "atb_speed/utils/model_factory.h"

namespace atb_speed {
namespace deepseekDense {
class DeepseekDenseDecoderModel : public Model {
public:
    struct Param {
        float rmsNormEps = 0;
        int headNum = 0;
        int dk = 0;
        int rank = 0;
        int rankSize = 1;
        int layerNum = 0;
        int rankRoot = 0;
        int coderType = 0;
        int isTriMask = 0;
        int kvHeadNum = 0;
        int numOfExperts = 64;
        std::string model = "deepseek16B_dense";
        std::string backend = "lccl";
        std::string rankTableFile = "";
        int rotaryCoeff = 2;
        std::vector<int> tokenOffset = {};
        std::vector<int> seqLen = {};
        void FromString(const std::string &param);
    };

    explicit DeepseekDenseDecoderModel(const std::string &param);
    ~DeepseekDenseDecoderModel();
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
    int32_t layerId_ = 0;
};

REGISTER_MODEL(deepseekDense, DeepseekDenseDecoderModel);

}
} // namespace atb_speed
#endif