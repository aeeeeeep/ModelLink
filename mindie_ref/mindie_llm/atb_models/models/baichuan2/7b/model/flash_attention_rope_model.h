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
#ifndef ATB_SPEED_MODELS_BAICHUAN2_7B_FLASH_ATTENTION_ROPE_MODEL_H
#define ATB_SPEED_MODELS_BAICHUAN2_7B_FLASH_ATTENTION_ROPE_MODEL_H
#include "atb_speed/base/model.h"
#include "atb_speed/utils/model_factory.h"

namespace atb_speed {
namespace baichuan2_7b {
class FlashAttentionRopeModel : public Model {
public:
    struct Param {
        double rmsNormEps = 0;
        int headNum = 0;
        int dk = 0;
        int layerNum = 0;
        int rank = 0;
        int rankSize = 1;
        std::string backend = "hccl";
        // isFA为true则使用Flash Attention; 反之，则使用Paged Attention
        bool isFA = true;
        // isPrefill为true时为全量阶段，encoder的isPrefill参数应为true;
        // isPrefill为false时为增量阶段，decoder的isPrefill参数应为false
        bool isPrefill = true;
        // isPack为true时QKV和MLP中的gate和up权重合并; 反之，则权重不合并
        bool isPack = true;
        void FromString(const std::string &param);
    };

    explicit FlashAttentionRopeModel(const std::string &param);
    ~FlashAttentionRopeModel();
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

} // namespace baichuan2_7b
} // namespace atb_speed
#endif
