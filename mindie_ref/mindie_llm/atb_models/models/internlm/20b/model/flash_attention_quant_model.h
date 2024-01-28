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
#ifndef ATB_SPEED_MODELS_INTERNLM_20B_FLASH_ATTENTION_QUANT_MODEL_H
#define ATB_SPEED_MODELS_INTERNLM_20B_FLASH_ATTENTION_QUANT_MODEL_H

#include "atb_speed/base/model.h"

namespace atb_speed {
namespace internlm_20b {
class FlashAttentionQuantModel : public Model {
public:
    struct Param {
        float rmsNormEps = 0;                // 模型config：rms_norm_eps
        int headNum = 0;                     // 计算方式：（config.num_attention_heads // rankSize）
        int dk = 0;                          // 计算方式：（config.hidden_size // config.num_attention_heads）
        int layerNum = 0;                    // 模型layer层数
        int rank = 0;                        // 多卡并行模型id
        int rankSize = 1;                    // 模型切分数量
        std::string backend = "hccl";
        std::vector<float> qProjInputScale;  // 量化参数
        std::vector<int> qProjInputOffset;
        std::vector<float> kProjInputScale;
        std::vector<int> kProjInputOffset;
        std::vector<float> vProjInputScale;
        std::vector<int> vProjInputOffset;
        std::vector<float> oProjInputScale;
        std::vector<int> oProjInputOffset;
        std::vector<float> gateProjInputScale;
        std::vector<int> gateProjInputOffset;
        std::vector<float> downProjInputScale;
        std::vector<int> downProjInputOffset;
        std::vector<int> float_layers;
        void FromString(const std::string &param);
    };

    explicit FlashAttentionQuantModel(const std::string &param);

    ~FlashAttentionQuantModel();

    uint32_t GetInputNum() const override;

    uint32_t GetOutputNum() const override;

    atb::Status InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
                           std::vector<atb::TensorDesc> &outTensorDescs) override;

private:
    int64_t BuildGraph() override;

    Param param_;

    atb::Status ParseParam(const std::string &param) override;

    atb::Status BindParamHostTensor(uint32_t nodeId) override;

    std::vector<int32_t> tokenOffset_;

    std::vector<int32_t> seqLen_;
};
} // namespace internlm_20b
} // namespace atb_speed
#endif
