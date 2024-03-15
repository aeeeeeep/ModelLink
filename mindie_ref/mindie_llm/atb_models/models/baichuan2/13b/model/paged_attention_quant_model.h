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

#include <vector>
#include "atb_speed/base/model.h"
#include "atb_speed/utils/model_factory.h"

namespace atb_speed {
namespace baichuan2_13b {
class PagedAttentionQuantModel : public Model {
public:
    struct Param {
        // isFA为true则使用Flash Attention; 反之，则使用Paged Attention
        bool isFA = false;
        // isPrefill为true时为全量阶段，encoder的isPrefill参数应为true; isPrefill为false时为增量阶段，decoder的isPrefill参数应为false
        bool isPrefill = false;
        // isBF16为true时采用BF16精度; 反之，则采用FP16精度
        bool isBF16 = false;
        bool isEmbeddingParallel = true;  // 默认是在hidden_states维度进行切分
        // isLmHeadParallel为true时，LmHead的权重在vacobSize维度进行切分; 反之，则不对权重进行切分
        bool isLmHeadParallel = true;
        bool supportSwiGLU = true;
        
        int numAttentionHeadsPerRank = 0;
        int hiddenSizePerAttentionHead = 0;
        int numHiddenLayers = 0;
        int numKeyValueHeadsPerRank = 0;
        float rmsNormEps = 0;
        int rank = 0;
        int worldSize = 1;
        std::string backend = "hccl";
        std::vector<int> seqLen = {};
        std::vector<std::vector<int>> packQuantType = {};
        std::vector<std::vector<int>> linearQuantType = {};
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

    Param param_;
    std::vector<int32_t> seqLen_;
};

} // namespace baichuan2_13b
} // namespace atb_speed
#endif
