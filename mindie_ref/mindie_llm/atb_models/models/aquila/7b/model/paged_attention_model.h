/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
#ifndef ATB_SPEED_MODELS_AQUILA_7B_PAGED_ATTENTION_MODEL_H
#define ATB_SPEED_MODELS_AQUILA_7B_PAGED_ATTENTION_MODEL_H

#include <vector>
#include "atb_speed/base/model.h"
#include "atb_speed/utils/model_factory.h"

namespace atb_speed {
namespace aquila_7b {
class PagedAttentionRopeModel : public Model {
public:
    struct Param {
        bool isFA = true;
        bool isPrefill = false;
        bool isBF16 = false;
        bool isEmbeddingParallel = false;
        bool isLmHeadParallel = true;
        bool supportSwiGLU = false;
        int numAttentionHeadsPerRank = 0;
        int hiddenSizePerAttentionHead = 0;
        int numHiddenLayers = 0;
        int numKeyValueHeadsPerRank = 0;
        float rmsNormEps = 0;
        int rank = 0;
        int rankSize = 1;
        std::string backend = "hccl";
        std::vector<int> tokenOffset = {};
        std::vector<int> seqLen = {};
        std::vector<std::vector<int>> packQuantType = {};
        std::vector<std::vector<int>> linearQuantType = {};
        void FromString(const std::string &param);
    };

    explicit PagedAttentionRopeModel(const std::string &param);

    ~PagedAttentionRopeModel();

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

REGISTER_MODEL(aquila_7b, PagedAttentionRopeModel);

} // namespace aquila_7b
} // namespace atb_speed
#endif