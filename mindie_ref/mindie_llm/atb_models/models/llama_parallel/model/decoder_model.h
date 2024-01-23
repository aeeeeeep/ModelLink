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
#ifndef ATB_SPEED_MODELS_LLAMA_PARALLEL_DECODER_MODEL_H
#define ATB_SPEED_MODELS_LLAMA_PARALLEL_DECODER_MODEL_H

#include <vector>
#include "atb_speed/base/model.h"

namespace atb_speed {
namespace llama_parallel {
class DecoderModel : public Model {
public:
    struct Param {
        bool isFA = true;                    // isFA为true则使用Flash Attention; 反之，则使用Paged Attention
        bool isPrefill = false;              // isPrefill为true时为全量阶段，encoder的isPrefill参数应为true; isPrefill为false时为增量阶段，decoder的isPrefill参数应为false
        bool isBF16 = false;                 // isBF16为true时采用BF16精度; 反之，则采用FP16精度
        bool isPack = true;                  // isPack为true时QKV和MLP中的gate和up权重合并; 反之，则权重不合并
        bool isEmbeddingParallel = false;    // isEmbeddingParallel为true时，embedding的权重在hiddenSize维度进行切分; 反之，则不对权重进行切分; 测试表明embedding切分并不会带来性能提升
        bool isLmHeadParallel = true;        // isLmHeadParallel为true时，LmHead的权重在vacobSize维度进行切分; 反之，则不对权重进行切分
        int quantType = 0;                   // 0 - No quant; 1- Quant in RmsNorm，dequant in Linear; 2 - Both quant and dequant in Linear
        float rmsNormEps = 0;
        int numAttentionHeadsPerRank = 0;
        int hiddenSizePerAttentionHead = 0;
        int numHiddenLayers = 0;
        int numKeyValueHeadsPerRank = 0;
        int rank = 0;
        int worldSize = 1;
        std::string backend = "hccl";
        std::vector<int> tokenOffset = {};
        std::vector<int> seqLen = {};
        void FromString(const std::string &param);
    };

    explicit DecoderModel(const std::string &param);
    ~DecoderModel();
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;
    atb::Status InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
                           std::vector<atb::TensorDesc> &outTensorDescs) override;

private:
    int64_t BuildGraph() override;
    atb::Status ParseParam(const std::string &param) override;
    atb::Status BindParamHostTensor(uint32_t nodeId) override;
    Param param_;
    std::vector<int> tokenOffset_;
    std::vector<int> seqLen_;
    int32_t layerId_ = 0;
};
}  // namespace llama_parallel
}  // namespace atb_speed
#endif
