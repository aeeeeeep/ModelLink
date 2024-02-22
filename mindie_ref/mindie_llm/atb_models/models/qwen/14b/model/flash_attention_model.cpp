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
#include "models/qwen/14b/model/flash_attention_model.h"

#include "atb/atb_infer.h"
#include "nlohmann/json.hpp"

#include "models/qwen/14b/layer/flash_attention_layer.h"
#include "operations/lmhead.h"

namespace atb_speed {
namespace qwen_14b {
enum InTensorId : int {
    IN_TENSOR_INPUTIDS = 0,
    IN_TENSOR_COSEMBED,
    IN_TENSOR_SINEMBED,
    IN_TENSOR_ATTENTIONMASK,
    IN_TENSOR_PAST_KEY,
    IN_TENSOR_PAST_VALUE,
    IN_TENSOR_TOKENOFFSET,
    IN_TENSOR_SEQLEN,
    IN_TENSOR_LOGNTENSOR,
    IN_HOLDER,
    IN_FINAL_NORM_SLICE_OFFSET,
    IN_TENSOR_MAX,
};

enum OutTensorId : int {
    OUT_TENSOR_HIDDENSTATES = 0,
    OUT_TENSOR_MAX,
};

// 这些变量在构建图的时候一一备注说明
const int WEIGHT_COUNT_PER_LAYER = 8;         // 每个block当中weight数量
const int WORDEMBEDDINGNODE_WEIGHT_COUNT = 1; // embedding对应的weight数量
const int FINALNORMNODE_WEIGHT_COUNT = 1;     // 所有block结束后RmsNorm对应的权重
const int OPERATION_COUNT_BEFORE_LAYER = 1;   // embedding占据一个node，源码inputs_embeds = self.wte(input_ids)
const int INTERMEDIATETENSOR_COUNT_BEFORE_LAYER = 1;
const int OPERATION_COUNT_AFTER_LAYER = 2;
const int OUT_TENSOR_HIDDENSTATES_ID = 0;
const int OUT_TENSOR_MAX_ID = 1;
const int IN_TENSOR_INPUTIDS_ID = 0;
const int WORDEMBEDDINGNODE_WEIGHT_ID = 0;
const int FIRST_INTERNAL_TENSORS = 0;
const int LAYER_FIRST_OUT_TENSORS = 0;
const int FA_ROPE_LAYER_IN_TOKENOFFSET_ID = 15;
const int FA_ROPE_LAYER_IN_SEQLEN_ID = 16;
const int OUT_TENSOR_HIDDENSTATES_ID_DIM_NUM = 3;
const int OUT_LM_HEAD_WEIGHT_COUNT = 1;

// 超参数初始化
void FlashAttentionModel::Param::FromString(const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);
    RmsNormEps = paramJson["RmsNormEps"].get<double>();
    headNum = paramJson["headNum"].get<int>();
    dk = paramJson["dk"].get<int>();
    layerNum = paramJson["layerNum"].get<int>();

    if (paramJson.contains("coderType")) {
        coderType = paramJson["coderType"].get<int>();
    }
    if (paramJson.contains("QKScale")) {
        QKScale = paramJson["QKScale"].get<float>();
    }
    if (paramJson.contains("rank")) {
        rank = paramJson["rank"].get<int>();
    }
    if (paramJson.contains("rankSize")) {
        rankSize = paramJson["rankSize"].get<int>();
    }
    if (paramJson.contains("backend")) {
        backend = paramJson["backend"];
    }
}

FlashAttentionModel::FlashAttentionModel(const std::string &param) : Model("FlashAttentionModel", param)
{
    param_.FromString(param);
}

FlashAttentionModel::~FlashAttentionModel() = default;

uint32_t FlashAttentionModel::GetInputNum() const { return graph_.inTensors.size(); } // 返回图中inTensors数目
uint32_t FlashAttentionModel::GetOutputNum() const { return graph_.outTensors.size(); }

atb::Status FlashAttentionModel::InferShape(const std::vector<atb::TensorDesc> &inTensorDescs, // 形状推断的视频路径
                                            std::vector<atb::TensorDesc> &outTensorDescs)
{
    if (outTensorDescs.size() != GetOutputNum()) {
        return atb::ERROR_INVALID_GRAPH;
    }

    const int64_t outDim = graph_.weightTensors.at(graph_.weightTensors.size() - 1)
                               .desc.shape.dims[0]; // weightTensors的最后一个tensor的第一个维度值
    outTensorDescs.at(OUT_TENSOR_HIDDENSTATES_ID) =
        graph_.weightTensors.at(WORDEMBEDDINGNODE_WEIGHT_ID).desc; // WORDEMBEDDINGNODE_WEIGHT_ID是第一个node节点，
    outTensorDescs.at(OUT_TENSOR_HIDDENSTATES_ID).shape.dimNum =
        OUT_TENSOR_HIDDENSTATES_ID_DIM_NUM; // 3维，[batchsize, seq_len,hiddenstates]

    size_t outTensorShapeDimIndex = 0;
    size_t inTensorShapeDimIndex = 0;

    //
    outTensorDescs.at(OUT_TENSOR_HIDDENSTATES_ID).shape.dims[outTensorShapeDimIndex++] =
        inTensorDescs.at(IN_TENSOR_INPUTIDS_ID).shape.dims[inTensorShapeDimIndex++];
    outTensorDescs.at(OUT_TENSOR_HIDDENSTATES_ID).shape.dims[outTensorShapeDimIndex++] = 1;
    outTensorDescs.at(OUT_TENSOR_HIDDENSTATES_ID).shape.dims[outTensorShapeDimIndex++] =
        outDim * param_.rankSize; // outTensorDescs的结果测试

    return atb::NO_ERROR;
}

int64_t FlashAttentionModel::BuildGraph()
{
    // 图中weightTensor总数量，分别是：1(word_embedding)、40*8（所有layer中weightTensor的总数量）、1(RmsNorm)、1(linear)
    const int weightTensorSize = WORDEMBEDDINGNODE_WEIGHT_COUNT + WEIGHT_COUNT_PER_LAYER * param_.layerNum +
                                 FINALNORMNODE_WEIGHT_COUNT + OUT_LM_HEAD_WEIGHT_COUNT;
    graph_.weightTensors.resize(weightTensorSize);

    graph_.inTensors.resize(IN_TENSOR_MAX + param_.layerNum);
    graph_.outTensors.resize(OUT_TENSOR_MAX);

    // 图中的Node节点数量
    const int nodeSize = param_.layerNum + OPERATION_COUNT_BEFORE_LAYER + OPERATION_COUNT_AFTER_LAYER; // 40 + 1 + 2
    graph_.nodes.resize(nodeSize);

    // internalTensors，数量等于node.size()-1，最后的tensor作为outTensor
    const int internalTensorSize = graph_.nodes.size() - 1;
    graph_.internalTensors.resize(internalTensorSize);

    // node1，对应wordEmbedding操作
    int nodeId = 0;
    auto &wordEmbeddingNode = graph_.nodes.at(nodeId++);
    atb::infer::GatherParam wordEmbeddingParam;
    atb::Operation *op = nullptr;              // operation指针
    CREATE_OPERATION(wordEmbeddingParam, &op); // 创建Operation/GraphOperation接口
    wordEmbeddingNode.operation.reset(op);     // 含义应当是初始化指针的位置
    wordEmbeddingNode.inTensors = {&graph_.weightTensors.at(WORDEMBEDDINGNODE_WEIGHT_ID),
                                   &graph_.inTensors.at(IN_TENSOR_INPUTIDS)};
    wordEmbeddingNode.outTensors = {
        &graph_.internalTensors.at(FIRST_INTERNAL_TENSORS)}; // 该operation的输出就是下一层的输入，也是中间值

    atb::Tensor *firstInTensor = &graph_.internalTensors.at(FIRST_INTERNAL_TENSORS); // 记录第一个中间结果
    // 40 * layer
    for (int layerId = 0; layerId < param_.layerNum; ++layerId) {
        auto &layerNode = graph_.nodes.at(nodeId++);               // 图中的节点
        atb_speed::qwen_14b::FlashAttentionRopeLayerParam opParam; // 获取layer层的参数
        opParam.rmsNormEps = param_.RmsNormEps;
        opParam.headNum = param_.headNum;
        opParam.dk = param_.dk;
        opParam.rank = param_.rank;
        opParam.rankSize = param_.rankSize;
        opParam.backend = param_.backend;
        opParam.coderType = param_.coderType;
        atb_speed::qwen_14b::FlashAttentionRopeLayer(opParam, &op);
        layerNode.operation.reset(op);
        layerNode.inTensors.resize(layerNode.operation->GetInputNum());

        size_t inTensorId = 0;
        layerNode.inTensors.at(inTensorId++) = firstInTensor; // 上一波的输出为本波的输入
        // 将本layer层有关的weight矩阵传入进来
        for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
            layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(
                layerId * WEIGHT_COUNT_PER_LAYER + weightTensorId + WORDEMBEDDINGNODE_WEIGHT_COUNT);
        }

        // layer层的输入
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_COSEMBED);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SINEMBED);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ATTENTIONMASK);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PAST_KEY);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PAST_VALUE);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_LOGNTENSOR);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_TOKENOFFSET);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SEQLEN);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_HOLDER);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_MAX + layerId);

        layerNode.outTensors = {&graph_.internalTensors.at(INTERMEDIATETENSOR_COUNT_BEFORE_LAYER + layerId)};
        firstInTensor = layerNode.outTensors.at(LAYER_FIRST_OUT_TENSORS);
    }

    // block结束后的RmsNorm
    auto &finalNormNode = graph_.nodes.at(nodeId++);
    atb::infer::RmsNormParam finalNormParam;
    finalNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    finalNormParam.normParam.epsilon = param_.RmsNormEps;
    CREATE_OPERATION(finalNormParam, &op);
    finalNormNode.operation.reset(op);
    const int finalLayerNormWeightTensorId =
        graph_.weightTensors.size() - FINALNORMNODE_WEIGHT_COUNT - OUT_LM_HEAD_WEIGHT_COUNT;
    const int finalLayerNormOutTensorId =
        internalTensorSize - 1; // 这里需要注意，internalTensorSize = graph_.nodes.size() - 2;
    finalNormNode.inTensors = {firstInTensor, &graph_.weightTensors.at(finalLayerNormWeightTensorId)};
    finalNormNode.outTensors = {&graph_.internalTensors.at(finalLayerNormOutTensorId)};

    const int finalLinearWeightTensorId = graph_.weightTensors.size() - OUT_LM_HEAD_WEIGHT_COUNT;
    auto &lmHeadNode = graph_.nodes.at(nodeId++);
    atb_speed::common::LmHeadParam lmHeadParam;
    lmHeadParam.unpadInputs = !param_.isFA;
    lmHeadParam.gatherAhead = param_.isPrefill;
    lmHeadParam.linearParallelParam.fusionLinearParam.quantType = false; // LmHead未接入量化
    if (param_.rankSize > 1) {
        lmHeadParam.linearParallelParam.parallelType = atb_speed::common::COLUMN_PARALLEL;
        lmHeadParam.linearParallelParam.rank = param_.rank;
        lmHeadParam.linearParallelParam.worldSize = param_.rankSize;
        lmHeadParam.linearParallelParam.backend = param_.backend;
    }
    LmHead(lmHeadParam, &op);
    lmHeadNode.operation.reset(op);
    lmHeadNode.inTensors = {&graph_.internalTensors.at(finalLayerNormOutTensorId),
                            // shape: [vocabSizePerRank, hiddenSize]
                            &graph_.weightTensors.at(finalLinearWeightTensorId),
                            // LmHead未接入量化，量化权重使用placeholder代替
                            &graph_.inTensors.at(IN_HOLDER), &graph_.inTensors.at(IN_HOLDER),
                            &graph_.inTensors.at(IN_HOLDER), &graph_.inTensors.at(IN_FINAL_NORM_SLICE_OFFSET)};
    // shape: FA: [batchSize, seqLen, vocabSize] PA: [seqLen, vocabSize]
    lmHeadNode.outTensors = {&graph_.outTensors.at(OUT_TENSOR_HIDDENSTATES)};

    return atb::NO_ERROR;
} // 构建图

// flashattention
atb::Status FlashAttentionModel::ParseParam(const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);
    tokenOffset_.clear();
    for (auto item : paramJson["tokenOffset"]) {
        tokenOffset_.push_back(item.get<int>());
    }
    seqLen_.clear();
    for (auto item : paramJson["seqLen"]) {
        seqLen_.push_back(item.get<int>());
    }
    return atb::NO_ERROR;
}

atb::Status FlashAttentionModel::BindParamHostTensor(uint32_t nodeId)
{
    if (nodeId < OPERATION_COUNT_BEFORE_LAYER || nodeId >= OPERATION_COUNT_BEFORE_LAYER + param_.layerNum) {
        return atb::NO_ERROR;
    }
    auto &node = graph_.nodes.at(nodeId);

    node.variantPack.inTensors.at(FA_ROPE_LAYER_IN_TOKENOFFSET_ID).hostData = tokenOffset_.data();
    node.variantPack.inTensors.at(FA_ROPE_LAYER_IN_SEQLEN_ID).hostData = seqLen_.data();
    ATB_LOG(INFO) << "BindParamHostTensor end";
    return atb::NO_ERROR;
}
} // namespace qwen_14b
} // namespace atb_speed