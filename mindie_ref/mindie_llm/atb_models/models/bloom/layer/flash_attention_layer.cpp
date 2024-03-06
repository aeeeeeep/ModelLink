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
#include "layers/mlp_gate_v2.h"
#include "layers/parallel_layer_v2.h"
#include "models/bloom/layer/flash_attention_layer.h"
#include "layers/plugin_op/w8a16_bias_operation.h"

namespace atb_speed {
namespace bloom_7b {

enum Bloom7BLayerInTensorId : int {
    IN_NORM_WEIGHT = 0,            // 0
    IN_NORM_BIAS,                  // 1

    IN_QKVMIXED_WEIGHT,            // 2
    IN_QKVMIXED_BIAS,              // 3
    IN_QKVMIXED_DEQSCALE,          // 4
    IN_QKVMIXED_OFFSET,            // 5
    IN_QKVMIXED_INPUT_SCALE,       // 6
    IN_QKVMIXED_INPUT_OFFSET,      // 7

    IN_SELFOUTLINEAR_WEIGHT,       // 8
    IN_SELFOUTLINEAR_BIAS,         // 9
    IN_SELFOUTLINEAR_DEQSCALE,     // 10
    IN_SELFOUTLINEAR_OFFSET,       // 11
    IN_SELFOUTLINEAR_INPUT_SCALE,  // 12
    IN_SELFOUTLINEAR_INPUT_OFFSET, // 13

    IN_SELFOUTNORM_WEIGHT,     // 14
    IN_SELFOUTNORM_BIAS,       // 15

    IN_HTO4H_WEIGHT,           // 16
    IN_HTO4H_BIAS,             // 17
    IN_HTO4H_DEQSCALE,         // 18
    IN_HTO4H_OFFSET,           // 19
    IN_HTO4H_INPUT_SCALE,      // 20
    IN_HTO4H_INPUT_OFFSET,     // 21

    IN_4HTOH_WEIGHT,           // 22
    IN_4HTOH_BIAS,             // 23
    IN_4HTOH_DEQSCALE,         // 24
    IN_4HTOH_OFFSET,           // 25
    IN_4HTOH_INPUT_SCALE,      // 26
    IN_4HTOH_INPUT_OFFSET,     // 27

    IN_HIDDEN_STATES,          // 28
    IN_ATTENTION_MASK,         // 29
    IN_CACHED_K,               // 30
    IN_CACHED_V,               // 31
    IN_TOKENOFFSET,            // 32
    IN_SEQLEN,                 // 33
    IN_PLACE_HOLDER,           // 34
    IN_LAYERID,                // 35

    IN_TENSOR_MAX              // 36
};

enum Bloom7BLayerOutTensorId : int {
    OUT_LAYEROUT = IN_TENSOR_MAX, // 36
    OUT_TENSOR_MAX                // 37
};

enum Bloom7BLayerINTERMEDIATETensorId : int {
    INTERMEDIATE_INPUTNORM_OUT = OUT_TENSOR_MAX,  // 37
    INTERMEDIATE_MIXEDLINEAROUTQKV,               // 38
    INTERMEDIATE_QUERY,                           // 39
    INTERMEDIATE_KEY,                             // 40
    INTERMEDIATE_VALUE,                           // 41

    INTERMEDIATE_SELFOUT,                         // 42
    INTERMEDIATE_SELFLINEAROUT,                   // 43
    INTERMEDIATE_SELFADDOUT,                      // 44
    INTERMEDIATE_SELFNORMOUT,                     // 45
    INTERMEDIATE_MLPOUT,                          // 46

    INTERMEDIATE_SELFLINEAROUT_BEFOREREDUCE,      // w8a16
    INTERMEDIATE_MATMULUP,                        // w8a16
    INTERMEDIATE_ACTIVATION_OUT,                  // w8a16
    INTERMEDIATE_MATMULDOWN,                      // w8a16

    INTERMEDIATE_TENSOR_MAX
};

static const uint64_t IN_TENSOR_COUNT = IN_TENSOR_MAX;
static const uint64_t OUT_TENSOR_COUNT = OUT_TENSOR_MAX - IN_TENSOR_MAX;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = INTERMEDIATE_TENSOR_MAX - OUT_TENSOR_MAX;
static const uint64_t HIDDEN_STATES_DIM = 3;

void SqueezeThirdDim(const atb::Dims &oldShape, atb::Dims &newShape)
{
    newShape.dimNum = oldShape.dimNum - 1;
    std::copy(std::begin(oldShape.dims), std::end(oldShape.dims), std::begin(newShape.dims));
    newShape.dims[newShape.dimNum - 1] = oldShape.dims[oldShape.dimNum - 1]; // squeeze second last dim
}


atb::Status CommomLayer(const Bloom7bCommonLayerParam &param, atb::Operation **operation)
{
    ATB_LOG(INFO) << "[+] Enter bloom model layer...  IN_TENSOR_COUNT: " << IN_TENSOR_COUNT; // 24
    ATB_LOG(INFO) << "[+] param.quantMode: " << param.quantMode;
    
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    if (param.quantMode == 0) {         // fp16
        opGraph.nodes.resize(9);
        opGraph.internalTensorNum -= 4;
    } else if (param.quantMode == 1) {  // w8a8
        opGraph.nodes.resize(9);
        opGraph.internalTensorNum -= 4;
    } else if (param.quantMode == 2) {  // w8a16
        opGraph.nodes.resize(13);
    }

    ATB_LOG(INFO) << "[+] Bloom layer: inTensorNum: " << opGraph.inTensorNum;
    ATB_LOG(INFO) << "[+] Bloom layer: outTensorNum: " << opGraph.outTensorNum;
    ATB_LOG(INFO) << "[+] Bloom layer: internalTensorNum: " << opGraph.internalTensorNum;
    
    opGraph.name = "Bloom7bCommonLayer";

    size_t nodeId = 0;

    atb::Node &inputNormNode = opGraph.nodes.at(nodeId++);
    atb::infer::LayerNormParam layerNormQuantParam;
    layerNormQuantParam.layerType = atb::infer::LayerNormParam::LAYER_NORM_NORM;
    const int32_t beginParamsAxis = 2;
    layerNormQuantParam.normParam.epsilon = param.layerNormEps;
    layerNormQuantParam.normParam.beginNormAxis = beginParamsAxis;
    layerNormQuantParam.normParam.beginParamsAxis = 1;
    if (param.quantMode == 1) {
        layerNormQuantParam.normParam.quantType = atb::infer::QUANT_INT8;
    }
    ATB_LOG(INFO) << "[+] Bloom layer: LayerNormParam ";
    CREATE_OPERATION(layerNormQuantParam, &inputNormNode.operation);
    inputNormNode.inTensorIds = {IN_HIDDEN_STATES, IN_NORM_WEIGHT, IN_NORM_BIAS};
    inputNormNode.outTensorIds = {INTERMEDIATE_INPUTNORM_OUT};

    ATB_LOG(INFO) << "[+] Bloom layer: inputNormNode ";

    atb::Node &mixdQkvLinearNode = opGraph.nodes.at(nodeId++);
    if (param.quantMode == 1) {
        atb::infer::LinearParam mixdQkvLinearParam;
        mixdQkvLinearParam.linearType = atb::infer::LINEAR_INT8INT8_INT32_FP16;
        CREATE_OPERATION(mixdQkvLinearParam, &mixdQkvLinearNode.operation);
        mixdQkvLinearNode.inTensorIds = {
            INTERMEDIATE_INPUTNORM_OUT, IN_QKVMIXED_WEIGHT, IN_QKVMIXED_BIAS, IN_QKVMIXED_DEQSCALE};
    } else if (param.quantMode == 2) {
        mixdQkvLinearNode.operation = new atb_speed::common::W8A16BiasOperation("mixdQkvLinearNode");
        // INPUT, WEIGHT, SCALE, OFFSET, BIAS
        mixdQkvLinearNode.inTensorIds = {INTERMEDIATE_INPUTNORM_OUT, IN_QKVMIXED_WEIGHT,
                                         IN_QKVMIXED_DEQSCALE, IN_QKVMIXED_OFFSET, IN_QKVMIXED_BIAS};
    } else {
        atb::infer::LinearParam mixdQkvLinearParam;
        CREATE_OPERATION(mixdQkvLinearParam, &mixdQkvLinearNode.operation);
        mixdQkvLinearNode.inTensorIds = {
            INTERMEDIATE_INPUTNORM_OUT, IN_QKVMIXED_WEIGHT, IN_QKVMIXED_BIAS};
    }
    mixdQkvLinearNode.outTensorIds = {INTERMEDIATE_MIXEDLINEAROUTQKV};

    ATB_LOG(INFO) << "[+] Bloom layer: mixdQkvLinearNode ";

    atb::Node &splitNode = opGraph.nodes.at(nodeId++);
    atb::infer::SplitParam splitParam = {3, 3};
    CREATE_OPERATION(splitParam, &splitNode.operation);
    splitNode.inTensorIds = {INTERMEDIATE_MIXEDLINEAROUTQKV};
    splitNode.outTensorIds = {INTERMEDIATE_QUERY, INTERMEDIATE_KEY, INTERMEDIATE_VALUE};
    splitNode.inTensorReshapeFuncs.resize(splitNode.inTensorIds.size());
    splitNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        size_t dim = 0;
        newShape.dims[dim++] = oldShape.dims[0]; // batch
        newShape.dims[dim++] = oldShape.dims[1]; // seq_len
        newShape.dims[dim++] = param.headNum;    // head_num
        newShape.dims[dim++] = 3;                // 3 -> q, k, v
        newShape.dims[dim++] = param.dk;         // dk
        newShape.dimNum = dim;                   // [batch, seq_len, head_num, 3, dk]
    };

    atb::Node &selfAttentionFusionNode = opGraph.nodes.at(nodeId++);
    atb::infer::SelfAttentionParam selfAttentionParam;
    selfAttentionParam.headNum = param.headNum;
    selfAttentionParam.qScale = 1.0f / std::sqrt(param.dk);
    selfAttentionParam.qkScale = 1.0f;
    selfAttentionParam.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_ALIBI;
    CREATE_OPERATION(selfAttentionParam, &selfAttentionFusionNode.operation);
    selfAttentionFusionNode.inTensorIds = {
        INTERMEDIATE_QUERY, INTERMEDIATE_KEY, INTERMEDIATE_VALUE, IN_CACHED_K,
        IN_CACHED_V, IN_ATTENTION_MASK, IN_TOKENOFFSET, IN_SEQLEN, IN_LAYERID
        };
    selfAttentionFusionNode.outTensorIds = {INTERMEDIATE_SELFOUT};
    selfAttentionFusionNode.inTensorReshapeFuncs.resize(selfAttentionFusionNode.inTensorIds.size());
    selfAttentionFusionNode.inTensorReshapeFuncs[0] = &SqueezeThirdDim;
    selfAttentionFusionNode.inTensorReshapeFuncs[1] = &SqueezeThirdDim;
    selfAttentionFusionNode.inTensorReshapeFuncs[2] = &SqueezeThirdDim;

    atb::Node &selfOutLinearNode = opGraph.nodes.at(nodeId++);
    if (param.quantMode == 1) {
        atb_speed::common::ParallelParamV2 selfOutLinearParam;
        selfOutLinearParam.commParam.rank = param.rank;
        selfOutLinearParam.commParam.rankSize = param.rankSize;
        selfOutLinearParam.isBias = true;
        selfOutLinearParam.isQuant = true;
        selfOutLinearParam.transposeB = true;
        selfOutLinearParam.quantParam.quantType = atb::infer::QUANT_INT8;
        selfOutLinearParam.quantParam.isQuantOp = true;
        selfOutLinearParam.quantParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_QUANT;
        selfOutLinearParam.quantParam.inputScale = param.denseInputScale;
        selfOutLinearParam.quantParam.inputOffset = param.denseInputOffset;
        atb_speed::common::RowParallelLinearV2(selfOutLinearParam, &selfOutLinearNode.operation);
        selfOutLinearNode.inTensorIds = {INTERMEDIATE_SELFOUT, IN_SELFOUTLINEAR_WEIGHT,
                                        IN_SELFOUTLINEAR_BIAS, IN_SELFOUTLINEAR_DEQSCALE,
                                        IN_PLACE_HOLDER, IN_PLACE_HOLDER, IN_PLACE_HOLDER};
        selfOutLinearNode.outTensorIds = {INTERMEDIATE_SELFLINEAROUT};
    } else if (param.quantMode == 2) {
        selfOutLinearNode.operation = new atb_speed::common::W8A16BiasOperation("selfOutLinearNode");
        selfOutLinearNode.inTensorIds = {INTERMEDIATE_SELFOUT, IN_SELFOUTLINEAR_WEIGHT, IN_SELFOUTLINEAR_DEQSCALE,
                                         IN_SELFOUTLINEAR_OFFSET, IN_SELFOUTLINEAR_BIAS};
        selfOutLinearNode.outTensorIds = {INTERMEDIATE_SELFLINEAROUT_BEFOREREDUCE};
        
        atb::Node &selfOutLinearAllReduceNode = opGraph.nodes.at(nodeId++);
        atb::infer::AllReduceParam selfOutLinearAllReduceParam;
        selfOutLinearAllReduceParam.rank = param.rank;
        selfOutLinearAllReduceParam.rankSize = param.rankSize;
        selfOutLinearAllReduceParam.backend = param.backend;
        CREATE_OPERATION(selfOutLinearAllReduceParam, &selfOutLinearAllReduceNode.operation);
        selfOutLinearAllReduceNode.inTensorIds = {INTERMEDIATE_SELFLINEAROUT_BEFOREREDUCE};
        selfOutLinearAllReduceNode.outTensorIds = {INTERMEDIATE_SELFLINEAROUT};
    } else {
        atb_speed::common::ParallelParamV2 selfOutLinearParam;
        selfOutLinearParam.commParam.rank = param.rank;
        selfOutLinearParam.commParam.rankSize = param.rankSize;
        selfOutLinearParam.isBias = true;
        selfOutLinearParam.transposeB = true;
        atb_speed::common::RowParallelLinearV2(selfOutLinearParam, &selfOutLinearNode.operation);
        selfOutLinearNode.inTensorIds = {INTERMEDIATE_SELFOUT, IN_SELFOUTLINEAR_WEIGHT,
                                        IN_SELFOUTLINEAR_BIAS, IN_PLACE_HOLDER, IN_PLACE_HOLDER,
                                        IN_PLACE_HOLDER, IN_PLACE_HOLDER};
        selfOutLinearNode.outTensorIds = {INTERMEDIATE_SELFLINEAROUT};
        ATB_LOG(INFO) << "RowParallelLinearV2 " << selfOutLinearParam.commParam.rankSize<< "-";
        ATB_LOG(INFO) << selfOutLinearParam.transposeB;
    }

    atb::Node &selfOutAddNode = opGraph.nodes.at(nodeId++);
    atb::infer::ElewiseParam selfAddParam;
    selfAddParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CREATE_OPERATION(selfAddParam, &selfOutAddNode.operation);
    selfOutAddNode.inTensorIds = {INTERMEDIATE_SELFLINEAROUT, IN_HIDDEN_STATES};
    selfOutAddNode.outTensorIds = {INTERMEDIATE_SELFADDOUT};

    atb::Node &selfNormNode = opGraph.nodes.at(nodeId++);
    if (param.quantMode == 1) {
        atb::infer::LayerNormParam selfNormParam;
        selfNormParam.layerType = atb::infer::LayerNormParam::LAYER_NORM_NORM;
        selfNormParam.normParam.quantType = atb::infer::QUANT_INT8;
        selfNormParam.normParam.epsilon = param.layerNormEps;
        selfNormParam.normParam.beginNormAxis = beginParamsAxis;
        selfNormParam.normParam.beginParamsAxis = 1;
        CREATE_OPERATION(selfNormParam, &selfNormNode.operation);
        selfNormNode.inTensorIds = {INTERMEDIATE_SELFADDOUT, IN_SELFOUTNORM_WEIGHT, IN_SELFOUTNORM_BIAS};
        selfNormNode.outTensorIds = {INTERMEDIATE_SELFNORMOUT};

        atb::Node &mlpNode = opGraph.nodes.at(nodeId++);
        atb_speed::common::MlpGateParamV2 mlpParam;
        mlpParam.isBias = true;
        mlpParam.isPack = false;
        mlpParam.isQuant = true;
        mlpParam.noGate = true;
        mlpParam.transposeB = true;
        mlpParam.commDownParam.rank = param.rank;
        mlpParam.commDownParam.rankSize = param.rankSize;
        mlpParam.quantUpParam.quantType = atb::infer::QUANT_INT8;
        mlpParam.quantUpParam.isQuantOp = false;
        mlpParam.activationType = atb::infer::ActivationType::ACTIVATION_GELU;
        mlpParam.quantDownParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_QUANT;
        mlpParam.quantDownParam.inputScale = param.ffnOutInputScale;
        mlpParam.quantDownParam.inputOffset = param.ffnOutInputOffset;
        atb_speed::common::MlpGateLayerV2(mlpParam, &mlpNode.operation);
        mlpNode.inTensorIds = {INTERMEDIATE_SELFNORMOUT, IN_HTO4H_WEIGHT, IN_PLACE_HOLDER, IN_4HTOH_WEIGHT,
                            IN_HTO4H_DEQSCALE, IN_PLACE_HOLDER, IN_4HTOH_DEQSCALE, IN_HTO4H_BIAS,
                            IN_PLACE_HOLDER, IN_4HTOH_BIAS, IN_PLACE_HOLDER, IN_PLACE_HOLDER, IN_PLACE_HOLDER,
                            IN_PLACE_HOLDER, IN_PLACE_HOLDER, IN_PLACE_HOLDER, IN_PLACE_HOLDER, IN_PLACE_HOLDER,
                            IN_PLACE_HOLDER};
        mlpNode.outTensorIds = {INTERMEDIATE_MLPOUT};
    } else if (param.quantMode == 2) {
        atb::infer::LayerNormParam selfNormParam;
        selfNormParam.layerType = atb::infer::LayerNormParam::LAYER_NORM_NORM;
        selfNormParam.normParam.epsilon = param.layerNormEps;
        selfNormParam.normParam.beginNormAxis = beginParamsAxis;
        selfNormParam.normParam.beginParamsAxis = 1;
        CREATE_OPERATION(selfNormParam, &selfNormNode.operation);
        selfNormNode.inTensorIds = {INTERMEDIATE_SELFADDOUT, IN_SELFOUTNORM_WEIGHT, IN_SELFOUTNORM_BIAS};
        selfNormNode.outTensorIds = {INTERMEDIATE_SELFNORMOUT};

        // ******************** W8A16 Quant + Parallel  MLP ********************
        // Node: Matmul up h -> 4h
        atb::Node &matmulUpNode = opGraph.nodes.at(nodeId++);
        matmulUpNode.operation = new atb_speed::common::W8A16BiasOperation("matmulUpNode");
        matmulUpNode.inTensorIds = {INTERMEDIATE_SELFNORMOUT, IN_HTO4H_WEIGHT, IN_HTO4H_DEQSCALE, IN_HTO4H_OFFSET, IN_HTO4H_BIAS};
        matmulUpNode.outTensorIds = {INTERMEDIATE_MATMULUP};

        // Node: gelu activation
        atb::Node &actNode = opGraph.nodes.at(nodeId++);
        atb::infer::ActivationParam actParam;
        actParam.activationType = atb::infer::ActivationType::ACTIVATION_GELU;
        CREATE_OPERATION(actParam, &actNode.operation);
        actNode.inTensorIds = {INTERMEDIATE_MATMULUP};
        actNode.outTensorIds = {INTERMEDIATE_ACTIVATION_OUT};

        // Node: Matmul down 4h -> h
        atb::Node &matmulDownNode = opGraph.nodes.at(nodeId++);
        matmulDownNode.operation = new atb_speed::common::W8A16BiasOperation("matmulDownNode");
        matmulDownNode.inTensorIds = {INTERMEDIATE_ACTIVATION_OUT, IN_4HTOH_WEIGHT, IN_4HTOH_DEQSCALE, IN_4HTOH_OFFSET, IN_4HTOH_BIAS};
        matmulDownNode.outTensorIds = {INTERMEDIATE_MATMULDOWN};

        // Node: Matmul down 4h -> h  all reduce
        atb::Node &matmulDownAllReduceNode = opGraph.nodes.at(nodeId++);
        atb::infer::AllReduceParam matmulDownAllReduceParam;
        matmulDownAllReduceParam.rank = param.rank;
        matmulDownAllReduceParam.rankSize = param.rankSize;
        matmulDownAllReduceParam.backend = param.backend;
        CREATE_OPERATION(matmulDownAllReduceParam, &matmulDownAllReduceNode.operation);
        matmulDownAllReduceNode.inTensorIds = {INTERMEDIATE_MATMULDOWN};
        matmulDownAllReduceNode.outTensorIds = {INTERMEDIATE_MLPOUT};
        // ******************** W8A16 Quant + Parallel  MLP  End ********************
    } else {
        atb::infer::LayerNormParam selfNormParam;
        selfNormParam.layerType = atb::infer::LayerNormParam::LAYER_NORM_NORM;
        selfNormParam.normParam.epsilon = param.layerNormEps;
        selfNormParam.normParam.beginNormAxis = beginParamsAxis;
        selfNormParam.normParam.beginParamsAxis = 1;
        CREATE_OPERATION(selfNormParam, &selfNormNode.operation);
        selfNormNode.inTensorIds = {INTERMEDIATE_SELFADDOUT, IN_SELFOUTNORM_WEIGHT, IN_SELFOUTNORM_BIAS};
        selfNormNode.outTensorIds = {INTERMEDIATE_SELFNORMOUT};

        atb::Node &mlpNode = opGraph.nodes.at(nodeId++);
        atb_speed::common::MlpGateParamV2 mlpParam;
        mlpParam.commDownParam.rank = param.rank;
        mlpParam.commDownParam.rankSize = param.rankSize;
        mlpParam.activationType = atb::infer::ActivationType::ACTIVATION_GELU;
        mlpParam.transposeB = true;
        mlpParam.isBias = true;
        mlpParam.noGate = true;
        mlpParam.isPack = false;
        atb_speed::common::MlpGateLayerV2(mlpParam, &mlpNode.operation);
        mlpNode.inTensorIds = {INTERMEDIATE_SELFNORMOUT, IN_HTO4H_WEIGHT, IN_PLACE_HOLDER, IN_4HTOH_WEIGHT,
                            IN_PLACE_HOLDER, IN_PLACE_HOLDER, IN_PLACE_HOLDER, IN_HTO4H_BIAS, IN_PLACE_HOLDER,
                            IN_4HTOH_BIAS, IN_PLACE_HOLDER, IN_PLACE_HOLDER, IN_PLACE_HOLDER, IN_PLACE_HOLDER,
                            IN_PLACE_HOLDER, IN_PLACE_HOLDER, IN_PLACE_HOLDER, IN_PLACE_HOLDER, IN_PLACE_HOLDER};
        mlpNode.outTensorIds = {INTERMEDIATE_MLPOUT};
    }

    atb::Node &mlpResidualAddNode = opGraph.nodes.at(nodeId++);
    atb::infer::ElewiseParam mlpResidualAddParam;
    mlpResidualAddParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CREATE_OPERATION(mlpResidualAddParam, &mlpResidualAddNode.operation);
    mlpResidualAddNode.inTensorIds = {INTERMEDIATE_MLPOUT, INTERMEDIATE_SELFADDOUT};
    mlpResidualAddNode.outTensorIds = {OUT_LAYEROUT};
    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        size_t dim = 0;
        outTensorDescs.at(dim++) = inTensorDescs.at(IN_HIDDEN_STATES);
        return atb::NO_ERROR;
    };

    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}
} // namespace bloom_7b
} // namespace atb_speed