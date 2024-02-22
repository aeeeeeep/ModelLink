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
#include "layer.h"
#include "models/llama_adapter/operation/self_attention_cross.h"
#include "models/llama_adapter/operation/apply_rotary_emb.h"
#include "models/llama_adapter/operation/mlp.h"

namespace atb_speed {
namespace llama_adapter {
enum LayerDecoderAdapterTensorId : int {
    IN_HIDDENSTATES = 0,
    IN_NORMWEIGHT,
    IN_QWEIGHT,
    IN_QBIAS,
    IN_KWEIGHT,
    IN_VWEIGHT,
    IN_GATETANH,
    IN_OWEIGHT,
    IN_OBIAS,
    IN_SELFOUTNORMWEIGHT,
    IN_MLPW1WEIGHT,
    IN_MLPW1BIAS,
    IN_MLPW2WEIGHT,
    IN_MLPW2BIAS,
    IN_MLPW3WEIGHT,
    IN_MLPW3BIAS,
    IN_FREQSCIS,
    IN_PASTKEY,
    IN_PASTVALUE,
    IN_ADAPTER,
    IN_ATTENTIONMASK,
    OUT_LLAMA7BLAYEROUT,
    OUT_PRESENTKEY,
    OUT_PRESENTVALUE,
    INTERMIDATE_INPUTNORMOUT,
    INTERMIDATE_Q,
    INTERMIDATE_K,
    INTERMIDATE_V,
    INTERMIDATE_ROPEQ,
    INTERMIDATE_ROPEK,
    INTERMIDATE_AV,
    INTERMIDATE_AK,
    INTERMIDATE_SELFOUT,
    INTERMIDATE_SELFLINEAROUT,
    INTERMIDATE_SELFRESIDUALADDOUT,
    INTERMIDATE_SELFNORMOUT,
    INTERMIDATE_MLPOUT,
};

static const uint64_t IN_TENSOR_COUNT = 21;
static const uint64_t OUT_TENSOR_COUNT = 3;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 13;
static const uint64_t NODE_COUNT = 13;

atb::Status DecoderAdapterLayer(const LayerParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.name = "llama_adapter_decoder_adapter_layer";
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    atb::Node &inputNormNode = opGraph.nodes.at(nodeId++);
    atb::Node &wQLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &wKLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &wVLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &ropeNode = opGraph.nodes.at(nodeId++);
    atb::Node &wAVLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &wAKLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfAttentionNode = opGraph.nodes.at(nodeId++);
    atb::Node &wOLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfResidualAddNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfNormNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpResidualAddNode = opGraph.nodes.at(nodeId++);

    atb::infer::RmsNormParam rmsNormParam;
    rmsNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    rmsNormParam.normParam.epsilon = param.rmsNormEps;
    CREATE_OPERATION(rmsNormParam, &inputNormNode.operation);
    inputNormNode.inTensorIds = { IN_HIDDENSTATES, IN_NORMWEIGHT };
    inputNormNode.outTensorIds = { INTERMIDATE_INPUTNORMOUT };

    atb::infer::LinearParam linearBiasParam;
    CREATE_OPERATION(linearBiasParam, &wQLinearNode.operation);
    wQLinearNode.inTensorIds = { INTERMIDATE_INPUTNORMOUT, IN_QWEIGHT, IN_QBIAS };
    wQLinearNode.outTensorIds = { INTERMIDATE_Q };

    atb::infer::LinearParam linearKParam;
    linearKParam.hasBias = false;
    CREATE_OPERATION(linearKParam, &wKLinearNode.operation);
    wKLinearNode.inTensorIds = { INTERMIDATE_INPUTNORMOUT, IN_KWEIGHT };
    wKLinearNode.outTensorIds = { INTERMIDATE_K };

    atb::infer::LinearParam linearVParam;
    linearVParam.hasBias = false;
    CREATE_OPERATION(linearVParam, &wVLinearNode.operation);
    wVLinearNode.inTensorIds = { INTERMIDATE_INPUTNORMOUT, IN_VWEIGHT };
    wVLinearNode.outTensorIds = { INTERMIDATE_V };

    atb_speed::llama_adapter::ApplyRotaryEmbParam ropeParam;
    ropeParam.model = param.model;
    atb_speed::llama_adapter::ApplyRotaryEmb(ropeParam, &ropeNode.operation);
    ropeNode.inTensorIds = { INTERMIDATE_Q, INTERMIDATE_K, IN_FREQSCIS };
    ropeNode.outTensorIds = { INTERMIDATE_ROPEQ, INTERMIDATE_ROPEK };
    ropeNode.inTensorReshapeFuncs.resize(ropeNode.inTensorIds.size());
    ropeNode.inTensorReshapeFuncs.at(0) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 4;
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = oldShape.dims[1];
        newShape.dims[2] = param.headNum;
        newShape.dims[3] = oldShape.dims[2] / param.headNum;
    };
    ropeNode.inTensorReshapeFuncs.at(1) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 4;
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = oldShape.dims[1];
        newShape.dims[2] = param.headNum;
        newShape.dims[3] = oldShape.dims[2] / param.headNum;
    };

    atb::infer::LinearParam linearAParam;
    linearAParam.hasBias = false;
    CREATE_OPERATION(linearAParam, &wAVLinearNode.operation);
    wAVLinearNode.inTensorIds = { IN_ADAPTER, IN_VWEIGHT };
    wAVLinearNode.outTensorIds = { INTERMIDATE_AV };

    CREATE_OPERATION(linearAParam, &wAKLinearNode.operation);
    wAKLinearNode.inTensorIds = { IN_ADAPTER, IN_KWEIGHT };
    wAKLinearNode.outTensorIds = { INTERMIDATE_AK };

    atb_speed::llama_adapter::SelfAttentionCrossParam selfAttentionCrossParam;
    selfAttentionCrossParam.dk = param.dk;
    selfAttentionCrossParam.headNum = param.headNum;
    selfAttentionCrossParam.model = param.model;
    atb_speed::llama_adapter::SelfAttentionCrossDeAdapter(selfAttentionCrossParam, &selfAttentionNode.operation);
    selfAttentionNode.inTensorIds = { INTERMIDATE_ROPEQ,
                                      INTERMIDATE_ROPEK,
                                      INTERMIDATE_V,
                                      IN_PASTKEY,
                                      IN_PASTVALUE,
                                      INTERMIDATE_AV,
                                      INTERMIDATE_AK,
                                      IN_GATETANH,
                                      IN_ATTENTIONMASK };
    selfAttentionNode.outTensorIds = { INTERMIDATE_SELFOUT, OUT_PRESENTKEY, OUT_PRESENTVALUE };

    selfAttentionNode.inTensorReshapeFuncs.resize(selfAttentionNode.inTensorIds.size());
    selfAttentionNode.inTensorReshapeFuncs.at(2) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 4;
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = oldShape.dims[1];
        newShape.dims[2] = param.headNum;
        newShape.dims[3] = oldShape.dims[2] / param.headNum;
    };
    selfAttentionNode.inTensorReshapeFuncs.at(5) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 4;
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = oldShape.dims[1];
        newShape.dims[2] = param.headNum;
        newShape.dims[3] = oldShape.dims[2] / param.headNum;
    };
    selfAttentionNode.inTensorReshapeFuncs.at(6) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 4;
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = oldShape.dims[1];
        newShape.dims[2] = param.headNum;
        newShape.dims[3] = oldShape.dims[2] / param.headNum;
    };

    CREATE_OPERATION(linearBiasParam, &wOLinearNode.operation);
    wOLinearNode.inTensorIds = { INTERMIDATE_SELFOUT, IN_OWEIGHT, IN_OBIAS };
    wOLinearNode.outTensorIds = { INTERMIDATE_SELFLINEAROUT };

    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CREATE_OPERATION(addParam, &selfResidualAddNode.operation);
    selfResidualAddNode.inTensorIds = { IN_HIDDENSTATES, INTERMIDATE_SELFLINEAROUT };
    selfResidualAddNode.outTensorIds = { INTERMIDATE_SELFRESIDUALADDOUT };

    CREATE_OPERATION(rmsNormParam, &selfNormNode.operation);
    selfNormNode.inTensorIds = { INTERMIDATE_SELFRESIDUALADDOUT, IN_SELFOUTNORMWEIGHT };
    selfNormNode.outTensorIds = { INTERMIDATE_SELFNORMOUT };

    atb_speed::llama_adapter::MlpParam mlpParam;
    mlpParam.model = param.model;
    atb_speed::llama_adapter::MlpAdapter(mlpParam, &mlpNode.operation);
    mlpNode.inTensorIds = { INTERMIDATE_SELFNORMOUT,
                            IN_MLPW1WEIGHT, IN_MLPW1BIAS,
                            IN_MLPW2WEIGHT, IN_MLPW2BIAS,
                            IN_MLPW3WEIGHT, IN_MLPW3BIAS };
    mlpNode.outTensorIds = { INTERMIDATE_MLPOUT };

    CREATE_OPERATION(addParam, &mlpResidualAddNode.operation);
    mlpResidualAddNode.inTensorIds = { INTERMIDATE_SELFRESIDUALADDOUT, INTERMIDATE_MLPOUT };
    mlpResidualAddNode.outTensorIds = { OUT_LLAMA7BLAYEROUT };

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
        atb::SVector<atb::TensorDesc> &outTensorDescs) {
        const atb::TensorDesc &keyTensorDesc = inTensorDescs.at(IN_PASTKEY);
        const atb::TensorDesc &valueTensorDesc = inTensorDescs.at(IN_PASTVALUE);
        outTensorDescs.at(0) = inTensorDescs.at(0);
        outTensorDescs.at(1) = keyTensorDesc;
        outTensorDescs.at(1).shape.dims[1] += 1;
        outTensorDescs.at(2) = valueTensorDesc;
        outTensorDescs.at(2).shape.dims[1] += 1;
        return atb::NO_ERROR;
    };

    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}
} // namespace llama_adapter
} // namespace atb_speed