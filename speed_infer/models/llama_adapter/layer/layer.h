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
#ifndef ATB_SPEED_MODELS_LLAMA_ADAPTER_LAYER_H
#define ATB_SPEED_MODELS_LLAMA_ADAPTER_LAYER_H

#include "atb/atb_infer.h"
#include "atb_speed/log.h"
#include "nlohmann/json.hpp"

namespace atb_speed {
namespace llama_adapter {
struct LayerParam {
    float rmsNormEps = 0;
    int headNum = 0;
    int dk = 0;
    std::string model = "llama7b_adapter";
};

atb::Status EncoderLayer(const LayerParam &param, atb::Operation **operation);

atb::Status EncoderAdapterLayer(const LayerParam &param, atb::Operation **operation);

atb::Status DecoderLayer(const LayerParam &param, atb::Operation **operation);

atb::Status DecoderAdapterLayer(const LayerParam &param, atb::Operation **operation);

static atb::Operation *CreateEncoderLayer(const nlohmann::json &paramJson)
{
    LayerParam param;
    param.rmsNormEps = paramJson["rmsNormEps"].get<float>();
    param.headNum = paramJson["headNum"].get<int>();
    param.dk = paramJson["dk"].get<int>();
    param.model = paramJson["model"].get<std::string>();
    ATB_LOG(INFO) << "LLaMAAdapterLayerEncoder rmsNormEps:" << param.rmsNormEps << ", headNum:" << param.headNum <<
        ", dk:" << param.dk << ", model:" << param.model;
    atb::Operation *op;
    EncoderLayer(param, &op);
    return op;
}

static atb::Operation *CreateEncoderAdapterLayer(const nlohmann::json &paramJson)
{
    LayerParam param;
    param.rmsNormEps = paramJson["rmsNormEps"].get<float>();
    param.headNum = paramJson["headNum"].get<int>();
    param.dk = paramJson["dk"].get<int>();
    param.model = paramJson["model"].get<std::string>();
    ATB_LOG(INFO) << "LLaMAAdapterLayerEncoderAdapter rmsNormEps:" << param.rmsNormEps << ", headNum:" <<
        param.headNum << ", dk:" << param.dk << ", model:" << param.model;
    atb::Operation *op;
    EncoderAdapterLayer(param, &op);
    return op;
}

static atb::Operation *CreateDecoderLayer(const nlohmann::json &paramJson)
{
    LayerParam param;
    param.rmsNormEps = paramJson["rmsNormEps"].get<float>();
    param.headNum = paramJson["headNum"].get<int>();
    param.dk = paramJson["dk"].get<int>();
    param.model = paramJson["model"].get<std::string>();
    ATB_LOG(INFO) << "LLaMAAdapterLayerDecoder rmsNormEps:" << param.rmsNormEps << ", headNum:" << param.headNum <<
        ", dk:" << param.dk << ", model:" << param.model;
    atb::Operation *op;
    DecoderLayer(param, &op);
    return op;
}

static atb::Operation *CreateDecoderAdapterLayer(const nlohmann::json &paramJson)
{
    LayerParam param;
    param.rmsNormEps = paramJson["rmsNormEps"].get<float>();
    param.headNum = paramJson["headNum"].get<int>();
    param.dk = paramJson["dk"].get<int>();
    param.model = paramJson["model"].get<std::string>();
    ATB_LOG(INFO) << "LLaMAAdapterLayerDecoderAdapter rmsNormEps:" << param.rmsNormEps << ", headNum:" <<
        param.headNum << ", dk:" << param.dk << ", model:" << param.model;
    atb::Operation *op;
    DecoderAdapterLayer(param, &op);
    return op;
}
} // namespace llama_7b
} // namespace atb_speed
#endif