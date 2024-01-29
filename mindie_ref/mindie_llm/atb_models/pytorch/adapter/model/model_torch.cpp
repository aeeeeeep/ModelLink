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

#include "model_torch.h"

#include <acl/acl.h>
#include <atb/utils.h>
#include <torch/torch.h>
#include <torch_npu/csrc/framework/OpCommand.h>

#include "atb_speed/base/context_factory.h"
#include "atb_speed/log.h"
#include "atb_speed/utils/config.h"
#include "atb_speed/utils/singleton.h"
#include "atb_speed/utils/statistic.h"
#include "atb_speed/utils/tensor_util.h"
#include "atb_speed/utils/timer.h"
#include "baichuan2/13b/model/flash_attention_model.h"
#include "baichuan2/13b/model/flash_attention_quant_model.h"
#include "baichuan2/13b/model/paged_attention_model.h"
#include "baichuan2/13b/model/paged_attention_quant_model.h"
#include "chatglm2//6b/model/paged_attention_model.h"
#include "llama/model/anti_quant_flashattention_model.h"
#include "llama/model/flash_attention_model.h"
#include "llama_adapter/model/adapter_model.h"
#include "llama_pa/model/paged_attention_model.h"
#include "llama_pa/model/quant_paged_attention_model.h"
#include "llama_parallel/model/decoder_model.h"
#include "pytorch/adapter/utils/utils.h"
#include "pytorch/adapter/workspace/workspace.h"
#include "telechat/model/model.h"
#include "baichuan2/7b/model/flash_attention_quant_model.h"
#include "baichuan2/7b/model/flash_attention_rope_model.h"
#include "baichuan2/7b/model/paged_attention_model.h"
#include "qwen/14b/model/flash_attention_model.h"
#include "aquila/7b/model/flash_attention_model.h"


void *ModelTorch::GetWorkSpace(uint64_t bufferSize)
{
    void *workspace = nullptr;
    if (bufferSize > 0) {
        workspace = atb_speed::GetSingleton<atb_speed::Workspace>().GetWorkspaceBuffer(bufferSize);
    }
    return workspace;
}

atb::Tensor ModelTorch::CreateInternalTensorFromDesc(const atb::TensorDesc &tensorDesc)
{
    torch::Tensor newAtTensor = Utils::CreateAtTensorFromTensorDesc(tensorDesc);
    atInternalTensors_.push_back(newAtTensor);
    return Utils::AtTensor2Tensor(newAtTensor);
}

void ModelTorch::RunTask(std::string taskName, std::function<int()> task)
{
#ifdef TORCH_SETCUSTOMHANDLER
    at_npu::native::OpCommand cmd;
    cmd.Name(taskName);
    cmd.SetCustomHandler(task);
    cmd.Run();
#else
    ATB_LOG(FATAL) << modelName_ << "torch_npu is low, can't support SetCustomHandler";
#endif
}

uint64_t GetNewModelId()
{
    static uint64_t modelId = 0;
    uint64_t newModelId = modelId++;
    return newModelId;
}

ModelTorch::ModelTorch(std::string modelName) : modelName_(modelName)
{
    modelId_ = GetNewModelId();
    context_ = atb_speed::ContextFactory::GetAtbContext(Utils::GetCurrentStream());
    ATB_LOG(INFO) << "ModelTorch new modelName:" << modelName_ << ", modelId:" << modelId_;
}

ModelTorch::~ModelTorch()
{
    context_.reset();
    atb_speed::ContextFactory::FreeAtbContext();
};

int64_t ModelTorch::SetParam(std::string param)
{
    ATB_LOG(INFO) << "ModelTorch set param start, modelName:" << modelName_ << ", param:" << param;
    if (modelName_ == "llama_anti_quant_flashattention_model") {
        model_ = std::make_shared<atb_speed::llama::AntiQuantFlashAttentionModel>(param);
    } else if (modelName_ == "llama_flashattention_model") {
        model_ = std::make_shared<atb_speed::llama::FlashAttentionModel>(param);
    } else if (modelName_ == "llama_pa_model" || modelName_ == "llama_65b_pa_model") {
        model_ = std::make_shared<atb_speed::llama_pa::PAModel>(param);
    } else if (modelName_ == "llama_quant_pa_model") {
        model_ = std::make_shared<atb_speed::llama_pa::QuantPAModel>(param);
    } else if (modelName_ == "llama_parallel_decoder_model") {
        model_ = std::make_shared<atb_speed::llama_parallel::DecoderModel>(param);
    } else if (modelName_ == "llama_adapter_encoder_model") {
        model_ = std::make_shared<atb_speed::llama_adapter::EncoderAdapterModel>(param);
    } else if (modelName_ == "llama_adapter_decoder_model") {
        model_ = std::make_shared<atb_speed::llama_adapter::DecoderAdapterModel>(param);
    } else if (modelName_ == "TelechatQuantFAModel") {
        model_ = std::make_shared<atb_speed::telechat::QuantFAModel>(param);
    } else if (modelName_ == "chatglm2_6b_decoder_pa_model") {
        model_ = std::make_shared<atb_speed::chatglm2_6b::PagedAttentionModel>(param);
    } else if (modelName_ == "baichuan2_13b_flash_attention_model") {
        model_ = std::make_shared<atb_speed::baichuan2_13b::FlashAttentionModel>(param);
    } else if (modelName_ == "baichuan2_13b_flash_attention_quant_model") {
        model_ = std::make_shared<atb_speed::baichuan2_13b::FlashAttentionQuantModel>(param);
    } else if (modelName_ == "baichuan2_13b_pa_model") {
        model_ = std::make_shared<atb_speed::baichuan2_13b::PagedAttentionModel>(param);
    } else if (modelName_ == "baichuan2_13b_pa_quant_model") {
        model_ = std::make_shared<atb_speed::baichuan2_13b::PagedAttentionQuantModel>(param);
    } else if (modelName_ == "minigpt4_vicuna_7b_encoder_model") {
        model_ = std::make_shared<atb_speed::minigpt4_vicuna_7b::FusionEncoderModel>(param);
    } else if (modelName_ == "minigpt4_vicuna_7b_decoder_model") {
        model_ = std::make_shared<atb_speed::minigpt4_vicuna_7b::FusionModel>(param);
    } else if (modelName_ == "baichuan2_7b_flash_attention_rope_model") {
        model_ = std::make_shared<atb_speed::baichuan2_7b::FlashAttentionRopeModel>(param);
    } else if (modelName_ == "baichuan2_7b_flash_attention_quant_model") {
        model_ = std::make_shared<atb_speed::baichuan2_7b::FlashAttentionQuantModel>(param);
    } else if (modelName_ == "baichuan2_7b_pa_model") {
        model_ = std::make_shared<atb_speed::baichuan2_7b::PagedAttentionModel>(param);
    } else if (modelName_ == "qwen_14b_flash_attention_model") {
        model_ = std::make_shared<atb_speed::qwen_14b::FlashAttentionModel>(param);
    } else if (modelName_ == "aquila_7b_flash_attention_model") {
        model_ = std::make_shared<atb_speed::aquila_7b::FlashAttentionRopeModel>(param);
    }  else {
        ATB_LOG(FATAL) << "not support modelName:" << modelName_;
        return atb::ERROR_INVALID_PARAM;
    }

    const char *taskQueueEnv = std::getenv("TASK_QUEUE_ENABLE");
    const char *blockingEnv = std::getenv("ASCEND_LAUNCH_BLOCKING");
    bool isTaskQueueEnable = !((taskQueueEnv != nullptr && std::string(taskQueueEnv) == "0") ||
                               (blockingEnv != nullptr && std::string(blockingEnv) == "1"));
    auto getWorkspaceFunc = std::bind(&ModelTorch::GetWorkSpace, this, std::placeholders::_1);
    auto createInternalTensorFromDescFunc =
        std::bind(&ModelTorch::CreateInternalTensorFromDesc, this, std::placeholders::_1);
    auto runTaskFunc = std::bind(&ModelTorch::RunTask, this, std::placeholders::_1, std::placeholders::_2);
    int64_t atbStatus = 0;
    if (isTaskQueueEnable) {
        atbStatus = model_->Init(getWorkspaceFunc, createInternalTensorFromDescFunc, runTaskFunc);
    } else {
        atbStatus = model_->Init(getWorkspaceFunc, createInternalTensorFromDescFunc, nullptr);
    }
    ATB_LOG(INFO) << "ModelTorch set param end";
    return atbStatus;
}

int64_t ModelTorch::SetWeight(std::vector<torch::Tensor> atWeightTensors)
{
    ATB_LOG(INFO) << "ModelTorch set weight:" << atWeightTensors.size();
    for (size_t i = 0; i < atWeightTensors.size(); ++i) {
        const torch::Tensor &atTensor = atWeightTensors.at(i);
        ATB_LOG(INFO) << "ModelTorch atWeightTensors[" << i << "]"
                      << " data:" << atTensor.data_ptr() << ", storage_offset:" << atTensor.storage_offset()
                      << ", format:" << Utils::GetTensorNpuFormat(atTensor) << ", shape:" << atTensor.sizes()
                      << ", options:" << atTensor.options();
    }
    std::vector<atb::Tensor> weigthTensors;
    AtTensor2Tensor(atWeightTensors, weigthTensors);
    return model_->SetWeight(weigthTensors);
}

int64_t ModelTorch::SetKVCache(std::vector<torch::Tensor> atKCacheTensors, std::vector<torch::Tensor> atVCacheTensors)
{
    ATB_LOG(INFO) << "ModelTorch set k cache tensors:" << atKCacheTensors.size()
                  << ", v cache tensors:" << atVCacheTensors.size();
    for (size_t i = 0; i < atKCacheTensors.size(); ++i) {
        const torch::Tensor &atkTensor = atKCacheTensors.at(i);
        ATB_LOG(INFO) << "ModelTorch atKCacheTensors[" << i << "]"
                      << " data:" << atkTensor.data_ptr() << ", storage_offset:" << atkTensor.storage_offset()
                      << ", format:" << Utils::GetTensorNpuFormat(atkTensor) << ", shape:" << atkTensor.sizes()
                      << ", options:" << atkTensor.options();
        const torch::Tensor &atvTensor = atVCacheTensors.at(i);
        ATB_LOG(INFO) << "ModelTorch atVCacheTensors[" << i << "]"
                      << " data:" << atvTensor.data_ptr() << ", storage_offset:" << atvTensor.storage_offset()
                      << ", format:" << Utils::GetTensorNpuFormat(atvTensor) << ", shape:" << atvTensor.sizes()
                      << ", options:" << atvTensor.options();
    }

    std::vector<atb::Tensor> kCacheTensors;
    std::vector<atb::Tensor> vCacheTensors;
    AtTensor2Tensor(atKCacheTensors, kCacheTensors);
    AtTensor2Tensor(atVCacheTensors, vCacheTensors);

    if (atb_speed::GetSingleton<atb_speed::Config>().IsConvertNCHWToND()) {
        for (auto &kCacheTensor : kCacheTensors) {
            if (kCacheTensor.desc.format == ACL_FORMAT_NCHW) {
                kCacheTensor.desc.format = ACL_FORMAT_ND;
            }
        }
        for (auto &vCacheTensor : vCacheTensors) {
            if (vCacheTensor.desc.format == ACL_FORMAT_NCHW) {
                vCacheTensor.desc.format = ACL_FORMAT_ND;
            }
        }
    }
    int64_t atbStatus = model_->SetKVCache(kCacheTensors, vCacheTensors);
    return atbStatus;
}

std::vector<torch::Tensor> ModelTorch::Execute(std::vector<torch::Tensor> atInTensors, std::string param)
{
    atInternalTensors_.clear();
    for (size_t i = 0; i < atInTensors.size(); ++i) {
        const torch::Tensor &atTensor = atInTensors.at(i);
        ATB_LOG(INFO) << "ModelTorch atInTensors[" << i << "]"
                      << " data:" << atTensor.data_ptr() << ", storage_offset:" << atTensor.storage_offset()
                      << ", format:" << Utils::GetTensorNpuFormat(atTensor) << ", shape:" << atTensor.sizes()
                      << ", options:" << atTensor.options();
    }

    std::vector<atb::Tensor> inTensors;
    AtTensor2Tensor(atInTensors, inTensors);
    if (atb_speed::GetSingleton<atb_speed::Config>().IsConvertNCHWToND()) {
        for (auto &inTensor : inTensors) {
            if (inTensor.desc.format == ACL_FORMAT_NCHW) {
                inTensor.desc.format = ACL_FORMAT_ND;
            }
        }
    }
    std::vector<atb::TensorDesc> inTensorDescs(model_->GetInputNum());
    for (size_t i = 0; i < inTensors.size(); ++i) {
        inTensorDescs.at(i) = inTensors.at(i).desc;
    }
    std::vector<atb::TensorDesc> outTensorDescs(model_->GetOutputNum());
    atb::Status st = model_->InferShape(inTensorDescs, outTensorDescs);
    ATB_LOG_IF(st != 0, FATAL) << "ModelTorch infer shape fail, error code: " << st;

    std::vector<torch::Tensor> atOutTensors(outTensorDescs.size());
    for (size_t i = 0; i < atOutTensors.size(); ++i) {
        ATB_LOG(INFO) << "ModelTorch outTensorDescs[" << i
                      << "]:" << atb_speed::TensorUtil::TensorDescToString(outTensorDescs.at(i));
        atb_speed::Timer timer;
        atOutTensors.at(i) = Utils::CreateAtTensorFromTensorDesc(outTensorDescs.at(i));
        atb_speed::GetSingleton<atb_speed::Statistic>().createTensorTime += timer.ElapsedMicroSecond();
        atb_speed::GetSingleton<atb_speed::Statistic>().mallocTorchTensorSize +=
            atb::Utils::GetTensorSize(outTensorDescs.at(i));
    }

    std::vector<atb::Tensor> outTensors;
    AtTensor2Tensor(atOutTensors, outTensors);
    if (atb_speed::GetSingleton<atb_speed::Config>().IsConvertNCHWToND()) {
        for (auto &outTensor : outTensors) {
            if (outTensor.desc.format == ACL_FORMAT_NCHW) {
                outTensor.desc.format = ACL_FORMAT_ND;
            }
        }
    }

    int64_t atbStatus = ExecuteOutImpl(inTensors, outTensors, param);
    if (atbStatus != atb::NO_ERROR) {
        std::vector<torch::Tensor> atNullOutTensors(0);
        return atNullOutTensors;
    }
    return atOutTensors;
}

int64_t ModelTorch::ExecuteOut(std::vector<torch::Tensor> atInTensors, std::vector<torch::Tensor> atOutTensors,
                               std::string param)
{
    atInternalTensors_.clear();
    std::vector<atb::Tensor> inTensors;
    AtTensor2Tensor(atInTensors, inTensors);

    std::vector<atb::Tensor> outTensors;
    AtTensor2Tensor(atOutTensors, outTensors);

    int64_t atbStatus = ExecuteOutImpl(inTensors, outTensors, param);
    return atbStatus;
}

int64_t ModelTorch::ExecuteOutImpl(std::vector<atb::Tensor> &inTensors, std::vector<atb::Tensor> &outTensors,
                                   const std::string &param)
{
    int64_t atbStatus = model_->Execute(context_.get(), inTensors, outTensors, param);
    executeCount_++;
    return atbStatus;
}

int64_t ModelTorch::AtTensor2Tensor(std::vector<torch::Tensor> &atTensors, std::vector<atb::Tensor> &opsTensors)
{
    for (auto &atTensor : atTensors) {
        Utils::ContiguousAtTensor(atTensor);
        atb::Tensor tensor = Utils::AtTensor2Tensor(atTensor);
        opsTensors.push_back(tensor);
    }
    return atb::NO_ERROR;
}

std::string ModelTorch::GetSaveTensorDir()
{
    std::string dir = std::to_string(executeCount_) + "/" + std::to_string(modelId_) + "_ModelTorch";
    return atb_speed::Config::GetSaveTensorDir() + "/" + dir;
}

TORCH_LIBRARY(ModelTorch, m)
{
    m.class_<ModelTorch>("ModelTorch")
        .def(torch::init<std::string>())
        .def("set_param", &ModelTorch::SetParam)
        .def("set_weight", &ModelTorch::SetWeight)
        .def("set_kv_cache", &ModelTorch::SetKVCache)
        .def("execute", &ModelTorch::Execute)
        .def("execute_out", &ModelTorch::ExecuteOut);
}