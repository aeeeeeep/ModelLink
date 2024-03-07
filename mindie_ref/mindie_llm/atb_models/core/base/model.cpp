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
#include "atb_speed/base/model.h"
#include <nlohmann/json.hpp>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic pop
#include <acl/acl.h>
#include <atb/types.h>
#include <atb/utils.h>

#include "atb_speed/log.h"
#include "atb_speed/utils/config.h"
#include "atb_speed/utils/singleton.h"
#include "atb_speed/utils/statistic.h"
#include "atb_speed/utils/tensor_util.h"
#include "atb_speed/utils/timer.h"
#include "atb_speed/utils/speed_probe.h"

namespace atb_speed {
static bool IsTensorDimsEqual(const atb::Dims &left, const atb::Dims &other)
{
    if (left.dimNum != other.dimNum) {
        return false;
    }
    
    for (uint64_t i = 0; i < left.dimNum; ++i) {
        if (left.dims[i] != other.dims[i]) {
            return false;
        }
    }

    return true;
}

std::string Model::Graph::ToString() const
{
    std::stringstream ss;
    for (size_t i = 0; i < weightTensors.size(); ++i) {
        ss << "weightTensors[" << i << "]:" << &weightTensors.at(i) << " " 
           << TensorUtil::TensorToString(weightTensors.at(i)) << std::endl;
    }
    for (size_t i = 0; i < inTensors.size(); ++i) {
        ss << "inTensors[" << i << "]:" << &inTensors.at(i) << " " << TensorUtil::TensorToString(inTensors.at(i)) 
           << std::endl;
    }
    for (size_t i = 0; i < outTensors.size(); ++i) {
        ss << "outTensors[" << i << "]:" << &outTensors.at(i) << " " << TensorUtil::TensorToString(outTensors.at(i)) 
           << std::endl;
    }
    for (size_t i = 0; i < internalTensors.size(); ++i) {
        ss << "internalTensors[" << i << "]:" << &internalTensors.at(i) << " " 
           << TensorUtil::TensorToString(internalTensors.at(i)) << std::endl;
    }
    ss << "nodes:" << nodes.size() << std::endl;

    for (size_t i = 0; i < nodes.size(); ++i) {
        auto &node = nodes.at(i);
        ss << "node[" << i << "] operation:" << node.operation.get() << ", operationName:" << node.operation->GetName() 
           << std::endl;
        for (auto tensorIt : node.inTensors) {
            ss << "node[" << i << "] inTensor:" << tensorIt << " " << TensorUtil::TensorToString(*tensorIt) 
               << std::endl;
        }
        for (auto tensorIt : node.outTensors) {
            ss << "node[" << i << "] outTensor:" << tensorIt << " " << TensorUtil::TensorToString(*tensorIt) 
               << std::endl;
        }
    }
    return ss.str();
}

void Model::Graph::Init()
{
    for (size_t i = 0; i < nodes.size(); i++) {
        auto &node = nodes.at(i);
        node.variantPack.inTensors.resize(node.inTensors.size());
        node.variantPack.outTensors.resize(node.outTensors.size());
    }
    InitTensorType();
    InitTensorMaxNodeMap();
}

void Model::Graph::InitTensorType()
{
    for (auto &node : nodes) {
        node.inTensorTypes.resize(node.inTensors.size());
        node.outTensorTypes.resize(node.outTensors.size());
        for (size_t i = 0; i < node.inTensors.size(); ++i) {
            node.inTensorTypes.at(i) =
                IsInternalTensor(node.inTensors.at(i)) ? Model::INTERMEDIATE_TENSOR : Model::NOT_INTERMEDIATE_TENSOR;
        }
        for (size_t i = 0; i < node.outTensors.size(); ++i) {
            node.outTensorTypes.at(i) =
                IsInternalTensor(node.outTensors.at(i)) ? Model::INTERMEDIATE_TENSOR : Model::NOT_INTERMEDIATE_TENSOR;
        }
    }
}

bool Model::Graph::IsInternalTensor(const atb::Tensor *tensor)
{
    for (auto &internalTensor : internalTensors) {
        if (&internalTensor == tensor) {
            return true;
        }
    }

    return false;
}

void Model::Graph::InitTensorMaxNodeMap()
{
    std::map<atb::Tensor *, uint64_t> tensorMaxNodeIdMap;
    maxNodeIdTensorMap.clear();

    for (size_t i = 0; i < internalTensors.size(); ++i) {
        atb::Tensor &internalTensor = internalTensors[i];
        uint64_t maxNodeId = 0;
        uint64_t dependNodeCount = 0;
        for (size_t nodeId = 0; nodeId < nodes.size(); ++nodeId) {
            auto &node = nodes.at(nodeId);
            for (auto inTensorIt : node.inTensors) {
                if (&internalTensor == inTensorIt) {
                    maxNodeId = nodeId;
                    dependNodeCount++;
                }
            }
        }
        tensorMaxNodeIdMap[&internalTensor] = maxNodeId;
        ATB_LOG_IF(dependNodeCount == 0, ERROR) 
            << "runner graph internal tensor[" << i << "] dependNodeCount is 0, graph wrong";
        maxNodeIdTensorMap[maxNodeId].insert(&internalTensor);
    }
}

Model::Model(const std::string &modelName, const std::string &param) : modelName_(modelName), param_(param)
{
    aclrtGetDevice(&currentDevId_);
}

Model::~Model() {}

int64_t Model::Init(GetWorkspaceFunc getWorkSpaceFunc, CreateTensorFromTensorDescFunc createTensorFromTensorDescFunc,
    RunTaskFunc runTaskFunc)
{
    const char *envStr = std::getenv("ATB_OPERATION_EXECUTE_ASYNC");
    isUsePlanExecuteAsync_ = (envStr != nullptr && std::string(envStr) == "1");
    if (isUsePlanExecuteAsync_ && !runTaskFunc) {
        std::thread thread = std::thread(std::bind(&Model::ThreadProcessTask, this));
        taskProcessThread_ = std::move(thread);
    }

    ATB_LOG(FATAL) << modelName_ << " new, isTaskQueueEnable:" << (runTaskFunc != nullptr) 
                   << ", isUsePlanExecuteAsync:" << isUsePlanExecuteAsync_ << ", currentDevId:" << currentDevId_;
    
    getWorkSpaceFunc_ = getWorkSpaceFunc;
    createTensorFromTensorDescFunc_ = createTensorFromTensorDescFunc;
    runTaskFunc_ = runTaskFunc;

    int64_t atbStatus = BuildGraph();
    graph_.Init();
    ATB_LOG(DEBUG) << modelName_ << " init graph:\n" << graph_.ToString();
    return atbStatus;
}

int64_t Model::SetWeight(const std::vector<atb::Tensor> &weightTensors)
{
    if (graph_.weightTensors.size() != weightTensors.size()) {
        ATB_LOG(ERROR) << modelName_ << " weightTensors.size:" << weightTensors.size() << " != "
                       << " graph.weightTensors.size:" << graph_.weightTensors.size();
        return atb::ERROR_INVALID_IN_TENSOR_NUM;
    }

    graph_.weightTensors = weightTensors;
    return atb::NO_ERROR;
}

int64_t Model::SetKVCache(const std::vector<atb::Tensor> &kCacheTensors, const std::vector<atb::Tensor> &vCacheTensors)
{
    if (graph_.kCacheTensors.size() != kCacheTensors.size()) {
        ATB_LOG(ERROR) << modelName_ << " kCacheTensors.size:" << kCacheTensors.size() << " != "
                       << " graph.kCacheTensors.size:" << graph_.kCacheTensors.size();
        return atb::ERROR_INVALID_IN_TENSOR_NUM;
    }

    if (graph_.vCacheTensors.size() != vCacheTensors.size()) {
        ATB_LOG(ERROR) << modelName_ << " vCacheTensors.size:" << vCacheTensors.size() << " != "
                       << " graph.vCacheTensors.size:" << graph_.vCacheTensors.size();
        return atb::ERROR_INVALID_IN_TENSOR_NUM;
    }

    graph_.kCacheTensors = kCacheTensors;
    graph_.vCacheTensors = vCacheTensors;
    return atb::NO_ERROR;
}

atb::Status Model::Execute(atb::Context *context, std::vector<atb::Tensor> &inTensors,
                           std::vector<atb::Tensor> &outTensors, const std::string &param)
{
    if (graph_.inTensors.size() != inTensors.size() || graph_.outTensors.size() != outTensors.size()) {
        ATB_LOG(ERROR) << modelName_ << " graph.inTensors.size:" << graph_.inTensors.size() 
                       << ", inTensors.size:" << inTensors.size() 
                       << ", graph.outTensors.size:" << graph_.outTensors.size() 
                       << ", outTensors.size:" << outTensors.size();
        return atb::ERROR_INVALID_GRAPH;
    }

    ParseParam(param);

    timer_.Reset();
    ClearInternalTensors();

    allTaskFinish_ = false;
    context_ = context;
    graph_.inTensors = inTensors;
    graph_.outTensors = outTensors;
    ATB_LOG(INFO) << modelName_ << " execute start, executeCount:" << executeCount_ << ", graph:\n"
                  << graph_.ToString();

    for (size_t nodeId = 0; nodeId < graph_.nodes.size(); ++nodeId) {
        BuildNodeVariantPack(nodeId);
        BindParamHostTensor(nodeId);
        atb::Status st = ExecuteNode(nodeId);
        if (st != 0) {
            return st;
        }
    }

    if (atb_speed::SpeedProbe::IsReportModelTopoInfo(modelName_)) {
        std::string modelTopo = GetModelTopoInfo();
        atb_speed::SpeedProbe::ReportModelTopoInfo(modelName_, modelTopo);
    }

    WaitAsyncPlanExecuteFinish();

    GetSingleton<Statistic>().totalTime += timer_.ElapsedMicroSecond();
    ATB_LOG(FATAL) << modelName_ << " executeCount:" << executeCount_ << ", Statistic:[" 
                   << GetSingleton<Statistic>().ToString() << "]";
    GetSingleton<Statistic>().Reset();

    executeCount_++;

    return atb::NO_ERROR;
}

atb::Status Model::ParseParam(const std::string &param)
{
    (void)param;
    return atb::NO_ERROR;
}

atb::Status Model::BindParamHostTensor(uint32_t nodeId)
{
    (void)nodeId;
    return atb::NO_ERROR;
}

void Model::BuildNodeVariantPack(int nodeId)
{
    auto &node = graph_.nodes.at(nodeId);

    atb::SVector<atb::TensorDesc> inTensorDescs;
    inTensorDescs.resize(node.variantPack.inTensors.size());
    for (size_t i = 0; i < node.inTensors.size(); ++i) {
        node.variantPack.inTensors.at(i) = *node.inTensors.at(i);
        inTensorDescs.at(i) = node.inTensors.at(i)->desc;
        ATB_LOG(INFO) << modelName_ << " nodes[" << nodeId << "] inTensors[" << i 
                      << "]:" << TensorUtil::TensorToString(node.variantPack.inTensors.at(i));
    }

    atb::SVector<atb::TensorDesc> outTensorDescs;
    outTensorDescs.resize(node.operation->GetOutputNum());
    atb::Status st = node.operation->InferShape(inTensorDescs, outTensorDescs);

    ATB_LOG_IF(st != 0, FATAL) << modelName_ << " nodes[" << nodeId << "] "
                               << " infer shape fail, error code: " << st;
    for (size_t i = 0; i < outTensorDescs.size(); ++i) {
        ATB_LOG(INFO) << modelName_ << " nodes[" << nodeId << "] outTensorDescs[" << i 
                      << "]:" << TensorUtil::TensorDescToString(outTensorDescs.at(i));
    }

    for (size_t i = 0; i < node.outTensors.size(); ++i) {
        node.variantPack.outTensors.at(i) = *node.outTensors.at(i);
        if (node.outTensorTypes.at(i) == Model::INTERMEDIATE_TENSOR) {
            node.variantPack.outTensors.at(i) = MallocInternalTensor(nodeId, i, outTensorDescs.at(i));
            *node.outTensors.at(i) = node.variantPack.outTensors.at(i);
        }
        if (!TensorUtil::TensorDescEqual(node.variantPack.outTensors.at(i).desc, outTensorDescs.at(i))) {
            ATB_LOG(FATAL) << modelName_ << "  nodes[" << nodeId << "] new outTensorDescs[" << i 
                           << "]:" << TensorUtil::TensorDescToString(outTensorDescs.at(i)) 
                           << ", node.variantPack.outTensors.at[" << i 
                           << "].desc:" << TensorUtil::TensorDescToString(node.variantPack.outTensors.at(i).desc);
        }
    }

    auto it = graph_.maxNodeIdTensorMap.find(nodeId);
    if (it != graph_.maxNodeIdTensorMap.end()) {
        for (auto tensorIt : it->second) {
            FreeInternalTensor(tensorIt->deviceData);
        }
    }
}

atb::Status Model::ExecuteNode(int nodeId)
{
    ExecuteNodeView(nodeId);
    auto &node = graph_.nodes.at(nodeId);

    Timer timer;
    atb::Status st = node.operation->Setup(node.variantPack, node.workspaceSize, context_);
    GetSingleton<Statistic>().planSetupTime += timer.ElapsedMicroSecond();
    if (st != 0) {
        ATB_LOG(ERROR) << modelName_ << " setup node[" << nodeId << "] fail, not call execute";
        return st;
    }

    ATB_LOG(INFO) << modelName_ << " get node[" << nodeId << "] workspace size:" << node.workspaceSize;

    if (node.workspaceSize > 0) {
        node.workspace = getWorkSpaceFunc_(node.workspaceSize);
    }

    if (isUsePlanExecuteAsync_) {
        Timer timer;
        ExecutePlanAsync(nodeId);
        GetSingleton<Statistic>().planAsyncTime += timer.ElapsedMicroSecond();
    } else {
        st = ExecutePlanSync(nodeId);   
    }
    return st;
}

void Model::ThreadProcessTask()
{
    ATB_LOG(FATAL) << modelName_ << " thread process operations start";
    int ret = aclrtSetDevice(currentDevId_);
    ATB_LOG_IF(ret != 0, ERROR) << "AsdRtDeviceSetCurrent fail, error:" << ret;

    size_t processTaskCount = 0;
    while (true) {
        int nodeId = PopTask();
        atb::Status st = ExecutePlanSync(nodeId);
        if (st != 0) {
            allTaskFinish_ = true;
            processTaskCount = 0;
            return;
        }
        processTaskCount++;
        if (processTaskCount == graph_.nodes.size()) {
            ATB_LOG(INFO) << modelName_ << " thread process all operations";
            processTaskCount = 0;
            allTaskFinish_ = true;
        }
    }
}

atb::Status Model::ExecutePlanSync(int nodeId)
{
    auto &node = graph_.nodes.at(nodeId);
    atb::VariantPack &variantPack = node.variantPack;

    ATB_LOG(INFO) << modelName_ << "execute node[" << nodeId << "] start";
    Timer timer;
    atb::Status st = node.operation->Execute(variantPack, (uint8_t*)(node.workspace), node.workspaceSize, context_);
    GetSingleton<Statistic>().planExecuteTime += timer.ElapsedMicroSecond();
    if (st != 0) {
        ATB_LOG(ERROR) << "execute node[" << nodeId << "] fail, error code: " << st;
    }
    return st;
}

void Model::ExecutePlanAsync(int nodeId)
{
    if (runTaskFunc_) {
        runTaskFunc_(modelName_ + std::to_string(nodeId), [=]() {
            ExecutePlanSync(nodeId);
            return 0;
        });
    } else {
        PushTask(nodeId);
    }
}

void Model::PushTask(int nodeId)
{
    std::unique_lock<std::mutex> lock(mutex_);
    taskQueue_.push(nodeId);
    lock.unlock();
    cond_.notify_one();
}

int Model::PopTask()
{
    std::unique_lock<std::mutex> lock(mutex_);
    while (taskQueue_.empty()) {
        cond_.wait(lock);
    }
    int nodeId = taskQueue_.front();
    taskQueue_.pop();
    return nodeId;
}

void Model::WaitAsyncPlanExecuteFinish()
{
    if (isUsePlanExecuteAsync_ && !runTaskFunc_) {
        while (true) {
            if (allTaskFinish_) {
                ATB_LOG(INFO) << modelName_ << " allTaskFinish is true, break";
                break;
            }
        }
    }
}

std::string Model::GetSaveTensorDir()
{
    std::string dir = std::to_string(executeCount_) + "/0_Model";
    return Config::GetSaveTensorDir() + "/" + dir;
}

void Model::ExecuteNodeView(int nodeId)
{
    auto &node = graph_.nodes.at(nodeId);
    if (node.inTensorReshapeFuncs.size() > 0) {
        for (size_t i = 0; i < node.inTensors.size() && node.inTensorReshapeFuncs.at(i) != nullptr; i++) {
            node.inTensorReshapeFuncs.at(i)(node.inTensors.at(i)->desc.shape, node.inTensors.at(i)->desc.shape);
        }
    }
}

bool Model::IsTensorDescEqual(const atb::TensorDesc &tensorDesc, const atb::Tensor &atbTensor) const
{
    return atbTensor.desc.dtype == tensorDesc.dtype && atbTensor.desc.format == tensorDesc.format &&
        IsTensorDimsEqual(atbTensor.desc.shape, tensorDesc.shape);
}

void Model::ClearInternalTensors()
{
    internalTensors_.clear();
}

atb::Tensor Model::MallocInternalTensor(size_t nodeId, size_t outTensorId, const atb::TensorDesc &tensorDesc)
{
    if (GetSingleton<Config>().IsLayerInternalTensorReuse()) {
        for (auto &it : internalTensors_) {
            if (it.second) { // Tensor被使用中，不能被分配其他Op
                continue;
            }

            if (IsTensorDescEqual(tensorDesc, it.first)) {
                it.second = true;
                ATB_LOG(INFO) << modelName_ << " use old internal tensor";
                return it.first;
            }
        }
    }

    ATB_LOG(INFO) << modelName_ << " create internal tensor, node[" << nodeId << "], outTensor[" << outTensorId << "]";
    atb_speed::Timer timer;
    atb::Tensor newTensor = createTensorFromTensorDescFunc_(tensorDesc);
    atb_speed::GetSingleton<atb_speed::Statistic>().createTensorTime += timer.ElapsedMicroSecond();
    atb_speed::GetSingleton<atb_speed::Statistic>().mallocTorchTensorSize += atb::Utils::GetTensorSize(tensorDesc);
    internalTensors_.push_back(std::make_pair(newTensor, true));
    return newTensor;
}

void Model::FreeInternalTensor(void *tensorDeviceData)
{
    if (GetSingleton<Config>().IsLayerInternalTensorReuse()) {
        for (auto &it : internalTensors_) {
            if (it.first.deviceData == tensorDeviceData) {
                it.second = false; // Tensor被释放，可以被后来者使用
                ATB_LOG(INFO) << modelName_ << " free internal tensor";
                break;
            }
        }
    }
}

void Model::GetModelTensorNameList(nlohmann::json &modelJson, std::map<atb::Tensor *, std::string> &tensorNameMap)
{
    std::string tensorName;
    for (size_t i = 0; i < graph_.weightTensors.size(); i++) {
        tensorName = modelName_ + "_weight_" + std::to_string(i);
        modelJson["weightTensors"].emplace_back(tensorName);
        atb::Tensor &weightTensor = graph_.weightTensors[i];
        tensorNameMap[&weightTensor] = tensorName;
    }
    
    for (size_t i = 0; i < graph_.inTensors.size(); i++) {
        tensorName = modelName_ + "_input_" + std::to_string(i);
        modelJson["inTensors"].emplace_back(tensorName);
        atb::Tensor &inTensor = graph_.inTensors[i];
        tensorNameMap[&inTensor] = tensorName;
    }

    for (size_t i = 0; i < graph_.outTensors.size(); i++) {
        tensorName = modelName_ + "_output_" + std::to_string(i);
        modelJson["outTensors"].emplace_back(tensorName);
        atb::Tensor &outTensor = graph_.outTensors[i];
        tensorNameMap[&outTensor] = tensorName;
    }

    for (size_t i = 0; i < graph_.internalTensors.size(); i++) {
        tensorName = modelName_ + "_internal_" + std::to_string(i);
        modelJson["internalTensors"].emplace_back(tensorName);
        atb::Tensor &internalTensor = graph_.internalTensors[i];
        tensorNameMap[&internalTensor] = tensorName;
    }

    for (size_t i = 0; i < graph_.kCacheTensors.size(); i++) {
        tensorName = modelName_ + "_kCache_" + std::to_string(i);
        modelJson["kCacheTensors"].emplace_back(tensorName);
        atb::Tensor &kCacheTensor = graph_.kCacheTensors[i];
        tensorNameMap[&kCacheTensor] = tensorName;
    }
        
    for (size_t i = 0; i < graph_.vCacheTensors.size(); i++) {
        tensorName = modelName_ + "_vCache_" + std::to_string(i);
        modelJson["vCacheTensors"].emplace_back(tensorName);
        atb::Tensor &vCacheTensor = graph_.vCacheTensors[i];
        tensorNameMap[&vCacheTensor] = tensorName;
    }
}

void Model::GetNodeTopoInfo(nlohmann::json &nodeJson, const Node &opNode,
    const std::map<atb::Tensor *, std::string> tensorNameMap)
{
    nodeJson["opName"] = opNode.operation->GetName();

    for (auto inTensor : opNode.inTensors) {
        auto it = tensorNameMap.find(inTensor);
        if (it != tensorNameMap.end()) {
            nodeJson["inTensors"].emplace_back(it->second);
        }
    }

    for (auto outTensor : opNode.outTensors) {
        auto it = tensorNameMap.find(outTensor);
        if (it != tensorNameMap.end()) {
            nodeJson["outTensors"].emplace_back(it->second);
        }
    }
}

std::string Model::GetModelTopoInfo()
{
    nlohmann::json modelJson;
    modelJson["modelName"] = modelName_;

    std::map<atb::Tensor *, std::string> tensorNameMap;
    GetModelTensorNameList(modelJson, tensorNameMap);

    for (size_t nodeId = 0; nodeId < graph_.nodes.size(); nodeId++) {
        const auto &opNode = graph_.nodes.at(nodeId);
        nlohmann::json nodeJson;
        GetNodeTopoInfo(nodeJson, opNode, tensorNameMap);
        modelJson["nodes"].emplace_back(nodeJson);
    }
    return modelJson.dump();
}

} // namespace atb_speed
