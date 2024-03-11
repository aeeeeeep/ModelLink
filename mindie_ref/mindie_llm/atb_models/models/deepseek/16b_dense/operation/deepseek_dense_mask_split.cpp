/*
* Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
*  * you may not use this file except in compliance with the License.
*  * You may obtain a copy of the License at
*  *
*  * http://www.apache.org/licenses/LICENSE-2.0
*  *
*  * Unless required by applicable law or agreed to in writing, software
*  * distributed under the License is distributed on an "AS IS" BASIS,
*  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  * See the License for the specific language governing permissions and
*  * limitations under the License.
*  */
#include "deepseek_dense_mask_split.h"
#include <atb/atb_infer.h>
#include "atb_speed/log.h"

namespace atb_speed {
namespace deepseekDense {
enum DeepseekDenseMaskSplitTensorId {
    IN_INPUT_MASK = 0,
    OUT_EXPERT_MASK_ZERO,
    OUT_EXPERT_MASK_ONE,
    OUT_EXPERT_MASK_TWO,
    OUT_EXPERT_MASK_THREE,
    OUT_EXPERT_MASK_FOUR,
    OUT_EXPERT_MASK_FIVE,
    OUT_EXPERT_MASK_SIX,
    OUT_EXPERT_MASK_SEVEN,
    INTERMIDATE_EXPERT_MASK_ZERO_THREE,
    INTERMIDATE_EXPERT_MASK_FOUR_SEVEN,
    INTERMIDATE_EXPERT_MASK_ZERO_ONE,
    INTERMIDATE_EXPERT_MASK_TWO_THREE,
    INTERMIDATE_EXPERT_MASK_FOUR_FIVE,
    INTERMIDATE_EXPERT_MASK_SIX_SEVEN
};

static const uint64_t IN_TENSOR_COUNT = 1;
static const uint64_t OUT_TENSOR_COUNT = 8;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 6;
static const uint64_t NODE_COUNT = 7;

atb::Status CreateDeepseekDenseMaskSplitOperation(const DeepseekDenseMaskSplitParam &param, atb::Operation **operation)
{
    std::shared_ptr<int64_t> batchDimPtr = std::make_shared<int64_t>(0);

    atb::GraphParam opGraph;
    opGraph.name = "DeepseekDenseMaskSplit";
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    atb::Node &expertMaskSplit0Node = opGraph.nodes.at(nodeId++);
    atb::Node &expertMaskSplit1Node = opGraph.nodes.at(nodeId++);
    atb::Node &expertMaskSplit2Node = opGraph.nodes.at(nodeId++);
    atb::Node &expertMaskSplit3Node = opGraph.nodes.at(nodeId++);
    atb::Node &expertMaskSplit4Node = opGraph.nodes.at(nodeId++);
    atb::Node &expertMaskSplit5Node = opGraph.nodes.at(nodeId++);
    atb::Node &expertMaskSplit6Node = opGraph.nodes.at(nodeId++);

    atb::infer::SplitParam splitParam = {param.splitDim, param.splitSize};
    CreateOperation(splitParam, &expertMaskSplit0Node.operation);
    expertMaskSplit0Node.inTensorIds = {IN_INPUT_MASK};
    expertMaskSplit0Node.outTensorIds = {
        INTERMIDATE_EXPERT_MASK_ZERO_THREE,
        INTERMIDATE_EXPERT_MASK_FOUR_SEVEN,
    };
    ATB_LOG(INFO) << "Expert Mask splited 0-3, 4-7 success";

    CreateOperation(splitParam, &expertMaskSplit1Node.operation);
    expertMaskSplit1Node.inTensorIds = {INTERMIDATE_EXPERT_MASK_ZERO_THREE};
    expertMaskSplit1Node.outTensorIds = {
        INTERMIDATE_EXPERT_MASK_ZERO_ONE,
        INTERMIDATE_EXPERT_MASK_TWO_THREE,
    };
    ATB_LOG(INFO) << "Expert Mask splited 0-1, 2-3 success";

    CreateOperation(splitParam, &expertMaskSplit2Node.operation);
    expertMaskSplit2Node.inTensorIds = {INTERMIDATE_EXPERT_MASK_ZERO_ONE};
    expertMaskSplit2Node.outTensorIds = {
        OUT_EXPERT_MASK_ZERO,
        OUT_EXPERT_MASK_ONE,
    };
    ATB_LOG(INFO) << "Expert Mask splited 0, 1 success";

    CreateOperation(splitParam, &expertMaskSplit3Node.operation);
    expertMaskSplit3Node.inTensorIds = {INTERMIDATE_EXPERT_MASK_TWO_THREE};
    expertMaskSplit3Node.outTensorIds = {
        OUT_EXPERT_MASK_TWO,
        OUT_EXPERT_MASK_THREE,
    };
    ATB_LOG(INFO) << "Expert Mask splited 2, 3 success";

    CreateOperation(splitParam, &expertMaskSplit4Node.operation);
    expertMaskSplit4Node.inTensorIds = {INTERMIDATE_EXPERT_MASK_FOUR_SEVEN};
    expertMaskSplit4Node.outTensorIds = {
        INTERMIDATE_EXPERT_MASK_FOUR_FIVE,
        INTERMIDATE_EXPERT_MASK_SIX_SEVEN,
    };
    ATB_LOG(INFO) << "Expert Mask splited 4-5, 6-7 success";

    CreateOperation(splitParam, &expertMaskSplit5Node.operation);
    expertMaskSplit5Node.inTensorIds = {INTERMIDATE_EXPERT_MASK_FOUR_FIVE};
    expertMaskSplit5Node.outTensorIds = {
        OUT_EXPERT_MASK_FOUR,
        OUT_EXPERT_MASK_FIVE,
    };
    ATB_LOG(INFO) << "Expert Mask splited 4, 5 success";

    CreateOperation(splitParam, &expertMaskSplit6Node.operation);
    expertMaskSplit6Node.inTensorIds = {INTERMIDATE_EXPERT_MASK_SIX_SEVEN};
    expertMaskSplit6Node.outTensorIds = {
        OUT_EXPERT_MASK_SIX,
        OUT_EXPERT_MASK_SEVEN,
    };
    ATB_LOG(INFO) << "Expert Mask splited 6, 7 success";

    return atb::CreateOperation(opGraph, operation);
}
}
} // namespace atb_speed