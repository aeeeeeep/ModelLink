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
#include "mixtral_dense_multi_layer_linear.h"
#include <atb/atb_infer.h>
#include "atb_speed/log.h"

namespace atb_speed {
namespace mixtralDense {
enum MixtralDenseMultiLayerLinearTensorId {
    IN_INPUTTENSOR = 0,
    IN_WEIGHTTENSOR,
    OUT_MATMULRESULTQTENSOR,
    OUT_MATMULRESULTKTENSOR,
    OUT_MATMULRESULTVTENSOR,
    INTERMIDATE_LINEAR_OUT,
    INTERNAL_KV
};

static const uint64_t IN_TENSOR_COUNT = 2;
static const uint64_t OUT_TENSOR_COUNT = 3;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 2;
static const uint64_t NODE_COUNT = 4;
static uint64_t DIM3 = 3;

atb::Status CreateMixtralDenseMultiLayerLinearOperation(const MixtralDenseMultiLayerLinearParam &param, atb::Operation **operation)
{
    std::shared_ptr<int64_t> batchDimPtr = std::make_shared<int64_t>(0);

    atb::GraphParam opGraph;
    opGraph.name = "MixtralDenseMultiLayerLinear";
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    atb::Node &linearNode = opGraph.nodes.at(nodeId++);
    atb::Node &qPassSliceNode = opGraph.nodes.at(nodeId++);
    atb::Node &kVPassSliceNode = opGraph.nodes.at(nodeId++);
    atb::Node &splitNode = opGraph.nodes.at(nodeId++);

    atb::SVector<int64_t> sliceOffsetQ = {0, 0, 0};
    atb::SVector<int64_t> sliceSizeQ = {-1, -1, param.headNum * param.dk};
    atb::SVector<int64_t> sliceOffsetKV = {0, 0, param.headNum * param.dk};
    atb::SVector<int64_t> sliceSizeKV = {-1, -1, 2 * param.dk};

    atb::infer::LinearParam linearParam;
    linearParam.transposeA = false;
    linearParam.transposeB = param.transpose;
    linearParam.hasBias = false;
    CreateOperation(linearParam, &linearNode.operation);
    linearNode.inTensorIds = {IN_INPUTTENSOR, IN_WEIGHTTENSOR};
    linearNode.outTensorIds = {INTERMIDATE_LINEAR_OUT};
    linearNode.inTensorReshapeFuncs.resize(linearNode.inTensorIds.size());
    linearNode.inTensorReshapeFuncs[0] = [batchDimPtr](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2; // dimNum: 2
        *batchDimPtr = oldShape.dims[0];
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
        newShape.dims[1] = oldShape.dims[2];
    };
    ATB_LOG(INFO) << "create input Matmul";

    atb::infer::SliceParam slicePassParam;
    slicePassParam.offsets = sliceOffsetQ;
    slicePassParam.size = sliceSizeQ;
    CREATE_OPERATION(slicePassParam, &qPassSliceNode.operation);
    qPassSliceNode.inTensorIds = {INTERMIDATE_LINEAR_OUT};
    qPassSliceNode.outTensorIds = {OUT_MATMULRESULTQTENSOR};
    qPassSliceNode.inTensorReshapeFuncs.resize(qPassSliceNode.inTensorIds.size());
    qPassSliceNode.inTensorReshapeFuncs[0] = [batchDimPtr](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 3; // dimNum: 3
        newShape.dims[0] = (*batchDimPtr);
        newShape.dims[1] = oldShape.dims[0] / (*batchDimPtr);
        newShape.dims[2] = oldShape.dims[1];
    };

    atb::infer::SliceParam slicePassKVParam;
    slicePassKVParam.offsets = sliceOffsetKV;
    slicePassKVParam.size = sliceSizeKV;
    CREATE_OPERATION(slicePassKVParam, &kVPassSliceNode.operation);
    kVPassSliceNode.inTensorIds = {INTERMIDATE_LINEAR_OUT};
    kVPassSliceNode.outTensorIds = {INTERNAL_KV};
    kVPassSliceNode.inTensorReshapeFuncs.resize(kVPassSliceNode.inTensorIds.size());
    kVPassSliceNode.inTensorReshapeFuncs[0] = [batchDimPtr](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 3; // dimNum: 3
        newShape.dims[0] = (*batchDimPtr);
        newShape.dims[1] = oldShape.dims[0] / (*batchDimPtr);
        newShape.dims[2] = oldShape.dims[1];
    };

    atb::infer::SplitParam splitParam = {2, 2};
    CreateOperation(splitParam, &splitNode.operation);
    splitNode.inTensorIds = {INTERNAL_KV};
    splitNode.outTensorIds = {OUT_MATMULRESULTKTENSOR, OUT_MATMULRESULTVTENSOR};

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                    atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        outTensorDescs.at(0).shape.dimNum = DIM3;
        outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
        outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1];

        outTensorDescs.at(1) = inTensorDescs.at(0);
        outTensorDescs.at(1).shape.dimNum = DIM3;
        outTensorDescs.at(1).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
        outTensorDescs.at(1).shape.dims[1] = inTensorDescs.at(0).shape.dims[1];

        outTensorDescs.at(2) = inTensorDescs.at(0);
        outTensorDescs.at(2).shape.dimNum = DIM3;
        outTensorDescs.at(2).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
        outTensorDescs.at(2).shape.dims[1] = inTensorDescs.at(0).shape.dims[1];

        outTensorDescs.at(0).shape.dims[2] = param.headNum * param.dk;
        outTensorDescs.at(1).shape.dims[2] = param.headNum * param.dk / 4;
        outTensorDescs.at(2).shape.dims[2] = param.headNum * param.dk / 4;

        return atb::NO_ERROR;
    };
    ATB_LOG(INFO) << "create input Split";

    return atb::CreateOperation(opGraph, operation);
}
}
} // namespace atb_speed