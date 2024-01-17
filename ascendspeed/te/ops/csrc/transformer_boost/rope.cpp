// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <string>
#include <vector>
#include <ATen/Tensor.h>

#include "../ops_interface.h"
#include "inc/adapter.h"

#ifdef ENABLE_ATB
#include "inc/atb_adapter.h"
#include "atb/operation.h"
#include "atb/infer_op_params.h"
#include "atb/train_op_params.h"
#endif 

using namespace std;

std::tuple<at::Tensor, at::Tensor> rope(const at::Tensor &input1, const at::Tensor &input2, 
    const at::Tensor &input3, const at::Tensor &input4, const at::Tensor &input5, int rotaryCoeff, int cosFormat)
{
#ifndef ENABLE_ATB
    TORCH_CHECK(false, "rope not implemented");
#else
    atb::infer::RopeParam param;
    param.rotaryCoeff = rotaryCoeff;
    param.cosFormat = cosFormat;
    at::Tensor output_tensor1 = CreateAtTensor(input1.sizes(), input1.scalar_type());
    at::Tensor output_tensor2 = CreateAtTensor(input2.sizes(), input2.scalar_type());

    ParamSetter paramsetter;
    paramsetter.Input(input1)
               .Input(input2)
               .Input(input3)
               .Input(input4)
               .Input(input5)
               .Output(output_tensor1)
               .Output(output_tensor2);

    RUN_ATB_CMD(param, paramsetter, "RopeOperation");
    return std::make_tuple(output_tensor1, output_tensor2);
#endif
}


std::tuple<at::Tensor, at::Tensor> rope_grad(const at::Tensor &input1, const at::Tensor &input2,
    const at::Tensor &input3, const at::Tensor &input4, const std::vector<int> qSeqLen)
{
#ifndef ENABLE_ATB
    TORCH_CHECK(false, "rope_grad not implemented");
#else
    atb::train::RopeGradParam param;
    for (auto item : qSeqLen) {
        param.qSeqLen.push_back(item);
    }
    at::Tensor output_tensor1 = CreateAtTensor(input1.sizes(), input1.scalar_type());
    at::Tensor output_tensor2 = CreateAtTensor(input2.sizes(), input2.scalar_type());

    ParamSetter paramsetter;
    paramsetter.Input(input1)
               .Input(input2)
               .Input(input3)
               .Input(input4)
               .Output(output_tensor1)
               .Output(output_tensor2);

    RUN_ATB_CMD(param, paramsetter, "RopeGradOperation");
    return std::make_tuple(output_tensor1, output_tensor2);
#endif
}