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
#include "atb/train_op_params.h"
#endif 

using namespace std;

void inferShapeGenAttentionMask(c10::SmallVector<int64_t, N> &size, const std::vector<int> seqLen, int headNum)
{
    int output_shape = 0;
    for (auto item : seqLen) {
        output_shape += headNum * item * item;
    }
    size = {output_shape};
}

at::Tensor gen_attention_mask(const at::Tensor &input_tensor, const std::vector<int> seqLen, int headNum)
{
#ifndef ENABLE_ATB
    TORCH_CHECK(false, "gen_attention_mask not implemented");
#else
    atb::train::GenAttentionMaskParam param;
    param.headNum = headNum;
    for (auto item : seqLen) {
        param.seqLen.push_back(item);
    }
    c10::SmallVector<int64_t, N> output_shape;
    inferShapeGenAttentionMask(output_shape, seqLen, headNum);
    at::Tensor output_tensor = CreateAtTensor(output_shape, input_tensor.scalar_type());

    ParamSetter paramsetter;
    paramsetter.Input(input_tensor)
               .Output(output_tensor);

    RUN_ATB_CMD(param, paramsetter, "GenAttentionMaskOperation");
    return output_tensor;
#endif
}