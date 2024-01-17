// Copyright (c) 2023 Huawei Technologies Co., Ltd
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

#include <torch/extension.h>
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include "ops_interface.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("npu_fusion_attention", &npu_flash_attention, "fusion attention forward");
    m.def("npu_fusion_attention_grad", &npu_flash_attention_grad, "fusion attention backward");
    m.def("npu_dropout_add_layer_norm", &npu_dropout_add_layer_norm, "Run Dropout + Add + LayerNorm forward kernel");
    m.def("npu_gen_attention_mask", &gen_attention_mask, "gen attentionmask on ascend device");
    m.def("npu_fast_softmax", &fast_softmax, "fast softmax forward");
    m.def("npu_fast_softmax_grad", &fast_softmax_grad, "fast softmax backward");
    m.def("npu_unpad_seqlen", &unpad_seqlen, "unpad seqlen input");
    m.def("npu_pad_seqlen", &pad_seqlen, "pad seqlen input");
    m.def("npu_rope", &rope, "rope forward");
    m.def("npu_rope_grad", &rope_grad, "rope backward");
    m.def("npu_strided_batch_matmul", &strided_batch_matmul, "stridedbatchmatmul forward");
}
