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

#ifndef ASCENDSPEED_TE_OPS_CSRC_UTILS_H
#define ASCENDSPEED_TE_OPS_CSRC_UTILS_H

#include <torch/extension.h>

namespace pybind11 {
namespace detail {
template <typename CharT>
struct type_caster<c10::basic_string_view<CharT>, enable_if_t<is_std_char_type<CharT>::value>>
    : string_caster<c10::basic_string_view<CharT>, true> {};

}
}

#endif