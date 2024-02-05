# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the License);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
import os
import numpy as np

from modelslim.pytorch.weight_compression import CompressConfig, Compressor


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, mode=0o750)
    return path


if __name__ == "__main__":
    weight_path = "./llama7b_sparsequant_parallel/0/quant_weight.npy"
    print("---->weight_path: ", weight_path)
    save_path = "./compress_0"
    print("---->save_path: ", save_path)
    index_root = make_dir(os.path.join(save_path, 'index'))
    weight_root = make_dir(os.path.join(save_path, 'weight'))
    info_root = make_dir(os.path.join(save_path, 'info'))

    compress_config = CompressConfig(do_pseudo_sparse=False, sparse_ratio=1, is_debug=True, record_detail_root=save_path)
    compressor = Compressor(compress_config, weight_path)
    compress_weight, compress_index, compress_info = compressor.run()

    compressor.export(compress_weight, weight_root)
    compressor.export(compress_index, index_root)
    compressor.export(compress_info, info_root, dtype=np.int64)