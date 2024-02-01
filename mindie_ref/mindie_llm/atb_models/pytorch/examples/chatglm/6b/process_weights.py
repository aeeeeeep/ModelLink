# Copyright 2024 Huawei Technologies Co., Ltd
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

import argparse
import os
import shutil

from transformers import AutoConfig
from manager import ModeManager


def parse_args():
    parser = argparse.ArgumentParser(description="Process Model weights.")
    parser.add_argument(
        "--model_path",
        required=True,
        help="Location of Model weights, which contains model folders")
    parser.add_argument(
        "--parallel_float_weight_path",
        default='tensor_parallel',
        help="Location to write the part weights")
    parser.add_argument(
        "--tp_size",
        type=int,
        default=1,
        help="the size of parallel"
    )
    parser.add_argument(
        "--model_file_cut",
        default="modeling_chatglm_cut.py",
        help="model_file_cut"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ENABLE_QUANT = os.environ.get("ENABLE_QUANT", "0") == "1"
    ENABLE_SPARSE = os.environ.get("ENABLE_SPARSE", "0") == "1"
    mode = ENABLE_SPARSE and "sparse" or ENABLE_QUANT and "quant" or "float"
    print(f"[info]: Processing weights with mode '{mode}'.")

    shutil.copy(args.model_file_cut, os.path.join(
        args.model_path, "modeling_chatglm.py"))

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    manager = ModeManager.get_manager(mode)
    manager(config, pretrained_model_path=args.model_path, world_size=args.tp_size).process_weights(args.tp_size)
