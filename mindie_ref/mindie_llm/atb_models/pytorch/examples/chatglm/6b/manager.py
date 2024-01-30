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

import json
import math
import os
import shutil
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import List, Dict, Union

import numpy as np
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
from transformers import AutoModel


class BaseManager(metaclass=ABCMeta):
    def __init__(
        self,
        config,
        rank=0,
        world_size=1,
        device="cpu",
        **kwargs,
    ):

        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.hidden_size_per_attention_head = config.hidden_size // config.num_attention_heads

        norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        pre_scale = []
        for layer_id in range(1, config.num_layers + 1):
            pre_scale.append(1 / (norm_factor * layer_id) * layer_id)
        post_scale = [1.0] * config.num_layers

        self.param_dict = {
            "rmsNormEps": config.layernorm_epsilon,
            "numHeadsPerPartition": config.num_attention_heads // self.world_size,
            "hiddenSizePerHead": self.hidden_size_per_attention_head,
            "transKey": True,
            "residualAddScale": (2 * config.num_layers) ** 0.5,
            "layerNum": config.num_layers,
            "headNum": config.num_attention_heads,
            "rank": self.rank,
            "rankSize": self.world_size,
            "backend": "hccl",
            "preScale": pre_scale,
            "postScale": post_scale,
            "isEncoder": True,
            "quantmodel": False,
            "correctNodeId": -1,
            "isSparse": False,
            "qkvInputScale": [], "qkvInputOffset": [],
            "denseInputScale": [], "denseInputOffset": [],
            "selfLnInputScale": [], "selfLnInputOffset": [],
            "ffnOutInputScale": [], "ffnOutInputOffset": [],
            "offsetX": [0]*200, "compressInfo": [0]*200
        }

        self.in_beta = torch.zeros(
            self.config.hidden_size, device=device, dtype=torch.float16)
        self.placeholder = torch.ones(1, device=device)
        self.layer_id_input = [torch.tensor(
            [i], dtype=torch.int32, device=device) for i in range(config.num_layers)]

    @abstractmethod
    def load_primal_weights(self) -> Union[Dict[str, torch.Tensor], Dict[str, Dict[str, torch.Tensor]]]:
        """
        Load model weights from pretrained_model_path or quant/sparse_weight_path
        """
        raise NotImplementedError()

    @abstractmethod
    def save_sliced_weights(self) -> None:
        """
        Save sliced weights to float_parallel_weight_path or quant/sparse_weight_path
        """
        raise NotImplementedError()

    @abstractmethod
    def process_weights(self) -> None:
        """
        Load, slice and save weights
        """
        raise NotImplementedError()

    @abstractmethod
    def init_param(self) -> str:
        """
        Initialize param for model_torch interface

        Example:
        acl_decoder_param = manager.init_param(*args, **kwargs)
        acl_decoder_operation.set_param(acl_decoder_param)
        """
        raise NotImplementedError()

    @abstractmethod
    def init_weights(self) -> List[torch.Tensor]:
        """
        Initialize weights for model_torch interface

        Example:
        weights = manager.init_weights(*args, **kwargs)
        acl_decoder_operation.set_weight(weights)
        """
        raise NotImplementedError()

    @abstractmethod
    def init_inputs(self) -> List[torch.Tensor]:
        """
        Initialize inputs for model_torch interface

        Example:
        inputs = manager.init_inputs(*args, **kwargs)
        acl_param = json.dumps({"tokenOffset": self.token_num, "seqLen": [seq_length] * batch_size})
        acl_model_out = acl_decoder_operation.execute(inputs, acl_param)
        """
        raise NotImplementedError()

    def _slice_tensors(self, key, tensor, tp_size) -> List[torch.Tensor]:
        """
        Slice tensors according to key name
        """
        if tp_size == 1:
            return [tensor]

        sliced_tensor_list = []
        if 'dense_h_to_4h' in key or 'query_key_value' in key or 'lm_head' in key:  # weight
            sliced_tensor_list = tensor.chunk(tp_size, dim=0)
        elif 'dense_4h_to_h.weight' in key or 'attention.dense.weight' in key:
            sliced_tensor_list = tensor.chunk(tp_size, dim=1)
        else:
            sliced_tensor_list = [tensor] * tp_size
        return sliced_tensor_list

    def slice_tensors(self, state_dict, tp_size, weight_type) -> List[Dict[str, torch.Tensor]]:
        """
        Slice tensors
        """
        state_dict_list = [{} for i in range(tp_size)]
        for key, tensor in state_dict.items():
            sliced_tensor_list = self._slice_tensors(key, tensor, tp_size)
            for i in range(tp_size):
                state_dict_list[i][key] = sliced_tensor_list[i]
        return state_dict_list


class Float(BaseManager):
    def __init__(self, config, pretrained_model_path="", **kwargs):
        super(Float, self).__init__(config, **kwargs)
        self.model_path = pretrained_model_path
        self.float_weight_path = Path(
            pretrained_model_path).joinpath(f"tensor_parallel_tp{self.world_size}")

    def load_primal_weights(self) -> Dict[str, torch.Tensor]:
        model = AutoModel.from_pretrained(
            self.model_path, trust_remote_code=True).half()
        return model.state_dict()

    def save_sliced_weights(self, state_dict_list, tp_size) -> None:
        self.config.world_size = tp_size
        parallel_model = AutoModel.from_config(
            self.config, trust_remote_code=True)
        for i in range(tp_size):
            target_dir = Path(self.float_weight_path).joinpath(
                "part_model", str(i))
            parallel_model.load_state_dict(state_dict_list[i])
            parallel_model.save_pretrained(target_dir)
            for source_file in ["configuration_chatglm.py", "quantization.py"]:
                shutil.copy(Path(self.model_path).joinpath(
                    source_file), target_dir)

    def process_weights(self, tp_size) -> None:
        if tp_size == 1:
            return
        if Path(self.float_weight_path).exists():
            print(
                f"[info]: The parallel float weights has exist in '{self.float_weight_path}'. Please remove it if you want to process float weights again.")
        else:
            state_dict = self.load_primal_weights()
            state_dict_list = self.slice_tensors(
                state_dict, tp_size, weight_type="weights")
            self.save_sliced_weights(state_dict_list, tp_size)
            print(
                f"[info]: The parallel float weights has been saved to '{self.float_weight_path}'.")

    def init_param(self, is_encoder=True) -> str:
        param_dict = {
            "isEncoder": is_encoder,
        }
        self.param_dict.update(param_dict)
        return json.dumps(self.param_dict)

    def init_weights(self, state_dict, is_format_nz=False) -> List[torch.Tensor]:
        weights_list = []
        for key in state_dict.keys():
            if 'rotary_emb.inv_freq' not in key:
                weights_list.append(state_dict[key])
        return weights_list

    def init_inputs(self, inputs_dict) -> List[torch.Tensor]:
        inputs_list = [
            inputs_dict["input_ids"],
            inputs_dict["rope_cos"],
            inputs_dict["rope_sin"],
            inputs_dict["seq_len_tensor"],
            inputs_dict["attention_mask_max"],
            inputs_dict["token_offset"],
            inputs_dict["k_cache_input"],
            inputs_dict["v_cache_input"],
            self.placeholder,
            self.placeholder,
        ]
        inputs_list.extend(self.layer_id_input)
        return inputs_list



class ModeManager:
    mode_dict = {
        'float': Float,
    }

    @classmethod
    def get_manager(cls, mode):
        if mode not in cls.mode_dict:
            return ValueError("only support 'float' mode")
        return cls.mode_dict[mode]
