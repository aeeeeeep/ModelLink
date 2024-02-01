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
        **kwargs,
    ):

        self.config = config
        self.rank = rank
        self.world_size = world_size

        self.hidden_size_per_attention_head = config.hidden_size // config.num_attention_heads
        self.num_attention_heads_per_partition = config.num_attention_heads
        self.num_multi_query_groups_per_partition = config.multi_query_group_num
        self.multi_query_group_num = max(
            config.multi_query_group_num // self.world_size, 1)

        pre_scale = [layer_id / (math.sqrt(self.config.kv_channels) * layer_id) for layer_id in
                     range(1, self.config.num_layers + 1)]
        post_scale = [1.0] * self.config.num_layers

        self.param_dict = {
            "rmsNormEps": config.layernorm_epsilon,
            "numHeadsPerPartition": config.num_attention_heads // self.world_size,
            "hiddenSizePerHead": config.kv_channels,
            "numGroupsPerPartition": self.multi_query_group_num,
            "transKey": True,
            "residualAddScale": 1,
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
            self.config.hidden_size, device="npu", dtype=torch.float16)
        self.placeholder = torch.ones(1, device="npu")
        self.layer_id_input = [torch.tensor(
            [i], dtype=torch.int32, device="npu") for i in range(self.config.num_layers)]

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

    def _slice_tensors(self, key, tensor, tp_size, weight_type) -> List[torch.Tensor]:
        """
        Slice tensors according to key name
        """
        if weight_type == "weights":
            assert tensor.dtype in [torch.float16, torch.int8]
        elif weight_type == "bias":
            assert tensor.dtype in [torch.int32]
        elif weight_type == "scales":
            assert tensor.dtype in [torch.int64]
        else:
            raise ValueError()

        if tp_size == 1:
            return [tensor]

        sliced_tensor_list = []
        if 'dense_h_to_4h' in key:  # weight
            chunk_tensors = torch.chunk(tensor, tp_size * 2, dim=0)
            sliced_tensor_list = [torch.cat([chunk_tensors[i], chunk_tensors[i + tp_size]], dim=0)
                                  for i in range(tp_size)]
        elif 'dense_4h_to_h' in key or 'self_attention.dense' in key:  # weight
            if weight_type == "weights":
                sliced_tensor_list = torch.chunk(tensor, tp_size, dim=1)
            elif weight_type == "bias":
                zero_tensor = torch.zeros(tensor.shape, dtype=tensor.dtype)
                sliced_tensor_list = [tensor] + [zero_tensor] * (tp_size - 1)
            else:
                sliced_tensor_list = [tensor] * tp_size
        elif 'query_key_value' in key:  # weight and bias
            query_layer, key_layer, value_layer = tensor.split(
                [
                    self.hidden_size_per_attention_head * self.num_attention_heads_per_partition,
                    self.hidden_size_per_attention_head * self.num_multi_query_groups_per_partition,
                    self.hidden_size_per_attention_head * self.num_multi_query_groups_per_partition
                ],
                dim=0
            )
            kv_tp_size = min(
                tp_size, self.num_multi_query_groups_per_partition)
            query_list = torch.chunk(query_layer, tp_size, dim=0)
            key_list = torch.chunk(key_layer, kv_tp_size, dim=0)
            value_list = torch.chunk(value_layer, kv_tp_size, dim=0)
            sliced_tensor_list = [torch.cat(
                [query_list[i], key_list[i*kv_tp_size//tp_size], value_list[i*kv_tp_size//tp_size]], dim=0) for i in range(tp_size)]
        elif 'output_layer' in key:
            sliced_tensor_list = torch.chunk(tensor, tp_size, dim=0)
        else:
            sliced_tensor_list = [tensor] * tp_size
        return sliced_tensor_list

    def slice_tensors(self, state_dict, tp_size, weight_type) -> List[Dict[str, torch.Tensor]]:
        """
        Slice tensors
        """
        state_dict_list = [{} for i in range(tp_size)]
        for key, tensor in state_dict.items():
            sliced_tensor_list = self._slice_tensors(
                key, tensor, tp_size, weight_type)
            for i in range(tp_size):
                state_dict_list[i][key] = sliced_tensor_list[i]
        return state_dict_list


class Float(BaseManager):
    def __init__(self, config, pretrained_model_path="", **kwargs):
        super(Float, self).__init__(config, **kwargs)
        self.model_path = pretrained_model_path
        self.float_weight_path = Path(
            pretrained_model_path).joinpath("tensor_parallel")

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
        self.float_weight_path = f"{self.float_weight_path}_tp{tp_size}"
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
        state_dict.pop('transformer.rotary_pos_emb.inv_freq')
        weights_list = list(state_dict.values())
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


class Quant(BaseManager):
    def __init__(self, config, pretrained_model_path="", **kwargs):
        super(Quant, self).__init__(config, **kwargs)
        self.quant_weight_path = os.environ.get("QUANT_WEIGHT_PATH")
        self.float = Float(config, pretrained_model_path)

    def load_primal_weights(self) -> Dict[str, Dict[str, torch.Tensor]]:
        state_dict = {
            "input_offset": np.load(Path(self.quant_weight_path).joinpath("input_offset.npy"), allow_pickle=True).item(),
            "input_scale": np.load(Path(self.quant_weight_path).joinpath("input_scale.npy"), allow_pickle=True).item(),
            "quant_weight": np.load(Path(self.quant_weight_path).joinpath("quant_weight.npy"), allow_pickle=True).item(),
            "deq_scale": np.load(Path(self.quant_weight_path).joinpath("deq_scale.npy"), allow_pickle=True).item(),
            "fp_bias": np.load(Path(self.quant_weight_path).joinpath("fp_bias.npy"), allow_pickle=True).item(),
        }
        return state_dict

    def load_sliced_weights(self):
        quant_parallel_path = f"{self.quant_weight_path}/tp{self.config.world_size}"
        state_dict = {
            "quant_weight": np.load(Path(quant_parallel_path).joinpath(f"quant_weight{self.rank}.npy"), allow_pickle=True).item(),
            "bias": np.load(Path(quant_parallel_path).joinpath(f"new_bias{self.rank}.npy"), allow_pickle=True).item(),
            "deq_scale": np.load(Path(quant_parallel_path).joinpath(f"new_deq_scale{self.rank}.npy"), allow_pickle=True).item(),
        }
        return state_dict

    def save_sliced_weights(self, state_dict_list, tp_size) -> None:
        quant_parallel_path = f"{self.quant_weight_path}/tp{tp_size}"
        if Path(quant_parallel_path).exists():
            print(
                f"[info]: The parallel quant weights has exist in '{quant_parallel_path}'. Please remove it if you want to process quant weights again.")
        else:
            os.mkdir(quant_parallel_path)
        for i in range(tp_size):
            np.save(Path(quant_parallel_path).joinpath(
                f"quant_weight{i}.npy"), state_dict_list["quant_weight"][i])
            np.save(Path(quant_parallel_path).joinpath(
                f"new_bias{i}.npy"), state_dict_list["bias"][i])
            np.save(Path(quant_parallel_path).joinpath(
                f"new_deq_scale{i}.npy"), state_dict_list["deq_scale"][i])

    def bias_correction(self, fp_bias_dict, quant_weight_dict, input_offset_dict, deq_scale_dict) -> Dict[str, torch.Tensor]:
        """
        待量化工具改进后,该函数可删除
        """
        new_bias_dict = {}
        for key in fp_bias_dict.keys():
            new_bias_dict[key] = fp_bias_dict[key].npu()/deq_scale_dict[key].npu(
            ) - quant_weight_dict[key].to(torch.float32).npu().sum(dim=1) * float(input_offset_dict[key])
            new_bias_dict[key] = new_bias_dict[key].detach().to(torch.int32)
        return new_bias_dict

    def deq_scale_process(self, deq_scale_dict) -> Dict[str, torch.Tensor]:
        """
        待量化工具改进后,该函数可删除
        """
        new_deq_scale_dict = {}
        for key, deq_scale in deq_scale_dict.items():
            deq_scale = deq_scale.numpy()
            new_deq_scale = np.frombuffer(deq_scale.tobytes(), dtype=np.uint32)
            new_deq_scale_dict.setdefault(
                key, torch.tensor(new_deq_scale.astype(np.int64)))
        return new_deq_scale_dict

    def process_weights(self, tp_size) -> None:
        self.float.process_weights(tp_size)  # 处理浮点权重
        state_dict = self.load_primal_weights()
        # >>> 待量化工具改进后,这段可删除
        state_dict["bias"] = self.bias_correction(
            state_dict["fp_bias"],
            state_dict["quant_weight"],
            state_dict["input_offset"],
            state_dict["deq_scale"],
        )
        state_dict["deq_scale"] = self.deq_scale_process(
            state_dict["deq_scale"])
        # <<<
        state_dict_list = {
            "quant_weight": self.slice_tensors(state_dict["quant_weight"], tp_size, weight_type="weights"),
            "bias": self.slice_tensors(state_dict["bias"], tp_size, weight_type="bias"),
            "deq_scale": self.slice_tensors(state_dict["deq_scale"], tp_size, weight_type="scales"),
        }
        self.save_sliced_weights(state_dict_list, tp_size)
        print(
            f"[info]: The processed quant weights has been saved to '{self.quant_weight_path}'.")

    def init_param(self, is_encoder=True) -> str:
        state_dict = self.load_primal_weights()
        input_scale_dict = state_dict["input_scale"]
        input_offset_dict = state_dict["input_offset"]

        qkv_input_scale = []
        qkv_input_offset = []
        dense_input_scale = []
        dense_input_offset = []
        self_ln_input_scale = []
        self_ln_input_offset = []
        ffn_out_input_scale = []
        ffn_out_input_offset = []

        for i in range(self.config.num_layers):
            if i in self.config.float_layers_id:
                qkv_input_scale.append(float(0))
                qkv_input_offset.append(float(0))
                dense_input_scale.append(float(0))
                dense_input_offset.append(float(0))
                self_ln_input_scale.append(float(0))
                self_ln_input_offset.append(float(0))
                ffn_out_input_scale.append(float(0))
                ffn_out_input_offset.append(float(0))
            else:
                query_key_value_name = f"transformer.encoder.layers.{i}.self_attention.query_key_value"
                dense_name = f"transformer.encoder.layers.{i}.self_attention.dense"
                dense_h_to_4h_name = f"transformer.encoder.layers.{i}.mlp.dense_h_to_4h"
                dense_4h_to_h_name = f"transformer.encoder.layers.{i}.mlp.dense_4h_to_h"
                qkv_input_scale.append(
                    float(1 / input_scale_dict[query_key_value_name]))
                qkv_input_offset.append(
                    int(input_offset_dict[query_key_value_name]))
                dense_input_scale.append(
                    float(1 / input_scale_dict[dense_name]))
                dense_input_offset.append(int(input_offset_dict[dense_name]))
                self_ln_input_scale.append(
                    float(1 / input_scale_dict[dense_h_to_4h_name]))
                self_ln_input_offset.append(
                    int(input_offset_dict[dense_h_to_4h_name]))
                ffn_out_input_scale.append(
                    float(1 / input_scale_dict[dense_4h_to_h_name]))
                ffn_out_input_offset.append(
                    int(input_offset_dict[dense_4h_to_h_name]))

        param_dict = {
            "isEncoder": is_encoder,
            "quantmodel": True,
            "correctNodeId": self.config.float_layers_id[0],
            "qkvInputScale": qkv_input_scale,
            "qkvInputOffset": qkv_input_offset,
            "denseInputScale": dense_input_scale,
            "denseInputOffset": dense_input_offset,
            "selfLnInputScale": self_ln_input_scale,
            "selfLnInputOffset": self_ln_input_offset,
            "ffnOutInputScale": ffn_out_input_scale,
            "ffnOutInputOffset": ffn_out_input_offset,
        }

        self.param_dict.update(param_dict)
        return json.dumps(self.param_dict)

    def init_weights(self, state_dict, is_format_nz=False) -> List[torch.Tensor]:
        weights_list = [
            state_dict['transformer.embedding.word_embeddings.weight']]

        # load quant weights
        quant_state_dict = self.load_sliced_weights()
        quant_weight_dict = quant_state_dict["quant_weight"]
        new_bias_dict = quant_state_dict["bias"]
        deq_scale_dict = quant_state_dict["deq_scale"]

        # adapt for NZ format
        if is_format_nz:
            transdata_operation = torch.classes.OperationTorch.OperationTorch(
                "TransdataOperation")
            transdata_param = json.dumps({})
            transdata_operation.set_param(transdata_param)
            for k, v in quant_weight_dict.items():
                quant_weight_dict[k] = transdata_operation.execute([v.npu()])[
                    0]

        for i in range(self.config.num_layers):
            if i in self.config.float_layers_id:
                for k, v in state_dict.items():
                    if f'.{i}.' in k:
                        weights_list.append(v)
            else:
                query_key_value_name = f"transformer.encoder.layers.{i}.self_attention.query_key_value"
                dense_name = f"transformer.encoder.layers.{i}.self_attention.dense"
                dense_h_to_4h_name = f"transformer.encoder.layers.{i}.mlp.dense_h_to_4h"
                dense_4h_to_h_name = f"transformer.encoder.layers.{i}.mlp.dense_4h_to_h"

                weights_list.append(
                    state_dict[f"transformer.encoder.layers.{i}.input_layernorm.weight"])
                weights_list.append(
                    quant_weight_dict[query_key_value_name].npu())
                weights_list.append(new_bias_dict[query_key_value_name].npu())
                weights_list.append(quant_weight_dict[dense_name].npu())
                weights_list.append(
                    state_dict[f"transformer.encoder.layers.{i}.post_attention_layernorm.weight"])
                weights_list.append(
                    quant_weight_dict[dense_h_to_4h_name].npu())
                weights_list.append(
                    quant_weight_dict[dense_4h_to_h_name].npu())

                weights_list.append(deq_scale_dict[query_key_value_name].npu())
                weights_list.append(deq_scale_dict[dense_name].npu())
                weights_list.append(new_bias_dict[dense_name].npu())
                weights_list.append(deq_scale_dict[dense_h_to_4h_name].npu())
                weights_list.append(new_bias_dict[dense_h_to_4h_name].npu())
                weights_list.append(deq_scale_dict[dense_4h_to_h_name].npu())
                weights_list.append(new_bias_dict[dense_4h_to_h_name].npu())

        weights_list.append(
            state_dict["transformer.encoder.final_layernorm.weight"])
        weights_list.append(state_dict["transformer.output_layer.weight"])
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
            self.in_beta,
            self.placeholder,
        ]
        inputs_list.extend(self.layer_id_input)
        return inputs_list


class Sparse(BaseManager):
    def __init__(self, config, pretrained_model_path="", **kwargs):
        super(Sparse, self).__init__(config, **kwargs)
        self.compress_weight_path = os.environ.get("COMPRESS_WEIGHT_PATH")
        self.quant = Quant(config, pretrained_model_path, **kwargs)

    def load_primal_weights(self):
        """
        当前稀疏权重生成顺序：量化工具导出->模型脚本切分->量化工具压缩，故该函数暂不涉及。
        """
        pass

    def _read_dat_file(self, data_dir, is_compress_info=False):
        data_dict = {}
        for file_name in os.listdir(data_dir):
            weight_name = file_name[:-4]
            if is_compress_info:
                data = np.fromfile(os.path.join(
                    data_dir, file_name), dtype=np.int64)
            else:
                data = np.fromfile(os.path.join(
                    data_dir, file_name), dtype=np.int8)
            data_dict.setdefault(weight_name, torch.tensor(data))
        return data_dict

    def load_sliced_weights(self):
        compress_w_path = Path(self.compress_weight_path).joinpath(
            f"compress{self.rank}", "weight")
        compress_index_path = Path(self.compress_weight_path).joinpath(
            f"compress{self.rank}", "index")
        compress_info_path = Path(self.compress_weight_path).joinpath(
            f"compress{self.rank}", "info")
        state_dict = {
            "compress_weight": self._read_dat_file(compress_w_path),
            "compress_info": self._read_dat_file(compress_info_path, is_compress_info=True),
            "compress_index": self._read_dat_file(compress_index_path),
            "input_offset": np.load(Path(self.quant.quant_weight_path).joinpath("input_offset.npy"), allow_pickle=True).item(),
        }
        return state_dict

    def save_sliced_weights(self):
        """
        当前稀疏权重生成顺序：量化工具导出->模型脚本切分->量化工具压缩，故该函数暂不涉及。
        """
        pass

    def process_weights(self, tp_size):
        self.quant.process_weights(tp_size)

    def init_param(self, is_encoder=True) -> str:
        param_dict = json.loads(self.quant.init_param(is_encoder=is_encoder))
        param_dict["isSparse"] = True
        return json.dumps(param_dict)

    def init_weights(self, state_dict, is_format_nz=False) -> List[torch.Tensor]:
        weights_list = [
            state_dict['transformer.embedding.word_embeddings.weight']]

        # load quant and compress weights
        quant_state_dict = self.quant.load_sliced_weights()  # 双芯
        compress_state_dict = self.load_sliced_weights()
        new_bias_dict = quant_state_dict["bias"]
        deq_scale_dict = quant_state_dict["deq_scale"]
        compress_weight_dict = compress_state_dict["compress_weight"]
        compress_info_dict = compress_state_dict["compress_info"]
        compress_index_dict = compress_state_dict["compress_index"]
        offset_x_dict = compress_state_dict["input_offset"]

        # weights_list
        for i in range(self.config.num_layers):
            if i in self.config.float_layers_id:
                for k, v in state_dict.items():
                    if f'.{i}.' in k:
                        weights_list.append(v)
            else:
                query_key_value_name = f"transformer.encoder.layers.{i}.self_attention.query_key_value"
                dense_name = f"transformer.encoder.layers.{i}.self_attention.dense"
                dense_h_to_4h_name = f"transformer.encoder.layers.{i}.mlp.dense_h_to_4h"
                dense_4h_to_h_name = f"transformer.encoder.layers.{i}.mlp.dense_4h_to_h"

                weights_list.append(
                    state_dict[f"transformer.encoder.layers.{i}.input_layernorm.weight"])
                weights_list.append(
                    compress_weight_dict[query_key_value_name].npu())
                weights_list.append(new_bias_dict[query_key_value_name].npu())
                weights_list.append(compress_weight_dict[dense_name].npu())
                weights_list.append(
                    state_dict[f"transformer.encoder.layers.{i}.post_attention_layernorm.weight"])
                weights_list.append(
                    compress_weight_dict[dense_h_to_4h_name].npu())
                weights_list.append(
                    compress_weight_dict[dense_4h_to_h_name].npu())

                weights_list.append(deq_scale_dict[query_key_value_name].npu())
                weights_list.append(deq_scale_dict[dense_name].npu())
                weights_list.append(new_bias_dict[dense_name].npu())
                weights_list.append(deq_scale_dict[dense_h_to_4h_name].npu())
                weights_list.append(new_bias_dict[dense_h_to_4h_name].npu())
                weights_list.append(deq_scale_dict[dense_4h_to_h_name].npu())
                weights_list.append(new_bias_dict[dense_4h_to_h_name].npu())

                for layer_name in [query_key_value_name, dense_name, dense_h_to_4h_name, dense_4h_to_h_name]:
                    weights_list.append(compress_index_dict[layer_name].npu())
                    weights_list.append(
                        offset_x_dict[layer_name].to(torch.int32).npu())
                    weights_list.append(compress_info_dict[layer_name].npu())

        weights_list.append(
            state_dict["transformer.encoder.final_layernorm.weight"])
        weights_list.append(state_dict["transformer.output_layer.weight"])
        return weights_list

    def init_inputs(self, inputs_dict) -> List[torch.Tensor]:
        inputs_list = self.quant.init_inputs(inputs_dict)
        return inputs_list


class ModeManager:
    mode_dict = {
        'float': Float,
        'quant': Quant,
        'sparse': Sparse
    }

    @classmethod
    def get_manager(cls, mode):
        if mode not in cls.mode_dict:
            return ValueError("only support 'float', 'quant' and 'sparse' mode")
        return cls.mode_dict[mode]
