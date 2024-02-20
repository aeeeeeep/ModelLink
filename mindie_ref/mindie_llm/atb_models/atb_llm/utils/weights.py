# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import json
import math
import os
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open, SafetensorError

from .hub import weight_files
from .log import logger


class Weights:
    def __init__(
            self,
            model_name_or_path,
            device,
            dtype,
            process_group,
            quantize=None,
            revision: Optional[str] = None,
            extension: Optional[str] = ".safetensors",
            aliases: Optional[Dict[str, List[str]]] = None
    ):
        self.filenames = weight_files(model_name_or_path, revision=revision, extension=extension)
        routing = {}
        for filename in self.filenames:
            with safe_open(filename, framework="pytorch") as f:
                for k in f.keys():
                    if k in routing:
                        logger.error(
                            f"Key {k} was found in multiple files: {filename} and {routing[k]}"
                        )
                        raise AssertionError
                    routing[k] = filename
        if aliases is None:
            aliases = {}
        self.aliases = aliases
        self.routing = routing
        self.device = device
        self.dtype = dtype
        self.process_group = process_group
        self._handles = {}

        self.init_quant_params(quantize, model_name_or_path)

        self.gptq_bits = None
        self.gptq_groupsize = None
        self.smooth_quant_act_scales = None

    @staticmethod
    def cut_weights(
            model,
            world_size,
            cut_W_pack_keys=None,
            cut_row_keys=None,
            cut_col_keys=None,
    ):
        if not cut_W_pack_keys:
            cut_W_pack_keys = ["W_pack"]
        if not cut_row_keys:
            cut_row_keys = ["gate_proj", "up_proj"]
        if not cut_col_keys:
            cut_col_keys = ["o_proj", "down_proj"]
        state_dict_list = [{} for _ in range(world_size)]
        for key, tensor in model.items():
            key_short = key.split(".")[-1]
            if key_short in cut_W_pack_keys:
                split_linear_size = 3  # q k v linear
                full_q_weights, full_k_weights, full_v_weights = torch.chunk(
                    tensor, split_linear_size, dim=0
                )
                cut_q_weights = torch.chunk(full_q_weights, world_size, dim=0)
                cut_k_weights = torch.chunk(full_k_weights, world_size, dim=0)
                cut_v_weights = torch.chunk(full_v_weights, world_size, dim=0)
                cut_tensor_list = []
                for i in range(world_size):
                    cut_tensor_list.append(
                        torch.concat(
                            (cut_q_weights[i], cut_k_weights[i], cut_v_weights[i]), dim=0
                        )
                    )
            elif key_short in cut_row_keys:
                cut_tensor_list = torch.chunk(tensor, world_size, dim=0)
            elif key_short in cut_col_keys:
                cut_tensor_list = torch.chunk(tensor, world_size, dim=1)
            else:
                cut_tensor_list = [tensor] * world_size
            for i in range(world_size):
                state_dict_list[i][key] = cut_tensor_list[i]
        return state_dict_list

    @staticmethod
    def cut_bias(
            bias,
            world_size,
            cut_W_pack_keys=None,
            cut_row_keys=None,
            cut_col_keys=None,
            is_bias=False,
    ):
        if not cut_W_pack_keys:
            cut_W_pack_keys = ["W_pack"]
        if not cut_row_keys:
            cut_row_keys = ["gate_proj", "up_proj"]
        if not cut_col_keys:
            cut_col_keys = ["o_proj", "down_proj"]
        state_dict_list = [{} for _ in range(world_size)]
        for key, tensor in bias.items():
            key_short = key.split(".")[-1]
            if key_short in cut_W_pack_keys:
                split_linear_size = 3  # q k v linear
                full_q_weights, full_k_weights, full_v_weights = torch.chunk(
                    tensor, split_linear_size, dim=0
                )
                cut_q_weights = torch.chunk(full_q_weights, world_size, dim=0)
                cut_k_weights = torch.chunk(full_k_weights, world_size, dim=0)
                cut_v_weights = torch.chunk(full_v_weights, world_size, dim=0)
                cut_tensor_list = []
                for i in range(world_size):
                    cut_tensor_list.append(
                        torch.concat(
                            (cut_q_weights[i], cut_k_weights[i], cut_v_weights[i]), dim=0
                        )
                    )
            elif key_short in cut_row_keys:
                cut_tensor_list = torch.chunk(tensor, world_size, dim=0)
            elif key_short in cut_col_keys:
                if is_bias:
                    try:
                        tensor = tensor / world_size
                    except ZeroDivisionError as e:
                        raise ZeroDivisionError from e
                cut_tensor_list = [tensor] * world_size
            for i in range(world_size):
                state_dict_list[i][key] = cut_tensor_list[i]
        return state_dict_list

    @staticmethod
    def bias_correction(fp_bias, quant_weight, input_offset, deq_scale):
        try:
            bias_correction = fp_bias.npu() / deq_scale.npu() - quant_weight.to(
                torch.float32
            ).npu().sum(dim=1) * float(input_offset)
        except ZeroDivisionError as e:
            raise ZeroDivisionError from e
        return bias_correction

    @staticmethod
    def process_deq_scale(deq_scale_dict):
        new_deq_scale_dict = {}
        for key, deq_scale in deq_scale_dict.items():
            deq_scale = deq_scale.numpy()
            new_deq_scale = np.frombuffer(deq_scale.tobytes(), dtype=np.int32)
            new_deq_scale_dict.setdefault(key, torch.tensor(new_deq_scale.astype(np.int64)))
        return new_deq_scale_dict

    def init_quant_params(self, quantize, model_name_or_path):
        if quantize == "gptq":
            self._set_gptq_params(model_name_or_path)

        if quantize == "smooth_quant":
            self._set_smooth_quant_params(model_name_or_path)

    def get_filename(self, tensor_name: str) -> (str, str):
        filename = self.routing.get(tensor_name, None)
        if filename is None:
            aliases = self.aliases.get(tensor_name, [])
            for alias in aliases:
                filename = self.routing.get(alias, None)
                if filename is not None:
                    return str(filename), alias
            logger.error(f"weight {tensor_name} does not exist")
            raise AssertionError
        return str(filename), tensor_name

    def get_shape(self, tensor_name: str):
        return self._get_slice(tensor_name).get_shape()

    def get_tensor(self, tensor_name: str):
        filename, tensor_name = self.get_filename(tensor_name)
        f = self._get_handle(filename)
        tensor = f.get_tensor(tensor_name)
        if tensor.dtype not in [torch.int8]:
            tensor = tensor.to(dtype=self.dtype)
        tensor = tensor.to(device=self.device)
        if tensor.dtype not in [torch.int32, torch.int64]:
            tensor = tensor.to(dtype=self.dtype)
        return tensor

    def get_whole_tensor(self, tensor_name: str, dim: int):
        slice_ = self._get_slice(tensor_name)

        start = 0
        stop = slice_.get_shape()[dim]

        if dim == 0:
            tensor = slice_[start:stop]
        elif dim == 1:
            tensor = slice_[:, start:stop]
        else:
            logger.error("Let's make that generic when needed")
            raise AssertionError
        if tensor.dtype not in [torch.int8]:
            tensor = tensor.to(dtype=self.dtype)
        return tensor

    def get_partial_sharded(self, tensor_name: str, dim: int, gqa_size: int = 1):
        world_size = self.process_group.size()
        rank = self.process_group.rank()

        slice_ = self._get_slice(tensor_name)
        size = slice_.get_shape()[dim]

        group_size = size // gqa_size
        if group_size >= world_size:
            block_size = size // world_size
            start = rank * block_size
            stop = (rank + 1) * block_size
        else:
            block_size = gqa_size
            start = (rank // (world_size // group_size)) * block_size
            stop = ((rank // (world_size // group_size)) + 1) * block_size
        
        if "c_attn.bias" in tensor_name:
            b = slice_[:]
            single_size = b.shape[0] // 3
            head_size = 128
            head_num = single_size // head_size
            rank_heads = math.ceil(head_num / world_size)
            if rank != world_size - 1:
                start = rank * (rank_heads * head_size)
                stop = (rank + 1) * (rank_heads * head_size)
                bq = slice_[start:stop]
                bk = slice_[start + single_size:stop + single_size]
                bv = slice_[start + 2 * single_size:stop + 2 * single_size]
            else:
                # last rank
                start = rank * (rank_heads * head_size)
                stop = head_num * head_size
                bq = slice_[start:stop]
                bk = slice_[start + single_size:stop + single_size]
                bv = slice_[start + 2 * single_size:stop + 2 * single_size]
            b_ = torch.cat([bq, bk, bv], dim=0)
            return b_

        if dim == 0:
            tensor = slice_[start:stop]
        elif dim == 1:
            tensor = slice_[:, start:stop]
        else:
            logger.error("Let's make that generic when needed")
            raise AssertionError
        if tensor.dtype not in [torch.int8]:
            tensor = tensor.to(dtype=self.dtype)
        return tensor

    def get_partial_sharded_padding(self, tensor_name: str, dim: int, gqa_size=1):
        world_size = self.process_group.size()
        rank = self.process_group.rank()

        slice_ = self._get_slice(tensor_name)
        size = slice_.get_shape()[dim]

        head_num = size // gqa_size
        block_head_num = (head_num + world_size - 1) // world_size

        block_size = block_head_num * gqa_size

        start = rank * block_size
        stop = (rank + 1) * block_size

        if rank != world_size - 1:
            if dim == 0:
                tensor = slice_[start:stop]
            elif dim == 1:
                tensor = slice_[:, start:stop]
            else:
                logger.error("Let's make that generic when needed")
                raise AssertionError
            return tensor
        else:
            if dim == 0:
                tensor = slice_[start:]
            elif dim == 1:
                tensor = slice_[:, start:]
            else:
                logger.error("Let's make that generic when needed")
                raise AssertionError
            dim0, dim1 = tensor.shape
            if dim == 0:
                dim0 = block_size
            else:
                dim1 = block_size

            tensor_zeros = torch.zeros(size=(dim0, dim1), dtype=tensor.dtype, device=tensor.device)
            tensor_zeros[:tensor.shape[0], :tensor.shape[1]] = tensor
            tensor = tensor_zeros
            return tensor

    def get_sharded(self, tensor_name: str, dim: int, gqa_size: int = 1):
        slice_ = self._get_slice(tensor_name)
        world_size = self.process_group.size()
        size = slice_.get_shape()[dim]
        if (size // gqa_size) % world_size == 0 or world_size % (size // gqa_size) == 0:
            return self.get_partial_sharded(tensor_name, dim, gqa_size)
        else:
            return self.get_partial_sharded_padding(tensor_name, dim, gqa_size)

    def get_smooth_quant_sharded(self, tensor_name: str, idx: int, dim: int, gqa_size: int = 1):
        slice_ = self.smooth_quant_act_scales[tensor_name][idx]
        return slice_

    def get_weights_col_packed_qkv(self, prefix: str, quantize: str, head_size: int):
        """
        Highly specific when the underlying tensor is a simple cat of Q,K,V instead of being
        already alternating Q,K,V within the main tensor
        """
        if quantize == "gptq":
            try:
                qweight = self._get_qweight(f"{prefix}.qweight")
            except RuntimeError as err:
                raise RuntimeError(
                    "Cannot load `gptq` weight, "
                    "make sure the model is already quantized, "
                    "or quantize it with `text-generation-server quantize ORIGINAL_MODEL_ID NEW_MODEL_ID`"
                ) from err

            qzeros = self._get_qweight(f"{prefix}.qzeros")
            scales = self._get_qweight(f"{prefix}.scales")
            scales = scales.to(dtype=self.dtype)
            g_idx = self.get_tensor(f"{prefix}.g_idx")

            bits, groupsize = self._get_gptq_params()
            weight = (qweight, qzeros, scales, g_idx, bits, groupsize, False)
        else:
            slice_ = self._get_slice(f"{prefix}.weight")
            total_size = slice_.get_shape()[0]
            if total_size % 3 != 0:
                raise ValueError("Prepacked qkv is not divisible by 3")
            single_size = total_size // 3
            world_size = self.process_group.size()
            rank = self.process_group.rank()

            if head_size is None:
                if single_size % world_size != 0:
                    raise RuntimeError(f"Prepacked qkv cannot be sharded across {world_size} shards")
                try:
                    block_size = single_size // world_size
                except ZeroDivisionError as e:
                    raise ZeroDivisionError from e
                start = rank * block_size
                stop = (rank + 1) * block_size
                q = slice_[start:stop]
                k = slice_[start + single_size:stop + single_size]
                v = slice_[start + 2 * single_size:stop + 2 * single_size]
                weight = torch.cat([q, k, v], dim=0)
            else:
                try:
                    head_num = single_size // head_size
                    rank_heads = math.ceil(head_num / world_size)
                except ZeroDivisionError as e:
                    raise ZeroDivisionError from e
                if rank != world_size - 1:
                    start = rank * (rank_heads * head_size)
                    stop = (rank + 1) * (rank_heads * head_size)
                    q = slice_[start:stop]
                    k = slice_[start + single_size:stop + single_size]
                    v = slice_[start + 2 * single_size:stop + 2 * single_size]
                    weight = torch.cat([q, k, v], dim=0)
                else:
                    # last rank
                    start = rank * (rank_heads * head_size)
                    stop = head_num * head_size
                    q = slice_[start:stop]
                    k = slice_[start + single_size:stop + single_size]
                    v = slice_[start + 2 * single_size:stop + 2 * single_size]

                    # padding
                    q_zero = torch.zeros(size=(rank_heads * head_size, slice_.get_shape()[1]))
                    k_zero = torch.zeros(size=(rank_heads * head_size, slice_.get_shape()[1]))
                    v_zero = torch.zeros(size=(rank_heads * head_size, slice_.get_shape()[1]))
                    q_zero[:q.shape[0], :q.shape[1]] = q
                    k_zero[:k.shape[0], :k.shape[1]] = k
                    v_zero[:v.shape[0], :v.shape[1]] = v
                    weight = torch.cat([q_zero, k_zero, v_zero], dim=0)

            weight = weight.to(device=self.device)
            weight = weight.to(dtype=self.dtype)
        return weight

    def get_weights_packed_qkv_starcoder(self, prefix: str, quantize: str, q_size, kv_size, bias):
        """
        Highly specific when the underlying tensor is a simple cat of Q,K,V instead of being
        already alternating Q,K,V within the main tensor
        """
        if quantize == "gptq":
            try:
                qweight = self._get_qweight(f"{prefix}.qweight")
            except RuntimeError as err:
                logger.error(
                    f"Cannot load `gptq` weight, make sure the model is already quantized, "
                    f"or quantize it with `text-generation-server quantize ORIGINAL_MODEL_ID NEW_MODEL_ID`"
                )
                raise AssertionError from err

            qzeros = self._get_qweight(f"{prefix}.qzeros")
            scales = self._get_qweight(f"{prefix}.scales")
            g_idx = self.get_tensor(f"{prefix}.g_idx")

            bits, groupsize = self._get_gptq_params()
            weight = (qweight, qzeros, scales, g_idx, bits, groupsize, False)
        else:
            if bias:
                slice_ = self.get_tensor(f"{prefix}.bias")
            else:
                slice_ = self.get_tensor(f"{prefix}.weight")
            total_size = slice_.shape[0]
            q_single_size = q_size  # 6144
            kv_single_size = kv_size  # 256
            world_size = self.process_group.size()
            rank = self.process_group.rank()

            if q_single_size % world_size != 0:
                logger.error(f"Prepacked q cannot be sharded across {world_size} shards")
                raise AssertionError

            query_layer, kv_layer = slice_.split((q_single_size, kv_single_size), dim=0)
            query_list = torch.chunk(query_layer, world_size, dim=0)
            weight = torch.cat([query_list[rank], kv_layer], dim=0)
            weight = weight.to(device=self.device)
        return weight

    def get_multi_weights_col(self, prefixes: List[str], quantize: str, dim: int, gqa_size: int = 1):
        if quantize == "gptq":
            try:
                qweight = torch.cat(
                    [self.get_sharded(f"{p}.qweight", dim=1) for p in prefixes], dim=1
                )
            except RuntimeError as err:
                logger.error(
                    "Cannot load `gptq` weight, make sure the model is already quantized"
                )
                raise AssertionError from err

            qzeros = torch.cat(
                [self.get_sharded(f"{p}.qzeros", dim=1) for p in prefixes], dim=1
            )
            scales = torch.cat(
                [self.get_sharded(f"{p}.scales", dim=1) for p in prefixes], dim=1
            )
            w = [self.get_tensor(f"{p}.g_idx") for p in prefixes]
            for w2 in w[1:]:
                torch.testing.assert_close(w2, w[0])
            g_idx = w[0]

            bits, groupsize = self._get_gptq_params()
            weight = (qweight, qzeros, scales, g_idx, bits, groupsize, False)
        if quantize == "smooth_quant":
            qweight = torch.cat(
                [self.get_sharded(f"{p}.weight", dim=0, gqa_size=gqa_size) for p in prefixes], dim=dim
            )
            weight_scales = torch.cat(
                [self.get_sharded(f"{p}.scales", dim=0, gqa_size=gqa_size) for p in prefixes], dim=dim
            )
            weight_zeros = None
            act_scales_temp = [self.get_smooth_quant_sharded(f"{p}", idx=0, dim=0, gqa_size=gqa_size).reshape(1) for p in prefixes]
            for one_act_scale in act_scales_temp[1:]:
                if not torch.equal(act_scales_temp[0], one_act_scale):
                    raise ValueError(
                        f"`act_scales` are not equal: {act_scales_temp}"
                    )
            act_scales = (lambda x: x if None in x else x[0])(act_scales_temp)
            act_zeros = (lambda x: None if None in x else x[0])(
                [self.get_smooth_quant_sharded(f"{p}", idx=1, dim=0, gqa_size=gqa_size) for p in prefixes])
            weight = (qweight, weight_scales, weight_zeros, act_scales, act_zeros)
        else:
            w = [self.get_sharded(f"{p}.weight", dim=0, gqa_size=gqa_size) for p in prefixes]
            weight = torch.cat(w, dim=dim)
        return weight

    def get_weights_col_packed_qkv_glm(self, prefix: str, quantize: str, q_size, kv_size, bias):
        """
        Highly specific when the underlying tensor is a simple cat of Q,K,V instead of being
        already alternating Q,K,V within the main tensor
        """
        if quantize == "gptq":
            try:
                qweight = self._get_qweight(f"{prefix}.qweight")
            except RuntimeError as err:
                logger.error(
                    "Cannot load `gptq` weight, make sure the model is already quantized"
                )
                raise AssertionError from err

            qzeros = self._get_qweight(f"{prefix}.qzeros")
            scales = self._get_qweight(f"{prefix}.scales")
            g_idx = self.get_tensor(f"{prefix}.g_idx")

            bits, groupsize = self._get_gptq_params()
            weight = (qweight, qzeros, scales, g_idx, bits, groupsize, False)
        else:
            if bias:
                slice_ = self.get_tensor(f"{prefix}.bias")
            else:
                slice_ = self.get_tensor(f"{prefix}.weight")
            total_size = slice_.shape[0]
            if total_size % 3 != 0:
                logger.error("Prepacked qkv is not divisible by 3")
                raise AssertionError
            q_single_size = q_size
            kv_single_size = kv_size
            world_size = self.process_group.size()
            rank = self.process_group.rank()

            if q_single_size % world_size != 0:
                logger.error(f"Prepacked qkv cannot be sharded across {world_size} shards")
                raise AssertionError
            q_block_size = q_single_size // world_size
            q_start = rank * q_block_size
            q_stop = (rank + 1) * q_block_size

            kv_block_size = kv_single_size // world_size
            kv_start = rank * 2 * q_block_size
            kv_stop = (rank + 1) * kv_block_size
            query_layer, key_layer, value_layer = slice_.split(
                [
                    q_single_size,
                    kv_single_size,
                    kv_single_size
                ],
                dim=0
            )
            query_list = torch.chunk(query_layer, world_size, dim=0)
            key_list = torch.chunk(key_layer, world_size, dim=0)
            value_list = torch.chunk(value_layer, world_size, dim=0)
            q, k, v = query_list[rank], key_list[rank], value_list[rank]
            weight = torch.cat([q, k, v], dim=0)
            weight = weight.to(device=self.device)
        return weight

    def get_gate_up_glm(self, prefix, size):
        world_size = self.process_group.size()
        rank = self.process_group.rank()
        gate_weight = self.get_tensor(f"{prefix}.weight")
        chunk_tensors = torch.chunk(gate_weight, world_size * size, dim=0)
        cut_tensor_list = [torch.cat([chunk_tensors[i], chunk_tensors[i + world_size]], dim=0)
                           for i in range(world_size)]
        return cut_tensor_list[rank]

    def get_multi_weights_row(self, prefix: str, quantize: str):
        if quantize == "gptq":
            use_exllama = True
            bits, groupsize = self._get_gptq_params()

            if bits != 4:
                use_exllama = False

            if self.process_group.size() > 1:
                g_idx = self.get_tensor(f"{prefix}.g_idx")
                if g_idx is not None:
                    if (
                            not torch.equal(
                                g_idx.cpu(),
                                torch.tensor(
                                    [i // groupsize for i in range(g_idx.shape[0])],
                                    dtype=torch.int32,
                                ),
                            )
                            and not (g_idx == 0).all()
                    ):
                        use_exllama = False

            try:
                qweight = self.get_sharded(f"{prefix}.qweight", dim=0)
            except RuntimeError as err:
                logger.error(
                    "Cannot load `gptq` weight, make sure the model is already quantized"
                )
                raise AssertionError from err

            from text_generation_server.utils.layers import HAS_EXLLAMA

            if use_exllama:
                if not HAS_EXLLAMA:
                    logger.warning(
                        "Exllama GPTQ cuda kernels (which are faster) could have been used, "
                        "but are not currently installed, try using BUILD_EXTENSIONS=True"
                    )
                    use_exllama = False
                else:
                    logger.info("Using exllama kernels")

            if use_exllama:
                if groupsize >= 0:
                    qzeros = self.get_sharded(f"{prefix}.qzeros", dim=0)
                    scales = self.get_sharded(f"{prefix}.scales", dim=0)
                else:
                    qzeros = self.get_tensor(f"{prefix}.qzeros")
                    scales = self.get_tensor(f"{prefix}.scales")

                if self.process_group.size() == 1:
                    g_idx = self.get_tensor(f"{prefix}.g_idx")
                else:
                    g_idx = None
            else:
                qzeros = self.get_tensor(f"{prefix}.qzeros")
                scales = self.get_tensor(f"{prefix}.scales")
                g_idx = self.get_sharded(f"{prefix}.g_idx", dim=0)

            weight = (qweight, qzeros, scales, g_idx, bits, groupsize, use_exllama)

        elif quantize == "smooth_quant":
            qweight = self.get_sharded(f"{prefix}.weight", dim=1)
            weight_scales = self.get_sharded(f"{prefix}.scales", dim=1)
            weight_zeros = None
            act_scales = self.get_smooth_quant_sharded(f"{prefix}", idx=0, dim=1)
            act_zeros = self.get_smooth_quant_sharded(f"{prefix}", idx=1, dim=1)
            weight = (qweight, weight_scales, weight_zeros, act_scales, act_zeros)
        else:
            weight = self.get_sharded(f"{prefix}.weight", dim=1)
        return weight

    def _get_handle(self, filename):
        if filename not in self._handles:
            f = safe_open(filename, framework="pytorch")
            self._handles[filename] = f
        return self._handles[filename]

    def _get_slice(self, tensor_name: str):
        filename, tensor_name = self.get_filename(tensor_name)
        f = self._get_handle(filename)
        slice_ = f.get_slice(tensor_name)
        return slice_

    def _get_gptq_params(self) -> Tuple[int, int]:
        try:
            bits = self.get_tensor("gptq_bits").item()
            groupsize = self.get_tensor("gptq_groupsize").item()
        except (SafetensorError, RuntimeError) as e:
            try:
                bits = self.gptq_bits
                groupsize = self.gptq_groupsize
            except Exception as err:
                raise AssertionError from err

        return bits, groupsize

    def _set_gptq_params(self, model_id):
        try:
            filename = hf_hub_download(model_id, filename="quantize_config.json")
            with open(filename, "r") as f:
                data = json.load(f)
            self.gptq_bits = data["bits"]
            self.gptq_groupsize = data["group_size"]
        except Exception as err:
            raise AssertionError from err

    def _set_smooth_quant_params(self, model_id):
        try:
            filename = os.path.join(model_id, 'act_scales_zero.pt')
            act_scales = torch.load(filename)
            self.smooth_quant_act_scales = act_scales
        except Exception as err:
            raise AssertionError from err
