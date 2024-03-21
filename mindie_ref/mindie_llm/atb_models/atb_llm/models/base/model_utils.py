# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.import collections
import collections
import json
import os
import re
import shutil
import stat
from typing import Union, Callable
import torch
from torch import Tensor, nn

from safetensors.torch import save_file as safe_save_file
from transformers.utils import is_safetensors_available
from transformers.modeling_utils import shard_checkpoint

from ...utils.weights import QUANTIZE_DTYPE_LIST

WEIGHTS_NAME = "pytorch_model.bin"
SAFE_WEIGHTS_NAME = "model.safetensors"


def unwrap_model_state_dict(state_dict):
    new_state_dict = {}
    for name, tensor in state_dict.items():
        new_name = name.replace('.linear.', '.')
        new_state_dict[new_name] = tensor
    return new_state_dict


class BaseModel(nn.Module):
    def save_pretrained(self,
                        save_directory,
                        max_shard_size: Union[int, str] = "10GB",
                        save_function: Callable = torch.save,
                        safe_serialization: bool = False):
        os.makedirs(save_directory, exist_ok=True)
        state_dict = unwrap_model_state_dict(self.state_dict())
        if safe_serialization:
            ptrs = collections.defaultdict(list)
            for name, tensor in state_dict.items():
                ident = (tensor.data_ptr(), tensor.device, tensor.shape, tensor.stride())
                ptrs[ident].append(name)

            shared_ptrs = {ptr: names for ptr, names in ptrs.items() if len(names) > 1}
            warn_names = set()
            for names in shared_ptrs.values():
                found = 0
                for name in names:
                    if name in state_dict:
                        found += 1
                        if found > 1:
                            del state_dict[name]
                            warn_names.add(name)
            weights_name = SAFE_WEIGHTS_NAME if safe_serialization else WEIGHTS_NAME

            shards, index = shard_checkpoint(state_dict, max_shard_size=max_shard_size, weights_name=weights_name)

            for filename in os.listdir(save_directory):
                full_filename = os.path.join(save_directory, filename)
                weights_no_suffix = weights_name.replace(".bin", "").replace(".safetensors", "")
                filename_no_suffix = filename.replace(".bin", "").replace(".safetensors", "")
                reg = re.compile(r"(.*?)-\d{5}-of-\d{5}")

                need_remove = (filename.startswith(weights_no_suffix) and
                               os.path.isfile(full_filename) and
                               filename not in shards.keys() and
                               reg.fullmatch(filename_no_suffix) is not None)
                if need_remove:
                    os.remove(full_filename)
            for shard_file, shard in shards.items():
                if safe_serialization:
                    safe_save_file(shard, os.path.join(save_directory, shard_file), metadata={"format": "pt"})
                else:
                    save_function(shard, os.path.join(save_directory, shard_file))
        if self.quantize:
            self.generate_description(save_directory)

    def generate_description(self, save_directory=None):
        model_description = {}
        state_dict = unwrap_model_state_dict(self.state_dict())
        quantize_type = self.quantize.upper()
        model_description['model_quant_type'] = quantize_type
        for name, tensor in state_dict.items():
            if '.weight' in name or '.bias' in name:
                if tensor.dtype in QUANTIZE_DTYPE_LIST:
                    model_description[name] = quantize_type
                else:
                    model_description[name] = 'FLOAT'
            else:
                model_description[name] = quantize_type
        if save_directory:
            os.makedirs(save_directory, exist_ok=True)
            save_path = os.path.join(save_directory, f'quant_model_description_{quantize_type.lower()}.json')
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            models = stat.S_IWUSR | stat.S_IRUSR
            with os.fdopen(os.open(save_path, flags, models), 'w', encoding='utf-8') as fw:
                json.dump(model_description, fw, indent=4)
        return model_description
