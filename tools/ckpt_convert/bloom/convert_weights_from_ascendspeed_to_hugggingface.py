# coding=utf-8
# 本脚本用途：把BLOOM 176B AscendSpeed模型格式转换成HuggingFace模型格式
# 使用样例：./convert_weights_from_ascendspeed_to_huggingface.sh

import argparse
import json
import os
import sys
import logging
from collections import namedtuple
import time
import functools
import shutil
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ascendspeed.data_classes import SaveAscendspeedModelConfig


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output-huggingface-model-dir", type=str, default="./output_huggingface_model_dir", help="output huggingface model weight dir")
    parser.add_argument("--ascendspeed-model-dir", type=str, default="./ascendspeed_model_dir", help="input ascendspeed model weight dir")
    parser.add_argument("--make-vocab-size-divisible-by", type=int, default=128,
                        help="should be consistent with ascendspeed")
    parser.add_argument("--tensor-model-parallel-size", type=int, default=1, help="degree of tensor model parallel")
    parser.add_argument("--pipeline-model-parallel-size", type=int, default=1,
                        help="degree of pipeline model parallel")
    parser.add_argument("--type", type=str, choices=["7B", "176B"], default="7B")
    parser.add_argument("--deepspeed", action="store_true", default=True)
    parser.add_argument("--partition-layers", type=str, help="the partition method of model when pipeline is used")
    return parser.parse_args()


def cal_time(func):
    '''
    统计函数执行时间方法
    '''
    @functools.wraps(func)
    def inner(*args, **kwargs):
        start = time.perf_counter()
        ret = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"Execute {func.__name__}: time cost {round(end - start, 2)}s")
        return ret
    return inner


def row_concat(w_concat, w, tp, r):
    if w_concat is None:
        return w
    dim1 = w.shape[0]
    # 将两个张量w_concat和w沿着0进行拼接
    return torch.cat([w_concat, w], dim=0)


def column_concat(w_concat, w, tp, r):
    if w_concat is None:
        return w
    dim1 = w.shape[1]
    return torch.cat([w_concat, w], dim=1)


class AscendspeedToHuggingfaceConvert:
    
    model_config = {
        "7B": [30, 4096, 32], # num_layers, hiddern_size, num_attention_heads
        "176B": [70, 14336, 112]
    }
    
    
    def __init__(self, args):
        self.huggingface_model = {}
        self.layer_weight_idx = {}
        self.tp_size = args.tensor_model_parallel_size
        self.pp_size = args.pipeline_model_parallel_size
        self.model_type = args.type
        self.output_huggingface_model_dir = args.output_huggingface_model_dir
        self.make_vocab_size_divisible_by = args.make_vocab_size_divisible_by
        self.ascendspeed_model_dir = args.ascendspeed_model_dir
        self.pp_layers = self.get_partition_layers(args.partition_layers)
        self.init_huggingface_model()
        
    def get_partition_layers(self, partition_layers):
        if self.model_type == "7B" and self.pp_size == 1:
            return [30]
        elif self.model_type == "7B" and self.pp_size == 2:
            return [15, 15]
        else:
            return list(map(int, partition_layers.split(',')))
    
    def init_huggingface_model(self):
        model_index = {}
        params = ["input_layernorm.bias", "input_layernorm.weight", "mlp.dense_4h_to_h.bias", "mlp.dense_4h_to_h.weight", \
            "mlp.dense_h_to_4h.bias", "mlp.dense_h_to_4h.weight", "post_attention_layernorm.bias", "post_attention_layernorm.weight", \
            "self_attention.dense.bias", "self_attention.dense.weight", "self_attention.query_key_value.bias", "self_attention.query_key_value.weight"]
        
        for pp_rank in range(self.pp_size):
            for offset in range(self.pp_layers[pp_rank]):
                layer_id = sum(self.pp_layers[:pp_rank]) + offset
                dest_model_filepath = "pytorch_model_{:05d}-of-00072.bin".format(layer_id + 2)
                for param in params:
                    self.layer_weight_idx[f"h.{layer_id}.{param}"] = dest_model_filepath
                    
                self.huggingface_model[dest_model_filepath] = {}
        self.layer_weight_idx["ln_f.bias"] = "pytorch_model_00072-of-00072.bin"
        self.layer_weight_idx["ln_f.weight"] = "pytorch_model_00072-of-00072.bin"
        self.layer_weight_idx["word_embeddings.weight"] = "pytorch_model_00001-of-00072.bin"
        self.layer_weight_idx["word_embeddings_layernorm.bias"] = "pytorch_model_00001-of-00072.bin"
        self.layer_weight_idx["word_embeddings_layernorm.weight"] = "pytorch_model_00001-of-00072.bin"
        model_index["weight_map"] = self.layer_weight_idx
        model_index["metadata"] = {"total_size": 0}
        self.huggingface_model["pytorch_model_00072-of-00072.bin"] = {}
        self.huggingface_model["pytorch_model_00001-of-00072.bin"] = {}
        
        if not os.path.exists(self.output_huggingface_model_dir):
            os.makedirs(self.output_huggingface_model_dir)
            
        with os.fdopen(os.open(os.path.join(args.output_huggingface_model_dir, "pytorch_model.bin.index.json"), \
            os.O_WRONLY | os.O_CREAT, 0o640), 'w') as f:
            f.write(json.dumps(model_index, indent=4))
        
        config_files = ["config.json", "special_tokens_map.json", "tokenizer_config.json", "tokenizer.json"]
        for _file in config_files:
            srcfile = os.path.join(self.ascendspeed_model_dir, _file)
            if os.path.exists(srcfile):
                shutil.copy2(srcfile, self.output_huggingface_model_dir)
            else:
                print(f"warning: {srcfile} does not exist!")
    
    def set_huggingface_weight_by_name(self, layer_weight, w):
        '''
        设置huggingface权重信息，通过layer_weight_idx找到对应保存权重的二进制文件中
        '''
        self.huggingface_model[self.layer_weight_idx[layer_weight]][layer_weight] = w    
    
    def check_has_layer_model(self):
        one_layer_model_path_sample = os.path.join(self.ascendspeed_model_dir, "layer_01-model_00-model_states.pt")
        if os.path.exists(one_layer_model_path_sample):
            return True
        return False
        
    def convert_from_layer_model(self, pp_size, tp_size, num_layers):
        weights_dicts = {"word_embeddings": None, "self_attention_qkv_weight": {}, "self_attention_qkv_bias": {}, \
            "self_attention_dense_weight": {}, "mlp_dense_h_to_4h_weight": {}, "mlp_dense_h_to_4h_bias": {}, "mlp_dense_4h_to_h_weight": {}}
        
        for pp_rank in range(pp_size):
            for tp_rank in range(tp_size):
                if pp_rank == 0:
                    model_path = os.path.join(self.ascendspeed_model_dir, "layer_01-model_{:02d}-model_states.pt".format(tp_rank))
                    ascendspeed_model = torch.load(model_path, map_location="cpu")
                    
                    self.set_huggingface_weight_by_name("word_embeddings_layernorm.weight", ascendspeed_model["word_embeddings.norm.weight"])
                    self.set_huggingface_weight_by_name("word_embeddings_layernorm.bias", ascendspeed_model["word_embeddings.norm.bias"])
                    word_embeddings_read = ascendspeed_model["word_embeddings.weight"]
                    weights_dicts["word_embeddings"] = row_concat(weights_dicts["word_embeddings"], word_embeddings_read, tp_size, tp_rank)
                
                if pp_rank == pp_size - 1:
                    as_layer_id = num_layers + 4
                    model_path = os.path.join(self.ascendspeed_model_dir, "layer_{:02d}-model_{:02d}-model_states.pt".format(as_layer_id, tp_rank))
                    ascendspeed_model = torch.load(model_path, map_location="cpu")
                    self.set_huggingface_weight_by_name("ln_f.weight", ascendspeed_model["weight"])
                    self.set_huggingface_weight_by_name("ln_f.bias", ascendspeed_model["bias"])
                
                for i in range(self.pp_layers[pp_rank]):
                    layer_id = sum(self.pp_layers[:pp_rank]) + i
                    as_layer_id = layer_id + 3
                    model_path = os.path.join(self.ascendspeed_model_dir, "layer_{:02d}-model_{:02d}-model_states.pt".format(as_layer_id, tp_rank))
                    ascendspeed_model = torch.load(model_path, map_location="cpu")
                    
                    self.set_huggingface_weight_by_name(f"h.{layer_id}.input_layernorm.weight", ascendspeed_model["input_layernorm.weight"])
                    self.set_huggingface_weight_by_name(f"h.{layer_id}.input_layernorm.bias", ascendspeed_model["input_layernorm.bias"])
                    
                    self.set_huggingface_weight_by_name(f"h.{layer_id}.post_attention_layernorm.weight", ascendspeed_model["post_attention_layernorm.weight"])
                    self.set_huggingface_weight_by_name(f"h.{layer_id}.post_attention_layernorm.bias", ascendspeed_model["post_attention_layernorm.bias"])
                    self.set_huggingface_weight_by_name(f"h.{layer_id}.self_attention.dense.bias", ascendspeed_model["self_attention.dense.bias"])
                    self.set_huggingface_weight_by_name(f"h.{layer_id}.mlp.dense_4h_to_h.bias", ascendspeed_model["mlp.dense_4h_to_h.bias"])
                    
                    self_attention_qkv_weight = ascendspeed_model["self_attention.query_key_value.weight"]
                    self_attention_qkv_bias = ascendspeed_model["self_attention.query_key_value.bias"]
                    self_attention_dense_weight = ascendspeed_model["self_attention.dense.weight"]
                    mlp_dense_h_to_4h_weight = ascendspeed_model["mlp.dense_h_to_4h.weight"]
                    mlp_dense_h_to_4h_bias = ascendspeed_model["mlp.dense_h_to_4h.bias"]
                    mlp_dense_4h_to_h_weight = ascendspeed_model["mlp.dense_4h_to_h.weight"]
                    
                    if layer_id not in weights_dicts["self_attention_qkv_weight"]:
                        weights_dicts["self_attention_qkv_weight"][layer_id] = None
                        weights_dicts["self_attention_qkv_bias"][layer_id] = None
                        weights_dicts["self_attention_dense_weight"][layer_id] = None
                        weights_dicts["mlp_dense_h_to_4h_weight"][layer_id] = None
                        weights_dicts["mlp_dense_h_to_4h_bias"][layer_id] = None
                        weights_dicts["mlp_dense_4h_to_h_weight"][layer_id] = None
                    
                    weights_dicts["self_attention_qkv_weight"][layer_id] = row_concat(weights_dicts["self_attention_qkv_weight"][layer_id], self_attention_qkv_weight, tp_size, tp_rank)
                    weights_dicts["self_attention_qkv_bias"][layer_id] = row_concat(weights_dicts["self_attention_qkv_bias"][layer_id], self_attention_qkv_bias, tp_size, tp_rank)
                    weights_dicts["self_attention_dense_weight"][layer_id] = column_concat(weights_dicts["self_attention_dense_weight"][layer_id], self_attention_dense_weight, tp_size, tp_rank)
                    weights_dicts["mlp_dense_h_to_4h_weight"][layer_id] = row_concat(weights_dicts["mlp_dense_h_to_4h_weight"][layer_id], mlp_dense_h_to_4h_weight, tp_size, tp_rank)
                    weights_dicts["mlp_dense_h_to_4h_bias"][layer_id] = row_concat(weights_dicts["mlp_dense_h_to_4h_bias"][layer_id], mlp_dense_h_to_4h_bias, tp_size, tp_rank)
                    weights_dicts["mlp_dense_4h_to_h_weight"][layer_id] = column_concat(weights_dicts["mlp_dense_4h_to_h_weight"][layer_id], mlp_dense_4h_to_h_weight, tp_size, tp_rank)
                    
                    
        self.set_huggingface_weight_by_name("word_embeddings.weight", weights_dicts["word_embeddings"])
        for layer_id in weights_dicts["self_attention_qkv_weight"]:
            self.set_huggingface_weight_by_name(f"h.{layer_id}.self_attention.query_key_value.weight", weights_dicts["self_attention_qkv_weight"][layer_id])
            self.set_huggingface_weight_by_name(f"h.{layer_id}.self_attention.query_key_value.bias", weights_dicts["self_attention_qkv_bias"][layer_id])
            self.set_huggingface_weight_by_name(f"h.{layer_id}.self_attention.dense.weight", weights_dicts["self_attention_dense_weight"][layer_id])
            self.set_huggingface_weight_by_name(f"h.{layer_id}.mlp.dense_h_to_4h.weight", weights_dicts["mlp_dense_h_to_4h_weight"][layer_id])
            self.set_huggingface_weight_by_name(f"h.{layer_id}.mlp.dense_h_to_4h.bias", weights_dicts["mlp_dense_h_to_4h_bias"][layer_id])
            self.set_huggingface_weight_by_name(f"h.{layer_id}.mlp.dense_4h_to_h.weight", weights_dicts["mlp_dense_4h_to_h_weight"][layer_id])
        
        return True
        
    def convert_from_mprank_model(self, pp_size, tp_size, num_layers):
        weights_dicts = {"word_embeddings": None, "self_attention_qkv_weight": {}, "self_attention_qkv_bias": {}, \
            "self_attention_dense_weight": {}, "mlp_dense_h_to_4h_weight": {}, "mlp_dense_h_to_4h_bias": {}, "mlp_dense_4h_to_h_weight": {}}
        
        for pp_rank in range(pp_size):
            for tp_rank in range(tp_size):
                model_path = os.path.join(self.ascendspeed_model_dir, f"{'mp_rank_{:02d}'.format(pp_rank * tp_size + tp_rank)}_model_states.pt")
                if not os.path.exists(model_path):
                    print(f"Error! {model_path} does not exist")
                    return False
                as_pt_model = torch.load(model_path, map_location="cpu")
                rank_model = as_pt_model["module"]["module"]

                if pp_rank == 0:
                    
                    self.set_huggingface_weight_by_name("word_embeddings_layernorm.weight", rank_model["tied_modules.embed.word_embeddings.norm.weight"])
                    self.set_huggingface_weight_by_name("word_embeddings_layernorm.bias", rank_model["tied_modules.embed.word_embeddings.norm.bias"])
                    word_embeddings_read = rank_model["tied_modules.embed.word_embeddings.weight"]
                    weights_dicts["word_embeddings"] = row_concat(weights_dicts["word_embeddings"], word_embeddings_read, tp_size, tp_rank)
                
                if pp_rank == pp_size - 1:
                    as_layer_id = num_layers + 4
                    self.set_huggingface_weight_by_name("ln_f.weight", rank_model[f"{as_layer_id}.weight"])
                    self.set_huggingface_weight_by_name("ln_f.bias", rank_model[f"{as_layer_id}.bias"])
                
                for i in range(self.pp_layers[pp_rank]):
                    layer_id = sum(self.pp_layers[:pp_rank]) + i
                    as_layer_id = layer_id + 3
                    
                    self.set_huggingface_weight_by_name(f"h.{layer_id}.input_layernorm.weight", rank_model[f"{as_layer_id}.input_layernorm.weight"])
                    self.set_huggingface_weight_by_name(f"h.{layer_id}.input_layernorm.bias", rank_model[f"{as_layer_id}.input_layernorm.bias"])
                    
                    self.set_huggingface_weight_by_name(f"h.{layer_id}.post_attention_layernorm.weight", rank_model[f"{as_layer_id}.post_attention_layernorm.weight"])
                    self.set_huggingface_weight_by_name(f"h.{layer_id}.post_attention_layernorm.bias", rank_model[f"{as_layer_id}.post_attention_layernorm.bias"])
                    self.set_huggingface_weight_by_name(f"h.{layer_id}.self_attention.dense.bias", rank_model[f"{as_layer_id}.self_attention.dense.bias"])
                    self.set_huggingface_weight_by_name(f"h.{layer_id}.mlp.dense_4h_to_h.bias", rank_model[f"{as_layer_id}.mlp.dense_4h_to_h.bias"])
                    
                    self_attention_qkv_weight = rank_model[f"{as_layer_id}.self_attention.query_key_value.weight"]
                    self_attention_qkv_bias = rank_model[f"{as_layer_id}.self_attention.query_key_value.bias"]
                    self_attention_dense_weight = rank_model[f"{as_layer_id}.self_attention.dense.weight"]
                    mlp_dense_h_to_4h_weight = rank_model[f"{as_layer_id}.mlp.dense_h_to_4h.weight"]
                    mlp_dense_h_to_4h_bias = rank_model[f"{as_layer_id}.mlp.dense_h_to_4h.bias"]
                    mlp_dense_4h_to_h_weight = rank_model[f"{as_layer_id}.mlp.dense_4h_to_h.weight"]
                    
                    if layer_id not in weights_dicts["self_attention_qkv_weight"]:
                        weights_dicts["self_attention_qkv_weight"][layer_id] = None
                        weights_dicts["self_attention_qkv_bias"][layer_id] = None
                        weights_dicts["self_attention_dense_weight"][layer_id] = None
                        weights_dicts["mlp_dense_h_to_4h_weight"][layer_id] = None
                        weights_dicts["mlp_dense_h_to_4h_bias"][layer_id] = None
                        weights_dicts["mlp_dense_4h_to_h_weight"][layer_id] = None
                    
                    weights_dicts["self_attention_qkv_weight"][layer_id] = row_concat(weights_dicts["self_attention_qkv_weight"][layer_id], self_attention_qkv_weight, tp_size, tp_rank)
                    weights_dicts["self_attention_qkv_bias"][layer_id] = row_concat(weights_dicts["self_attention_qkv_bias"][layer_id], self_attention_qkv_bias, tp_size, tp_rank)
                    weights_dicts["self_attention_dense_weight"][layer_id] = column_concat(weights_dicts["self_attention_dense_weight"][layer_id], self_attention_dense_weight, tp_size, tp_rank)
                    weights_dicts["mlp_dense_h_to_4h_weight"][layer_id] = row_concat(weights_dicts["mlp_dense_h_to_4h_weight"][layer_id], mlp_dense_h_to_4h_weight, tp_size, tp_rank)
                    weights_dicts["mlp_dense_h_to_4h_bias"][layer_id] = row_concat(weights_dicts["mlp_dense_h_to_4h_bias"][layer_id], mlp_dense_h_to_4h_bias, tp_size, tp_rank)
                    weights_dicts["mlp_dense_4h_to_h_weight"][layer_id] = column_concat(weights_dicts["mlp_dense_4h_to_h_weight"][layer_id], mlp_dense_4h_to_h_weight, tp_size, tp_rank)
        
        self.set_huggingface_weight_by_name("word_embeddings.weight", weights_dicts["word_embeddings"])
        for layer_id in weights_dicts["self_attention_qkv_weight"]:
            self.set_huggingface_weight_by_name(f"h.{layer_id}.self_attention.query_key_value.weight", weights_dicts["self_attention_qkv_weight"][layer_id])
            self.set_huggingface_weight_by_name(f"h.{layer_id}.self_attention.query_key_value.bias", weights_dicts["self_attention_qkv_bias"][layer_id])
            self.set_huggingface_weight_by_name(f"h.{layer_id}.self_attention.dense.weight", weights_dicts["self_attention_dense_weight"][layer_id])
            self.set_huggingface_weight_by_name(f"h.{layer_id}.mlp.dense_h_to_4h.weight", weights_dicts["mlp_dense_h_to_4h_weight"][layer_id])
            self.set_huggingface_weight_by_name(f"h.{layer_id}.mlp.dense_h_to_4h.bias", weights_dicts["mlp_dense_h_to_4h_bias"][layer_id])
            self.set_huggingface_weight_by_name(f"h.{layer_id}.mlp.dense_4h_to_h.weight", weights_dicts["mlp_dense_4h_to_h_weight"][layer_id])
        
        return True

    @cal_time
    def generate_huggingface_weight(self):
        try:
            num_layer, _, _ = AscendspeedToHuggingfaceConvert.model_config[self.model_type]
        except KeyError:
            print(f"Error! {self.model_type} is not supported!")
            return False
        if self.check_has_layer_model():
            self.convert_from_layer_model(self.pp_size, self.tp_size, num_layer)
        else:
            self.convert_from_mprank_model(self.pp_size, self.tp_size, num_layer)
        os.makedirs(self.output_huggingface_model_dir, exist_ok=True)
        for file_name in self.huggingface_model:
            dest_path = os.path.join(self.output_huggingface_model_dir, file_name)
            print(f"Saving huggingface model to : {dest_path}")
            torch.save(self.huggingface_model[file_name], dest_path)

if __name__ == '__main__':
    args = get_args()
    coverter = AscendspeedToHuggingfaceConvert(args)
    coverter.generate_huggingface_weight()