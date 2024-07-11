import unittest
import sys
import os
import subprocess
from pathlib import Path
import torch

from utils import judge_expression
import modellink


class CovertCkptFromHuggingfaceArgs:
    model_type = "GPT"
    loader = "llama2_hf"
    saver = "megatron"
    target_tensor_parallel_size = "8"
    load_dir = "/home/dataset/ci_engineering/llama-2-7b-hf"
    save_dir = "/home/dataset/ci_engineering/llama2-7B-tp8-pp1"
    tokenizer_model = "/home/dataset/ci_engineering/llama-2-7b-hf/tokenizer.model"


class CovertVppCkptFromHuggingfaceArgs:
    model_type = "GPT"
    loader = "llama2_hf"
    saver = "megatron"
    target_tensor_parallel_size = "2"
    target_pipeline_parallel_size = "4"
    load_dir = "/home/dataset/ci_engineering/llama-2-7b-hf"
    save_dir = "/home/dataset/ci_engineering/llama2-7B-tp2-pp4-vpp4"
    tokenizer_model = "/home/dataset/ci_engineering/llama-2-7b-hf/tokenizer.model"
    num_layers_per_virtual_pipeline_stage = "2"


class CovertDynamicCkptFromHuggingfaceArgs:
    model_type = "GPT"
    loader = "llama2_hf"
    saver = "megatron"
    target_tensor_parallel_size = "2"
    target_pipeline_parallel_size = "4"
    load_dir = "/home/dataset/ci_engineering/llama-2-7b-hf/"
    save_dir = "/home/dataset/ci_engineering/llama-2-7b-mg-tp2-pp4-dynamic-test/"
    base_dir = "/home/dataset/ci_engineering/llama-2-7b-mg-tp2-pp4-dynamic-base/"
    tokenizer_model = "/home/dataset/ci_engineering/llama-2-7b-hf/tokenizer.model"
    num_layer_list = '6,8,8,10'


class TestConvertCkptFromHuggingface:

    def test_file_exsit(self):
        args = CovertCkptFromHuggingfaceArgs()
        """
        Test if the file in the `--load-dir` exsit, including `.bin`, `.json`...
        """
        bin_file = 0
        for file_name in os.listdir(args.load_dir):
            if file_name.endswith(".bin"):
                bin_file += 1
        judge_expression(bin_file == 2)
        judge_expression(os.path.exists(os.path.join(args.load_dir, "pytorch_model.bin.index.json")))

    def test_convert_dynamic_weights_form_huggingface(self):
        from utils import weight_compare
        args = CovertDynamicCkptFromHuggingfaceArgs()
        """
        Test whether the weight to be converted as we want in `--save-dir`. We will check the model layer name, 
        including embedding, final_norm, output and encoder. In the encoder, there will be some different layers 
        to compose the unique transformer layer and all these layer stack to compose the entity of the model.
        """
        base_dir = Path(__file__).absolute().parent.parent.parent
        file_path = os.path.join(base_dir, "tools/checkpoint/convert_ckpt.py")
        arguments = [
            "--model-type", args.model_type,
            "--loader", args.loader,
            "--num-layer-list", args.num_layer_list,
            "--saver", args.saver,
            "--target-tensor-parallel-size", args.target_tensor_parallel_size,
            "--target-pipeline-parallel-size", args.target_pipeline_parallel_size,
            "--load-dir", args.load_dir,
            "--save-dir", args.save_dir,
            "--tokenizer-model", args.tokenizer_model
        ]
        subprocess.run(["python3", file_path] + arguments)
        judge_expression(weight_compare(args.base_dir, args.save_dir))

    def test_convert_weights_form_huggingface(self):
        args = CovertCkptFromHuggingfaceArgs()
        """
        Test whether the weight to be converted as we want in `--save-dir`. We will check the model layer name, 
        including embedding, final_norm, output and encoder. In the encoder, there will be some different layers 
        to compose the unique transformer layer and all these layer stack to compose the entity of the model.
        """
        base_dir = Path(__file__).absolute().parent.parent.parent
        file_path = os.path.join(base_dir, "tools/checkpoint/convert_ckpt.py")
        arguments = [
            "--model-type", args.model_type,
            "--loader", args.loader,
            "--saver", args.saver,
            "--target-tensor-parallel-size", args.target_tensor_parallel_size,
            "--load-dir", args.load_dir,
            "--save-dir", args.save_dir,
            "--tokenizer-model", args.tokenizer_model
        ]
        subprocess.run(["python3", file_path] + arguments)
        output_dir = os.path.join(args.save_dir, "iter_0000001")
        weight_content = torch.load(os.path.join(output_dir, "mp_rank_00/model_optim_rng.pt"))
        weight_common_content = weight_content['model']['language_model'] # extract commmon content

        # embedding, encoder, output_layer is three out layers.
        judge_expression(len(os.listdir(output_dir)) == int(args.target_tensor_parallel_size))
        judge_expression(weight_common_content['embedding']['word_embeddings']['weight'].size() == torch.Size([4000, 4096]))
        judge_expression(weight_common_content['encoder']['final_norm.weight'].size() == torch.Size([4096]))

        # encoder has a common final_norm and each one has folliowing six layers
        weight_common_content['encoder'].pop('final_norm.weight')
        judge_expression(weight_common_content['encoder']['layers.0.self_attention.query_key_value.weight'].size() == torch.Size([1536, 4096]))
        judge_expression(weight_common_content['encoder']['layers.0.self_attention.dense.weight'].size() == torch.Size([4096, 512]))
        judge_expression(weight_common_content['encoder']['layers.0.mlp.dense_h_to_4h.weight'].size() == torch.Size([2752, 4096]))
        judge_expression(weight_common_content['encoder']['layers.0.mlp.dense_4h_to_h.weight'].size() == torch.Size([4096, 1376]))
        judge_expression(weight_common_content['encoder']['layers.0.input_norm.weight'].size() == torch.Size([4096]))
        judge_expression(weight_common_content['encoder']['layers.0.post_attention_norm.weight'].size() == torch.Size([4096]))

        judge_expression(weight_common_content['output_layer']['weight'].size() == torch.Size([4000, 4096]))

    def test_convert_weights_vpp_form_huggingface(self):
        args = CovertVppCkptFromHuggingfaceArgs()
        """
        Test whether the weight to be converted as we want in `--save-dir`. We will check the model layer name, 
        including embedding, final_norm, output and encoder. In the encoder, there will be some different layers 
        to compose the unique transformer layer and all these layer stack to compose the entity of the model.
        """
        base_dir = Path(__file__).absolute().parent.parent.parent
        file_path = os.path.join(base_dir, "tools/checkpoint/convert_ckpt.py")
        arguments = [
            "--model-type", args.model_type,
            "--loader", args.loader,
            "--saver", args.saver,
            "--target-tensor-parallel-size", args.target_tensor_parallel_size,
            "--target-pipeline-parallel-size", args.target_pipeline_parallel_size,
            "--load-dir", args.load_dir,
            "--save-dir", args.save_dir,
            "--tokenizer-model", args.tokenizer_model,
            "--num-layers-per-virtual-pipeline-stage", args.num_layers_per_virtual_pipeline_stage
        ]
        subprocess.run(["python3", file_path] + arguments)
        output_dir = os.path.join(args.save_dir, "iter_0000001")
        weight_content = torch.load(os.path.join(output_dir, "mp_rank_00_000/model_optim_rng.pt"))
        weight_common_content = weight_content['model0']['language_model']  # extract commmon content

        judge_expression(len(os.listdir(output_dir)) == int(args.target_tensor_parallel_size) * int(args.target_pipeline_parallel_size))
        judge_expression(weight_common_content['embedding']['word_embeddings']['weight'].size() == torch.Size([16000, 4096]))

        judge_expression(weight_common_content['encoder']['layers.0.self_attention.query_key_value.weight'].size() == torch.Size([6144, 4096]))
        judge_expression(weight_common_content['encoder']['layers.0.self_attention.dense.weight'].size() == torch.Size([4096, 2048]))
        judge_expression(weight_common_content['encoder']['layers.0.mlp.dense_h_to_4h.weight'].size() == torch.Size([11008, 4096]))
        judge_expression(weight_common_content['encoder']['layers.0.mlp.dense_4h_to_h.weight'].size() == torch.Size([4096, 5504]))
        judge_expression(weight_common_content['encoder']['layers.0.input_norm.weight'].size() == torch.Size([4096]))
        judge_expression(weight_common_content['encoder']['layers.0.post_attention_norm.weight'].size() == torch.Size([4096]))
