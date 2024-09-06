import unittest
import sys
import os
import glob
import subprocess
from pathlib import Path
import torch
from safetensors.torch import load_file as safe_load_file

import modellink
from test_tools.utils import judge_expression


class CovertCkptFromMegatronArgs:
    model_type = "GPT"
    loader = "megatron"
    saver = "megatron"
    target_tensor_parallel_size = "1"
    save_dir = "/data/llama2-7B-tp1-pp1"
    load_dir = "/data/llama2-7B-tp8-pp1"


class CovertCkptDeepseekV2MegatronToHuggingfaceArgs:
    model_type = "GPT"
    model_type_hf = "deepseek2"
    loader = "mg_mcore"
    saver = "mg_mcore"
    target_tensor_parallel_size = "1"
    target_pipeline_parallel_size = "1"
    target_expert_parallel_size = "1"
    load_dir = "/data/deepseek2-236b-mg-tp1pp4ep8-mcore-base/"
    save_dir = "/data/deepseek2-236b-hf/"


class TestConvertCkptFromMegatron:

    def test_convert_ckpt_deepseek2_mg2hf(self):
        args = CovertCkptDeepseekV2MegatronToHuggingfaceArgs()

        base_dir = Path(__file__).absolute().parents[3]
        file_path = os.path.join(base_dir, "tools/checkpoint/convert_ckpt.py")
        arguments = [
            "--use-mcore-models",
            "--model-type", args.model_type,
            "--model-type-hf", args.model_type_hf,
            "--save-model-type", "huggingface",
            "--moe-grouped-gemm",
            "--loader", args.loader,
            "--saver", args.saver,
            "--target-tensor-parallel-size", args.target_tensor_parallel_size,
            "--target-pipeline-parallel-size", args.target_pipeline_parallel_size,
            "--target-expert-parallel-size", args.target_expert_parallel_size,
            "--save-dir", args.save_dir,
            "--load-dir", args.load_dir,
            "--params-dtype", "bf16"
        ]
        exit_code = subprocess.run(["python", file_path] + arguments).returncode
        judge_expression(exit_code == 0)
        convert_file_path = os.path.join(args.save_dir, "mg2hf/*")
        files = glob.glob(convert_file_path)
        judge_expression(len(files) >= 17)
        judge_expression(os.path.exists(os.path.join(args.save_dir, "mg2hf/model.safetensors.index.json")))
        judge_expression(os.path.exists(os.path.join(args.save_dir, "mg2hf/config.json")))
        judge_expression(os.path.exists(os.path.join(args.save_dir, "mg2hf/modeling_deepseek.py")))

        n = 12
        for i in range(1, n + 1):
            print(f"model-{str(i).rjust(5, '0')}-of-{str(n).rjust(5, '0')}.safetensors")
            a = safe_load_file(
                f"{args.save_dir}mg2hf/model-{str(i).rjust(5, '0')}-of-{str(n).rjust(5, '0')}.safetensors")
            b = safe_load_file(
                f"{args.save_dir}/model-{str(i).rjust(5, '0')}-of-{str(n).rjust(5, '0')}.safetensors")

            for key in a.keys():
                judge_expression( a[key].equal(b[key]))

    def test_combine_lora_weights_from_megatron(self):
        args = CovertCkptFromMegatronArgs()

        base_dir = Path(__file__).absolute().parent.parent.parent
        file_path = os.path.join(base_dir, "tools/checkpoint/convert_ckpt.py")
        arguments = [
            "--model-type", args.model_type,
            "--loader", args.loader,
            "--saver", args.saver,
            "--save-model-type", "megatron",
            "--load-dir", args.load_dir,
            "--target-tensor-parallel-size", "1",
            "--target-pipeline-parallel-size", "1",
            "--save-dir", args.save_dir
        ]

        subprocess.run(["python", file_path] + arguments)

        output_dir = os.path.join(args.save_dir, "iter_0000001")
        weight_content = torch.load(os.path.join(output_dir, "mp_rank_00/model_optim_rng.pt"))
        weight_common_content = weight_content['model']['language_model'] # extract commmon content

        # embedding, encoder, output_layer is three out layers.
        judge_expression(len(os.listdir(output_dir)) == int(args.target_tensor_parallel_size))
        judge_expression(weight_common_content['embedding']['word_embeddings']['weight'].size() == torch.Size([32000, 4096]))
        judge_expression(weight_common_content['encoder']['final_norm.weight'].size() == torch.Size([4096]))

        # encoder has a common final_norm and each one has folliowing six layers
        weight_common_content['encoder'].pop('final_norm.weight')
        judge_expression(weight_common_content['encoder']['layers.0.self_attention.query_key_value.weight'].size() == torch.Size([12288, 4096]))
        judge_expression(weight_common_content['encoder']['layers.0.self_attention.dense.weight'].size() == torch.Size([4096, 4096]))
        judge_expression(weight_common_content['encoder']['layers.0.mlp.dense_h_to_4h.weight'].size() == torch.Size([22016, 4096]))
        judge_expression(weight_common_content['encoder']['layers.0.mlp.dense_4h_to_h.weight'].size() == torch.Size([4096, 11008]))
        judge_expression(weight_common_content['encoder']['layers.0.input_norm.weight'].size() == torch.Size([4096]))
        judge_expression(weight_common_content['encoder']['layers.0.post_attention_norm.weight'].size() == torch.Size([4096]))

        judge_expression(weight_common_content['output_layer']['weight'].size() == torch.Size([32000, 4096]))