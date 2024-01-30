import argparse
import os
import sys
import time

import torch
import torch_npu
from transformers import AutoTokenizer
from configuration_gpt_neox import GPTNeoXConfig
from patches.models.modeling_gpt_neox_rope_fusion_model import GPTNeoXForCausalLM

sys.path.append(os.path.join(os.path.realpath(os.path.dirname(__file__)), "../"))
from common.launcher import BaseLauncher


def get_args():
    parser = argparse.ArgumentParser(
        description="load Model weights and run.")
    parser.add_argument(
        "--load_path",
        default="/data/acltransformer_testdata/weights/gptneox20b",
        help="Location of Model weights, which contains model folders",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="Run devices of Model",
    )

    return parser.parse_args()


class GPTNeox20B(BaseLauncher):
    def init_model(self):
        start_time = time.time()
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, padding_side="left")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        config = GPTNeoXConfig.from_pretrained(self.model_path)
        config.is_decoder = True
        print(f"Done load model config in cpu cost: {time.time() - start_time}s, ", config)

        model = GPTNeoXForCausalLM.from_pretrained(self.model_path, config=config)
        model.resize_token_embeddings(len(tokenizer))
        print(f"Done load model in cpu, cost: {time.time() - start_time}s")
        model.gradient_checkpointing_disable()
        model.half().npu()
        model.eval()

        return model, tokenizer

    def infer(self, query, is_warm_up=False):
        with torch.no_grad():
            inputs = self.tokenizer(query, return_tensors="pt").input_ids.npu()
            max_new_tokens = 10 if is_warm_up else 30

            start_time = time.time()
            output_ids = self.model.generate(inputs, do_sample=False, max_new_tokens=max_new_tokens)
            end_time = time.time()
            time_cost = end_time - start_time
            answers = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            print("output ids is", output_ids)
            print("answer is: ", answers)
            print(f"infer cost: {time_cost}s")

        return answers

    def infer_batch(self, query, is_warm_up=False):
        with torch.no_grad():
            inputs = self.tokenizer(query, padding="max_length", max_length=1024, return_tensors="pt")
            max_new_tokens = 10 if is_warm_up else 30

            start_time = time.time()
            output_ids = self.model.generate(inputs.input_ids.npu(), attention_mask=inputs.attention_mask.npu(),
                                             do_sample=False, max_new_tokens=max_new_tokens)
            end_time = time.time()
            time_cost = end_time - start_time
            answers = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            print("output ids is", output_ids)
            for answer in answers:
                print("answer is: ", answer, end="\n\n")
            print(f"infer cost: {time_cost}s")

        return answers


if __name__ == '__main__':
    option = {"NPU_FUZZY_COMPILE_BLACKLIST": "Tril,SoftmaxV2,LayerNormGrad,ReduceProd"}
    args = get_args()
    print(f"[WARNING] USE npu:{args.device}")
    model_launcher = GPTNeox20B(args.device, args.load_path, option)
    prompts = ["hello, please introduce yourself."]

    print("-------warm up------")
    model_launcher.infer(prompts, is_warm_up=True)

    print("-------inference------")
    model_launcher.infer(prompts)

    print("-------inference batch------")
    prompts = [
        "hello, please introduce yourself.",
        "Common sense questions and answers\n\nQuestion: How to learn a new language\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: Why do we need to learn a new language\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: What is the best way to study\nFactual answer:"
    ]
    model_launcher.infer_batch(prompts)
