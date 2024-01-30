import os
import sys
import time

import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch_npu
from torch_npu.contrib import transfer_to_npu

pwd = os.path.realpath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, "../../.."))
from launcher import BaseLauncher
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant with a deep knowledge of code and software design. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
"""

def setup_model_parallel():
    torch.distributed.init_process_group("hccl")
    local_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    if local_rank == 0:
        torch_npu.npu.set_device(6)
    elif local_rank == 1:
        torch_npu.npu.set_device(7)
    torch.manual_seed(1)
    return local_rank, world_size


class CodeLlamaParallelLauncher(BaseLauncher):
    def init_model(self, local_rank, world_size, args):
        """
        模型初始化
        :return:
        """
        pwd = os.path.realpath(os.path.dirname(__file__))
        model_path = os.path.join(pwd, "../..", "model")
        tokenizer_path = args.load_path + '/tokenizer'
        model_path = args.load_path + '/part_model/' + str(local_rank) + '/'
        print(f"model_path = {model_path}")
        tokenizer = AutoTokenizer.from_pretrained("/home/jjfa/hyx/ascend-speed-inference/models/codellama/34b/CodeLlama-34b-Instruct-hf")
        

        print("tokenizer")
        model = AutoModelForCausalLM.from_pretrained(model_path).half().npu()
        model= model.eval()
        return model, tokenizer
    
    def get_prompt(self, message: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
        texts = [f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n']
        # The first user input is _not_ stripped
        do_strip = False
        message = message.strip() if do_strip else message
        texts.append(f'{message} [/INST]')
        return ''.join(texts)

    def infer(self, query):
        """
        推理代码
        :param query:
        :return:
        """
        
        system = DEFAULT_SYSTEM_PROMPT
        prompts = self.get_prompt(query, system)
        gen_kwargs = dict(
            max_new_tokens=512,
            top_p=0.9,
            top_k=50,
            temperature=0.1,
            do_sample=False,
            num_beams=1,
        )
        inputs = self.tokenizer(prompts, return_tensors="pt", add_special_tokens=False)
        input_id = inputs["input_ids"]
        print(f"input={input_id}")
        generation_kwargs = dict(
            inputs=inputs["input_ids"].npu(),
            attention_mask=inputs['attention_mask'].npu(),
            eos_token_id=self.tokenizer.eos_token_id,
            # repetition_penalty=1.1,
            **gen_kwargs
        )
        print(f"问:{query}")
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **generation_kwargs
            )
        print(f"outputs generate end, out={outputs[0]}")
        print(f"output.shape = {outputs[0].shape}")
        output = self.tokenizer.decode(outputs[0].cpu().numpy().tolist())
        print(f'CodeLlama34b-答:\n')
        print(output)
        end_time = time.time()
        time_cost = end_time - start_time
        print(f"cost {time_cost}s")
        return output
        
    def infer_without_prompt(self, query):
        """
        推理代码
        :param query:
        :return:
        """
        
        system = DEFAULT_SYSTEM_PROMPT
        prompts = query
        gen_kwargs = dict(
            max_new_tokens=512,
            bos_token_id=1,
            eos_token_id=2,
            top_p=0.9,
            top_k=50,
            temperature=0.1,
            do_sample=True,
            num_beams=1,
        )
        inputs = self.tokenizer(prompts, return_tensors="pt", add_special_tokens=False)
        input_id = inputs["input_ids"]
        print(f"input={input_id}")
        generation_kwargs = dict(
            inputs=inputs["input_ids"].npu(),
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.1,
            **gen_kwargs
        )
        print(f"问:{query}")
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **generation_kwargs
            )
        print(f"outputs generate end, out={outputs[0]}")
        print(f"output.shape = {outputs[0].shape}")

        output = self.tokenizer.decode(outputs[0].cpu().numpy().tolist())
        print(f'CodeLlama34b-答:\n')
        print(output)
        end_time = time.time()
        time_cost = end_time - start_time
        print(f"cost {time_cost}s")
        return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="load Model weights and run.")
    parser.add_argument(
        "--load_path",
        default="/data/models/llama-13b-part_model_2",
        help="Location of Model weights, which contains model folders",
    )
    args = parser.parse_args()
    # initialize parallel
    local_rank, world_size = setup_model_parallel()
    print(f"local_rank = {local_rank}, world_size = {world_size}")

    codeLlama = CodeLlamaParallelLauncher(device_ids=str(local_rank), world_size=str(world_size), args=args, using_acl_transformers=True)
    torch.distributed.barrier()
    print("---------------warm-up---------------")
    codeLlama.infer('hello')

    print("---------------inference---------------")
    codeLlama.infer('Please use Python to implement a binary search algorithm.')
    
