import torch
import time
import os
import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoTokenizer
import torch_npu
from torch_npu.contrib import transfer_to_npu
import argparse
import datetime
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


def setup_model_parallel():
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12349"
    local_rank = int(os.getenv("LOCAL_RANK", '0'))
    world_size = int(os.getenv("WORLD_SIZE", '0'))
    # torch_npu.npu.set_device(local_rank + 8)
    torch_npu.npu.set_device(local_rank)

    torch.set_num_threads(8)
    
    torch.distributed.init_process_group(
        backend='lccl',
        world_size=world_size, rank=local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="load Model weights and run.")
    parser.add_argument(
        "--load_path",
        default = "/home/data/acltransformer_testdata/weights/mistral-7B-v0.1",
        help = "Location of Model weights, which contains model folders",
    )
    args = parser.parse_args()
    #initialize parallel
    local_rank, world_size = setup_model_parallel()

    torch.npu.set_compile_mode(jit_compile=False)
    option = {}
    option["NPU_FUZZY_COMPILE_BLACKLIST"] = "Tril,SoftmaxV2,LayerNormGrad,ReduceProd"
    torch.npu.set_option(option)

    tokenizer_path = '/home/data/acltransformer_testdata/weights/deepseek-moe-16b-chat'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False, padding_side='left', trust_remote_code=True)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # pad or not
    part_model_path=args.load_path + '/part_model/' + str(local_rank) + '/'
    model = AutoModelForCausalLM.from_pretrained(part_model_path, torch_dtype=torch.float16, trust_remote_code=True,attn_implementation="eager").npu()
    
    prompt = [
                {"role": "user", "content": "Common sense questions and answers\n\nQuestion: How to learn a new language\nFactual answer"},
    		  	# [{"role": "user", "content": "Common sense questions and answers\n\nQuestion: Why do we need to learn a new language\nFactual answer:"}],
    		  	# [{"role": "user", "content": "Common sense questions and answers\n\nQuestion: What is the best way to study\nFactual answer:"}],
    		  	# [{"role": "user", "content": "Common sense questions and answers\n\nQuestion: What is the capital of France\nFactual answer:"}],
    		  	# [{"role": "user", "content": "Common sense questions and answers\n\nQuestion: Should I take notes during class\nFactual answer:"}],
    		  	# [{"role": "user", "content": "Common sense questions and answers\n\nQuestion: How to use photoshop\nFactual answer:"}],
    		  	# [{"role": "user", "content": "Common sense questions and answers\n\nQuestion: What is depression\nFactual answer:"}],
    		  	# [{"role": "user", "content": "Common sense questions and answers\n\nQuestion: Who was the first president of the United States\nFactual answer:"}],
    		  	# [{"role": "user", "content": "Common sense questions and answers\n\nQuestion: What is the name of the vice president of the United States\nFactual answer:"}],
    		  	# [{"role": "user", "content": "Common sense questions and answers\n\nQuestion: How to learn programming\nFactual answer:"}],
    		  	# [{"role": "user", "content": "Common sense questions and answers\n\nQuestion: How does artificial intelligence work\nFactual answer:"}],
    		  	# [{"role": "user", "content": "Common sense questions and answers\n\nQuestion: Why should we learn math\nFactual answer:"}],
    		  	# [{"role": "user", "content": "Common sense questions and answers\n\nQuestion: How does machine learning work\nFactual answer:"}],
    		  	# [{"role": "user", "content": "Common sense questions and answers\n\nQuestion: How to live a happy life\nFactual answer:"}],
    		  	# [{"role": "user", "content": "Common sense questions and answers\n\nQuestion: How to run a company\nFactual answer:"}],
    		  	# [{"role": "user", "content": "Common sense questions and answers\n\nQuestion: Who is the CEO of Google\nFactual answer:"}],
    		 ]

    ########################################################################################################
    # Personal input shape construction and multi batch input is currently NOT supported by the tokenizer!!!
    ########################################################################################################
    batch_list = [28]
    seq_in_list = [1024] 
    seq_out_list = [250]
    for batch in batch_list:
        if torch.distributed.get_rank() == 0:
            print("---------------warm-up---------------")
        multi_str = ""
        for i in range(1200):
            multi_str += "A "
        test_prompt = [multi_str] * batch

        inputs_warm_up = tokenizer.apply_chat_template(test_prompt, tokenize=True, add_generation_prompt=True, return_tensors="pt")

        with torch.no_grad():
            _ = model.generate(inputs_warm_up.npu(), max_new_tokens=4)
        
        torch.npu.synchronize() 
        if torch.distributed.get_rank() == 0:
            print("---------------warm-up success---------------")
        for seqlen_in in seq_in_list:
            for seqlen_out in seq_out_list:                
                # print("---------------multibatch---------------")
                multi_str = ""
                for i in range(seqlen_in - 2):
                    multi_str += "A "
                multi_prompt = [multi_str] * batch

                inputs = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, return_tensors="pt")
                
                if torch.distributed.get_rank() == 0:
                    print("---------------inputs batch num---------------")
                    print(len(inputs))
            
                start_time = time.time()
                torch.npu.synchronize()

                with torch.no_grad():
                    generate_ids = model.generate(inputs.npu(), max_new_tokens=seqlen_out)

                end_time = time.time()
                torch.npu.synchronize()
                if torch.distributed.get_rank() == 0:
                    print("-----------------Batch, seq_in, seq_out------------------------", batch, seqlen_in, seqlen_out)
                    save_path="deepseek16B_performance/"
                    old_name=save_path+"time"
                    current_date = datetime.datetime.now()
                    formatted_date = current_date.strftime("%Y-%m-%d %H:%M:%S")
                    new_name=save_path + formatted_date + "_SIN_"+ str(seqlen_in) + "_SOUT_"+str(seqlen_out) + "_BS_" + str(batch)
                    os.rename(old_name, new_name)

                # decode
                res = tokenizer.decode(generate_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

                # decode
                res = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                # output
                if torch.distributed.get_rank() == 0:
                    print("########################################################################################################################")
                    for item in res:
                            print(item)
                    print("########################################################################################################################")
