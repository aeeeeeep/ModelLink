import time
import os
import argparse
import datetime
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
from transformers import AutoTokenizer, AutoModelForCausalLM


def setup_model_parallel():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "123"
    local_rank = int(os.getenv("LOCAL_RANK", '0'))
    rank = int(os.getenv("RANK", '0'))
    world_size = int(os.getenv("WORLD_SIZE", '0'))
    
    torch_npu.npu.set_device(local_rank)
    if rank == 0:
        rank = local_rank

    torch.set_num_threads(8)
    torch.distributed.init_process_group(
        backend='hccl',
        world_size=world_size, rank=rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


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

    torch.npu.set_compile_mode(jit_compile=False)
    option = {}
    option["NPU_FUZZY_COMPILE_BLACKLIST"] = "Tril,SoftmaxV2,LayerNormGrad,ReduceProd"
    torch.npu.set_option(option)

    tokenizer_path = args.load_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    part_model_path = args.load_path
 
    model = AutoModelForCausalLM.from_pretrained(part_model_path, torch_dtype=torch.float16, trust_remote_code=True).npu()

    prompt = [
        "This is a great test"
    ]

    batch_list = [1]
    seq_in_list = [1]
    seq_out_list = [200]
    for batch in batch_list:
        multi_str = ""
        for _ in range(1200):
            multi_str += "A "
        test_prompt = prompt * batch

        inputs_warm_up = tokenizer(test_prompt, return_tensors="pt")

        with torch.no_grad():
            _ = model.generate(inputs_warm_up.input_ids.npu(), attention_mask=inputs_warm_up.attention_mask.npu(),
                               max_new_tokens=20)
        torch.npu.synchronize()
        for seqlen_in in seq_in_list:
            for seqlen_out in seq_out_list:
                multi_str = ""
                for _ in range(seqlen_in - 2):
                    multi_str += "A "
                multi_prompt = prompt * batch
                inputs = tokenizer(multi_prompt, padding=True, return_tensors="pt")

                start_time = time.time()
                torch.npu.synchronize()

                with torch.no_grad():
                    generate_ids = model.generate(inputs.input_ids.npu(), attention_mask=inputs.attention_mask.npu(),
                                                  max_new_tokens=seqlen_out)

                end_time = time.time()
                torch.npu.synchronize()
                # decode
                res = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                # output
                for item in res:
                    if torch.distributed.get_rank() == 0:
                        print(item)

    # time analysis
    if torch.distributed.get_rank() == 0:
        new_tokens = len(generate_ids[0]) - len(inputs.input_ids[0])
        print("---------------Time analysis---------------")
        print("--------------------------------------------------------------------------------------------------")
        print(
            f"Output tokens number: {len(generate_ids[0])},\nInput tokens number: {len(inputs.input_ids[0])},\ntotal new tokens generated: {new_tokens}")
        print(
            f"Output generated in {(end_time - start_time):.2f} s ({new_tokens / (end_time - start_time + 0.001):.2f} tokens/s, {new_tokens} tokens)")
