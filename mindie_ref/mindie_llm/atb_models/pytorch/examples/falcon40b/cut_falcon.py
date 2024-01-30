import torch
import time
from transformers import AutoTokenizer, FalconForCausalLM
import os
import argparse


def cut_weights(model, world_size):
    state_dict_list = [dict() for _ in range(world_size)]
    for key, tensor in model.state_dict().items():
        cut_tensor_list = []
        key_short = ".".join([key.split(".")[-2], key.split(".")[-1]])
        if key_short in ("dense_h_to_4h.weight"):
            cut_tensor_list = torch.chunk(tensor, world_size, dim=0)

        # Falcon-40B 默认没有 query_key_value.bias， 可以在 config.json 中修改 "bias": true 开启 , "query_key_value.bias"
        elif key_short in ("query_key_value.weight"):
            # Falcon 40B: query_key_value.weight tensor shape is (9216, 8192)
            num_attention_heads = 128  # 'FalconConfig' object has no attribute 'num_heads'
            head_dim = model.config.hidden_size // num_attention_heads # 8192//128=64
            config_num_kv_heads = 8
            # 9216 = 8192 + 512 + 512  // 4 world size
            # 2304 = 2048 + 128 + 128  // 64 head_dim
            # 36   = 32   + 2   + 2  
            tensor = tensor.view(config_num_kv_heads, -1, head_dim, model.config.hidden_size)
            query_layer, key_layer, value_layer = tensor.split(
                [
                    16,  # 64*128  8192
                    1,   # 64*8    512
                    1    # 64*8    512
                ],
                dim = 1
            )
            query_list = torch.chunk(query_layer, world_size, dim=0)
            key_list   = torch.chunk(key_layer,   world_size, dim=0)
            value_list = torch.chunk(value_layer, world_size, dim=0)
            for i in range(world_size):
                cut_tensor_list.append(torch.cat([query_list[i], key_list[i], value_list[i]], dim=1).view(-1, 8192))
        elif key_short in ["dense_4h_to_h.weight", "dense.weight"]:
            cut_tensor_list = torch.chunk(tensor, world_size, dim=1)
        else:
            cut_tensor_list = [tensor] * world_size

        for i in range(world_size):
            state_dict_list[i][key] = cut_tensor_list[i]
    return state_dict_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cut Model weights.")
    parser.add_argument("--input_path", default="./model/", help="Location of Model weights, which contains model folders",)
    parser.add_argument("--output_path", default='./', help="Location to write the part weights")
    parser.add_argument("--world_size", default=4, help="world_size")
    args = parser.parse_args()
    args.world_size = int(args.world_size)
    tokenizer = AutoTokenizer.from_pretrained(args.input_path, use_fast=False)  # load the tokenizer
    tokenizer.save_pretrained(args.output_path+'/tokenizer')  # save the tokenizer
    print("[~] Loading model ...")
    START_TIME = time.time()
    model = FalconForCausalLM.from_pretrained(args.input_path, torch_dtype=torch.float16)
    print(f"[+] Load model success: {int(time.time() - START_TIME)//60} min")
    state_dict_list = cut_weights(model, args.world_size)  # cut the weight

    model_config = model.config
    model_config.world_size = args.world_size
    model_config.torch_dtype = torch.float16
    create_model = FalconForCausalLM(model_config)
    print(f"[~] Saving parts ...")
    for i in range(args.world_size):
        # load the weights to the model
        create_model.load_state_dict(state_dict_list[i])
        create_model = create_model.half()
        save_path = os.path.join(args.output_path, "part_model", str(i))
        create_model.save_pretrained(save_path)  # save model
        print(f"[{i}] save to {save_path}")
    print('[+] save succcessfully')