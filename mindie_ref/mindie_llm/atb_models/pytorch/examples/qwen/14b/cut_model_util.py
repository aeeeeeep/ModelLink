import argparse
import os
import shutil

import torch
from transformers import AutoModelForCausalLM
import inspect


def parse_keys(long_key):
    list_keys = long_key.split(".")
    if len(list_keys) == 3:  # transformer.wte.weight
        return ".".join(list_keys[1:])
    elif len(list_keys) == 2:  # lm_head.weight
        return ".".join(list_keys)
    else:
        return ".".join(list_keys[3:])


"""cut_c_attn_keys_=['attn.c_attn.weight','attn.c_attn.bias']
cut_mlp_keys_=['mlp.w1.weight','mlp.w2.weight']
cut_c_attn_mlp_keys_=['attn.c_proj.weight','mlp.c_proj.weight']"""


# cut_row_keys: dim 0  cut_col_keys: dim 1  nn.linear: x*A.T
def cut_weights(model, world_size, cut_W_pack_keys, cut_row_keys, cut_col_keys):
    state_dict_list = [{} for _ in range(world_size)]
    for key, tensor in model.state_dict().items():
        key_short = parse_keys(key)
        if key_short in cut_W_pack_keys:
            split_linear_size = 3  # q k v linear, 
            full_q_weights, full_k_weights, full_v_weights = torch.chunk(tensor, split_linear_size, dim=0)
            cut_q_weights = torch.chunk(full_q_weights, world_size, dim=0)
            cut_k_weights = torch.chunk(full_k_weights, world_size, dim=0)
            cut_v_weights = torch.chunk(full_v_weights, world_size, dim=0)
            cut_tensor_list = []
            for i in range(world_size):
                cut_tensor_list.append(torch.concat((cut_q_weights[i], cut_k_weights[i], cut_v_weights[i]), dim=0))
        elif key_short in cut_row_keys:
            cut_tensor_list = torch.chunk(tensor, world_size, dim=0)
        elif key_short in cut_col_keys:
            cut_tensor_list = torch.chunk(tensor, world_size, dim=1)
        elif "lm_head.weight" in key:  # cut lm_head
            cut_tensor_list = torch.chunk(tensor, world_size, dim=1)
        else:
            cut_tensor_list = [tensor] * world_size
        for i in range(world_size):
            state_dict_list[i][key] = cut_tensor_list[i]

    return state_dict_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cut Model weights.")
    parser.add_argument(
        "--input_path",
        default="/pytorch/examples/qwen/14b/model",
        help="Location of Model weights, which contains model folders",
    )
    parser.add_argument(
        "--output_path",
        default='/pytorch/examples/qwen/14b/model/part_model',
        help="Location to write the part weights",
    )
    parser.add_argument(
        "--rank_size",
        default=2,
        help="rank_size",
        type=int
    )
    parser.add_argument(
        "--cut_c_attn_keys",
        default=['attn.c_attn.weight', "attn.c_attn.bias"],
        help="cut_c_attn_keys",
    )
    parser.add_argument(
        "--cut_mlp_keys",
        default=['mlp.w2.weight', 'mlp.w1.weight'],
        help="cut_row_keys",
    )
    parser.add_argument(
        "--cut_c_attn_mlp_keys",
        default=['attn.c_proj.weight', 'mlp.c_proj.weight'],
        help="cut_col_keys",
    )

    args = parser.parse_args()
    args.rank_size = int(args.rank_size)
    model_path = args.input_path
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).half()  # 都加载模型和权重
    
    print(f"load model from {os.path.basename(inspect.getmodule(model).__file__)} successfully!")
    print(f"load model from {os.path.realpath(inspect.getmodule(model).__file__)} successfully!")
    # step 4: cut weight
    state_dict_list = cut_weights(model, args.rank_size, args.cut_c_attn_keys, args.cut_mlp_keys,
                                  args.cut_c_attn_mlp_keys)

    # step 5: create new model config, add the world size parameter, the model size will be cut according to the world size in the model file
    model_config = model.config
    model_config.rank_size = args.rank_size

    # step 6: create new model according to the new model config
    creat_model = AutoModelForCausalLM.from_config(model_config, trust_remote_code=True)  # 根据模型参数加载模型
    
    print(f"load model from {os.path.basename(inspect.getmodule(creat_model).__file__)} successfully!")
    print(f"load model from {os.path.realpath(inspect.getmodule(creat_model).__file__)} successfully!")
    
    for i in range(args.rank_size):
        # step 7: load weights to each rank model
        creat_model.load_state_dict(state_dict_list[i])  # the shape model的形状是：
        # step 8: save each rank model
        target_dir = os.path.join(args.output_path, str(i))
        os.makedirs(target_dir, exist_ok=True)
        creat_model.save_pretrained(target_dir)
        creat_model.config.auto_map["AutoModelForCausalLM"] = "modeling_qwen_ascend.QWenLMHeadModel"
        creat_model.config.save_pretrained(target_dir)
        for source_file in ["configuration_qwen.py", "qwen_generation_utils.py", "cpp_kernels.py", "qwen.tiktoken", "modeling_qwen_ascend.py"]:
            shutil.copy(os.path.join(model_path, source_file), target_dir)

    print('Tensor parallelism weights have been successfully saved.')
