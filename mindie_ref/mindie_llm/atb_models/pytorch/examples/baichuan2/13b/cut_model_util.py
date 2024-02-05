import argparse
import glob
import os
import shutil

import torch
from transformers import AutoModelForCausalLM

pwd = os.path.realpath(os.path.dirname(__file__))


# cut_row_keys: dim 0  cut_col_keys: dim 1  nn.linear: x*A.T
def cut_weights(model_ins,
                world_size,
                cut_w_pack_keys=None,
                cut_row_keys=None,
                cut_col_keys=None):
    if cut_w_pack_keys is None:
        cut_w_pack_keys = ['W_pack']
    if cut_row_keys is None:
        cut_row_keys = ['gate_proj', 'up_proj']
    if cut_col_keys is None:
        cut_col_keys = ['o_proj', 'down_proj']
    _state_dict_list = [{} for _ in range(world_size)]
    for key, tensor in model_ins.state_dict().items():
        key_short = key.split('.')[-2]
        if key_short in cut_w_pack_keys:
            split_linear_size = 3  # q k v linear
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
        elif "lm_head" in key:
            cut_tensor_list = torch.chunk(tensor, world_size, dim=0)
        else:
            cut_tensor_list = [tensor] * world_size
        for i in range(world_size):
            _state_dict_list[i][key] = cut_tensor_list[i]
    return _state_dict_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cut Model weights.")
    parser.add_argument(
        "--input_path",
        default="/data/models/baichuan2/new_13b",
        help="Location of Model weights, which contains model folders",
    )
    parser.add_argument(
        "--output_path",
        default='/data/models/baichuan2/new_13b/baichuan2-13b-part-test',
        help="Location to write the part weights",
    )
    parser.add_argument(
        "--world_size",
        default=2,
        help="world_size",
        type=int
    )
    parser.add_argument(
        "--cut_W_pack_keys",
        default=['W_pack'],
        help="cut_W_pack_keys",
    )
    parser.add_argument(
        "--cut_row_keys",
        default=['gate_proj', 'up_proj'],
        help="cut_row_keys",
    )
    parser.add_argument(
        "--cut_col_keys",
        default=['o_proj', 'down_proj'],
        help="cut_col_keys",
    )

    args = parser.parse_args()
    args.world_size = int(args.world_size)

    model_path = os.path.realpath(args.input_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).half()

    # step 4: cut weight
    state_dict_list = cut_weights(model, args.world_size, args.cut_W_pack_keys, args.cut_row_keys, args.cut_col_keys)

    # step 5: create new model config, add the world size parameter,
    # the model size will be cut according to the world size in the model file
    model_config = model.config
    model_config.world_size = args.world_size

    # step 6: create new model according to the new model config
    creat_model = AutoModelForCausalLM.from_config(model_config, trust_remote_code=True)
    for i in range(args.world_size):
        # step 7: load weights to each rank model
        creat_model.load_state_dict(state_dict_list[i])
        # step 8: save each rank model
        target_dir = os.path.realpath(os.path.join(args.output_path, str(i)))
        os.makedirs(target_dir, exist_ok=True)
        creat_model.save_pretrained(target_dir)
        creat_model.config.auto_map["AutoModelForCausalLM"] = 'modeling_baichuan_ascend.BaichuanForCausalLM'
        if hasattr(creat_model.config, "_name_or_path"):
            creat_model.config._name_or_path = target_dir
        creat_model.config.save_pretrained(target_dir)
        for source_file in glob.glob(os.path.join(model_path, "*.py")):
            shutil.copy(os.path.join(model_path, source_file), target_dir)
        for source_file in glob.glob(os.path.join(pwd, "modeling*.py")):
            shutil.copy(os.path.join(pwd, source_file), target_dir)
    print('Tensor parallelism weights have been successfully saved.')
