import argparse
import os
import shutil

import numpy as np
import torch
import torch_npu


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cut Model weights.")
    parser.add_argument(
        "--input_path",
        default="/home/ctl/models/7b_quant",
        help="Location of Model weights, which contains model folders",
    )
    parser.add_argument(
        "--output_path",
        default='/home/ctl/models/7b_quant_cut',
        help="Location to write the part weights",
    )
    parser.add_argument(
        "--world_size",
        default=2,
        help="world_size",
    )
    args = parser.parse_args()
    args.world_size = int(args.world_size)
    return args


def cut_weights(model, world_size, cut_W_pack_keys=['W_pack'], cut_row_keys=['gate_proj', 'up_proj'],
                cut_col_keys=['o_proj', 'down_proj']):
    state_dict_list = [{} for _ in range(world_size)]
    for key, tensor in model.items():
        key_short = key.split('.')[-1]
        print(key_short)
        if key_short in cut_W_pack_keys:
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
        else:
            cut_tensor_list = [tensor] * world_size
        for i in range(world_size):
            state_dict_list[i][key] = cut_tensor_list[i]
    return state_dict_list


def cut_bias(bias, world_size, cut_W_pack_keys=['W_pack'], cut_row_keys=['gate_proj', 'up_proj'],
             cut_col_keys=['o_proj', 'down_proj'], is_bias=False):
    state_dict_list = [{} for _ in range(world_size)]
    for key, tensor in bias.items():
        key_short = key.split('.')[-1]
        print(key_short)
        if key_short in cut_W_pack_keys:
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
            if is_bias:
                tensor = tensor / world_size
            cut_tensor_list = [tensor] * world_size
        for i in range(world_size):
            state_dict_list[i][key] = cut_tensor_list[i]
    return state_dict_list


def bias_correction(fp_bias, quant_weight, input_offset, deq_scale):
    bias_correction = fp_bias.npu()/deq_scale.npu() - quant_weight.to(torch.float32).npu().sum(dim=1) * float(input_offset)

    return bias_correction

def process_deq_scale(deq_scale_dict):
    new_deq_scale_dict = {}
    for key, deq_scale in deq_scale_dict.items():
        deq_scale = deq_scale.numpy()
        new_deq_scale = np.frombuffer(deq_scale.tobytes(), dtype=np.int32)
        new_deq_scale_dict.setdefault(key, torch.tensor(new_deq_scale.astype(np.int64)))
    return new_deq_scale_dict


if __name__ == "__main__":
    # parse args
    opts = parse_args()

    weight_path = opts.input_path
    print(f"=========quant weight path:{weight_path} ==========")
    quant_weight_dict = np.load(weight_path + "/quant_weight.npy", allow_pickle=True).item()
    deq_scale_dict = np.load(weight_path + "/deq_scale.npy", allow_pickle=True).item()
    fp_bias_dict = np.load(weight_path + "/fp_bias.npy", allow_pickle=True).item()
    input_offset_dict = np.load(weight_path + "/input_offset.npy", allow_pickle=True).item()

    print(f"========= Quant Weight BiasCorrection ==========")
    bias_dict = {}
    for i in fp_bias_dict.keys():
        bias_dict[i] = bias_correction(fp_bias_dict[i],
                                       quant_weight_dict[i],
                                       int(input_offset_dict[i]),
                                       deq_scale_dict[i]).cpu()
    print(f"========= Quant Weight DeqScaleCorrection ==========")
    new_deq_scale_dict = process_deq_scale(deq_scale_dict)

    print(f"========= Quant Weight Cut Start ==========")
    state_quant_weight_dict_list = cut_weights(quant_weight_dict, 2, cut_W_pack_keys=['W_pack'],
                                               cut_row_keys=['gate_proj', 'up_proj'],
                                               cut_col_keys=['o_proj', 'down_proj'])
    print(f"========= Quant Bias Cut Start ==========")
    state_bias_dict_list = cut_bias(bias_dict, 2, cut_W_pack_keys=['W_pack'], cut_row_keys=['gate_proj', 'up_proj'],
                                    cut_col_keys=['o_proj', 'down_proj'], is_bias=True)
    print(f"========= Quant DeqScale Cut Start ==========")
    state_deq_scale_dict_list = cut_bias(new_deq_scale_dict, 2, cut_W_pack_keys=['W_pack'],
                                         cut_row_keys=['gate_proj', 'up_proj'],
                                         cut_col_keys=['o_proj', 'down_proj'], is_bias=False)

    save_path = opts.output_path
    print(f"=========part quant weight path:{save_path} ==========")
    for i in range(opts.world_size):
        base_path = os.path.join(save_path, str(i))
        print(base_path)
        os.makedirs(base_path, exist_ok=True)
        np.save(os.path.join(base_path, "quant_weight.npy"), state_quant_weight_dict_list[i])
        np.save(os.path.join(base_path, "bias.npy"), state_bias_dict_list[i])
        np.save(os.path.join(base_path, "deq_scale.npy"), state_deq_scale_dict_list[i])
        shutil.copyfile(os.path.join(weight_path, "input_offset.npy"), os.path.join(base_path, "input_offset.npy"))
        shutil.copyfile(os.path.join(weight_path, "input_scale.npy"), os.path.join(base_path, "input_scale.npy"))

    print('Tensor parallelism weights have been successfully saved.')
    print("the location of parallel quant weight is {}".format(save_path))
