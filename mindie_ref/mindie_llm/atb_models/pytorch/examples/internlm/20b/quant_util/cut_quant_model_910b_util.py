import argparse
import torch
import os
import shutil
import numpy as np
import torch_npu
from torch_npu.contrib import transfer_to_npu


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cut Model weights.")
    parser.add_argument(
        "--input_path",
        default="./",
        help="Location of Model weights, which contains model folders",
    )
    parser.add_argument(
        "--output_path",
        default='./paraller/',
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


def cut_weights(model, world_size, cut_row_keys=['q_proj', 'k_proj', 'v_proj', 'gate_proj', 'up_proj'],
                cut_col_keys=['o_proj', 'down_proj']):
    print('*****************************cut_weights begin***********************************')
    state_dict_list = [{} for _ in range(world_size)]
    for key, tensor in model.items():
        print(f"key = {key}")
        key_short = key.split('.')[-1]
        if key_short in cut_row_keys:
            cut_tensor_list = torch.chunk(tensor, world_size, dim=0)
        elif key_short in cut_col_keys:
            cut_tensor_list = torch.chunk(tensor, world_size, dim=1)
        else:
            cut_tensor_list = [tensor] * world_size
        for i in range(world_size):
            state_dict_list[i][key] = cut_tensor_list[i]
            tensor = cut_tensor_list[i]
            print(f'state_dict_list[{i}][{key}] = {tensor.shape}')
    print('*****************************cut_weights end***********************************')
    return state_dict_list


def cut_bias(bias, world_size, cut_row_keys=['q_proj', 'k_proj', 'v_proj', 'gate_proj', 'up_proj'],
             cut_col_keys=['o_proj', 'down_proj'], is_bias=False):
    state_dict_list = [{} for _ in range(world_size)]
    print('*****************************cut_bias begin***********************************')
    for key, tensor in bias.items():
        key_short = key.split('.')[-1]
        if key_short in cut_row_keys:
            cut_tensor_list = torch.chunk(tensor, world_size, dim=0)
        elif key_short in cut_col_keys:
            if is_bias:
                tensor = tensor / 2.0
            cut_tensor_list = [tensor] * world_size
        for i in range(world_size):
            state_dict_list[i][key] = cut_tensor_list[i]
            tensor = cut_tensor_list[i]
            print(f'state_dict_list[{i}][{key}] = {tensor.shape}')
    print('*****************************cut_bias end***********************************')
    return state_dict_list


def bias_correction(fp_bias, quant_weight, input_offset, deq_scale):
    correction = quant_weight.to(torch.float32).npu().sum(dim=1) * float(input_offset) * deq_scale.npu()
    bias_correction = fp_bias.npu() - correction
    return bias_correction


def bias_correction_910b(fp_bias, quant_weight, input_offset, deq_scale):
    bias_correction = fp_bias.npu() / deq_scale.npu() - quant_weight.to(torch.float32).npu().sum(dim=1) * float(
        input_offset)
    return bias_correction


if __name__ == "__main__":
    # parse args
    opts = parse_args()

    weight_path = opts.input_path
    print(f"=========quant weight path:{weight_path} ==========")
    quant_weight_dict = np.load(weight_path + "/quant_weight.npy", allow_pickle=True).item()
    deq_scale_dict = np.load(weight_path + "/deq_scale.npy", allow_pickle=True).item()
    input_offset_dict = np.load(weight_path + "/input_offset.npy", allow_pickle=True).item()
    fp_bias_dict = np.load(weight_path + "/fp_bias.npy", allow_pickle=True).item()
    quant_bias_dict = np.load(weight_path + "/quant_bias.npy", allow_pickle=True).item()

    fp_bias_corr = {}
    for i in quant_weight_dict.keys():
        fp_bias_corr[i] = bias_correction_910b(fp_bias_dict[i],
                                               quant_weight_dict[i],
                                               int(input_offset_dict[i]),
                                               deq_scale_dict[i]).cpu()
        print(f"fp_bias_corr[{i}] = {fp_bias_corr[i]}, fp_bias_dict[{i}] = {fp_bias_dict[i]}")

    print('cut quant_weight_dict')
    state_quant_weight_dict_list = cut_weights(quant_weight_dict, opts.world_size, cut_row_keys=['q_proj', 'k_proj',
                                                                                                 'v_proj', 'gate_proj',
                                                                                                 'up_proj'],
                                               cut_col_keys=['o_proj', 'down_proj'])

    print('cut fp_bias_corr_dict')
    fp_state_corr_bias_dict_list = cut_bias(fp_bias_corr, opts.world_size,
                                            cut_row_keys=['q_proj', 'k_proj', 'v_proj', 'gate_proj', 'up_proj'],
                                            cut_col_keys=['o_proj', 'down_proj'], is_bias=True)

    print('cut quant_bias_dict')
    quant_state_bias_dict_list = cut_bias(quant_bias_dict, opts.world_size,
                                          cut_row_keys=['q_proj', 'k_proj', 'v_proj', 'gate_proj', 'up_proj'],
                                          cut_col_keys=['o_proj', 'down_proj'], is_bias=True)

    print('cut deq_scale_dict')
    state_deq_scale_dict_list = cut_bias(deq_scale_dict, opts.world_size,
                                         cut_row_keys=['q_proj', 'k_proj', 'v_proj', 'gate_proj', 'up_proj'],
                                         cut_col_keys=['o_proj', 'down_proj'], is_bias=False)

    save_path = opts.output_path
    print(f"=========part quant weight path:{save_path} ==========")
    for i in range(opts.world_size):
        base_path = os.path.join(save_path, str(i))
        print(base_path)
        os.makedirs(base_path, exist_ok=True)
        np.save(os.path.join(base_path, "quant_weight.npy"), state_quant_weight_dict_list[i])
        np.save(os.path.join(base_path, "fp_bias_corr.npy"), fp_state_corr_bias_dict_list[i])
        np.save(os.path.join(base_path, "quant_bias.npy"), quant_state_bias_dict_list[i])
        np.save(os.path.join(base_path, "deq_scale.npy"), state_deq_scale_dict_list[i])
        shutil.copyfile(os.path.join(weight_path, "input_offset.npy"), os.path.join(base_path, "input_offset.npy"))
        shutil.copyfile(os.path.join(weight_path, "input_scale.npy"), os.path.join(base_path, "input_scale.npy"))

    print('save succcessfully')
    print("the location of parallel quant weight is {}".format(save_path))
