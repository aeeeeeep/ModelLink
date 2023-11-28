import torch
import torch_npu
import ascendspeed_te_ops


def npu_scatter_list_(var, indice, updates, mask, reduce="updates", axis=-2):
    ascendspeed_te_ops.npu_scatter_list_(var, indice, updates, mask, reduce, axis)

    return var


def npu_scatter_list(var, indice, updates, mask, reduce="updates", axis=-2):
    output = ascendspeed_te_ops.npu_scatter_list(var, indice, updates, mask, reduce, axis)
    
    return output