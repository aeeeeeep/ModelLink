from megatron import training
from modellink.data.data_samplers import build_pretraining_data_loader


def apply_data_patch():
    training.build_pretraining_data_loader = build_pretraining_data_loader