# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import argparse
import os
import torch
from atb_llm.runner import ModelRunner
from atb_llm.utils.cpu_binding import NpuHbmInfo
from atb_llm.utils.log import logger, print_log
from atb_llm.models.base.model_utils import unwrap_model_state_dict

from modelslim.pytorch.weight_compression import CompressConfig, Compressor
from ..convert_utils import copy_tokenizer_files, modify_config


class SparseCompressor:
    def __init__(self, **kwargs):
        self.rank = kwargs.get('rank', '0')
        self.world_size = kwargs.get('world_size', '1')

        self.model_path = kwargs.get('model_path', None)
        self.save_directory = kwargs.get('save_directory', None)
        self.multiprocess_num = kwargs.get('multiprocess_num', 16)
        self.save_split_w8s8s_dir = kwargs.get('save_split_w8s8s_dir', None)

        self.model = ModelRunner(self.model_path, rank=self.rank, world_size=self.world_size)
        self.dtype = self.model.dtype
        self.quantize = self.model.quantize
        self.model.load_weights()

        self.device = self.model.device
        self.max_memory = NpuHbmInfo.get_hbm_capacity(self.rank, self.world_size, self.model.soc_info.need_nz)
        self.init_memory = int(
            self.max_memory * NpuHbmInfo.get_hbm_usage(self.rank, self.world_size, self.model.soc_info.need_nz))
        print_log(self.rank, logger.info, f'hbm_capacity(GB): {self.max_memory / (1024 ** 3)}, '
                                          f'init_memory(GB): {self.init_memory / (1024 ** 3)}')

        self.warm_up_memory = 0
        self.warm_up_num_blocks = 0
        self.cache_manager = None

        if self.save_split_w8s8s_dir is not None:
            self.model.save_pretrained(save_directory=f'{self.save_split_w8s8s_dir}_{self.world_size}',
                                       safe_serialization=True)

    def compress(self):
        model_dict = unwrap_model_state_dict(self.model.model.state_dict())
        quant_desc = self.model.model.generate_description()
        compress_config = CompressConfig(do_pseudo_sparse=False, sparse_ratio=1, is_debug=True,
                                         record_detail_root=self.save_directory,
                                         multiprocess_num=self.multiprocess_num)
        compressor = Compressor(compress_config, weight=model_dict, quant_model_description=quant_desc)
        compressor.run()
        part_save_directory = os.path.join(self.save_directory, f'part{self.rank}')
        compressor.export_safetensors(part_save_directory)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
                        help="model and tokenizer path",
                        default='/data/acltransformer_testdata/weights/llama2/llama-2-70b',
                        )
    parser.add_argument('--save_directory', type=str, required=True)
    parser.add_argument('--multiprocess_num', type=int, default=16)
    parser.add_argument('--save_split_w8s8s_dir', type=str, default=None)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    input_dict = {
        'rank': rank,
        'world_size': world_size,
        **vars(args)
    }

    model_path = args.model_path
    save_directory = args.save_directory

    sparse_compressor = SparseCompressor(**input_dict)

    sparse_compressor.compress()

    if rank == 0:
        modify_config(model_path, save_directory, torch.float16, 'w8a8s')
        copy_tokenizer_files(model_path, save_directory)
