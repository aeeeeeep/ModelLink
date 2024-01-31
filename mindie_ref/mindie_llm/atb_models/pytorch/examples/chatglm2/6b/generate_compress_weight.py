import os
import argparse
import numpy as np
from modelslim.pytorch.weight_compression import CompressConfig, Compressor


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, mode=0o750)
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="Cut compress weights.")
    parser.add_argument("--weight_path",
                        help="path of quant weight")
    parser.add_argument("--save_path",
                        help="save path")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for i in range(2):
        weight_path = os.path.join(args.weight_path, f"quant_weight{str(i)}.npy")
        save_path = os.path.join(args.save_path, f"compress{str(i)}")
        index_root = make_dir(os.path.join(save_path, 'index'))
        weight_root = make_dir(os.path.join(save_path, 'weight'))
        info_root = make_dir(os.path.join(save_path, 'info'))

        compress_config = CompressConfig(do_pseudo_sparse=False, sparse_ratio=1, is_debug=True, record_detail_root=save_path)
        compressor = Compressor(compress_config, weight_path)
        compress_weight, compress_index, compress_info = compressor.run()

        compressor.export(compress_weight, weight_root)
        compressor.export(compress_index, index_root)
        compressor.export(compress_info, info_root, dtype=np.int64)