import os
import sys
import argparse

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

from megatron.core.datasets.indexed_dataset import (
    IndexedDataset,
    IndexedDatasetBuilder,
    get_bin_path,
    get_idx_path,
)


def get_args():
    parser = argparse.ArgumentParser()

    group = parser.add_argument_group(title="input data")
    group.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to directory containing all document files to merge",
    )
    group.add_argument('--keys', nargs='+', default=['text_document'],
                       help='The `bin-idx` pair files with the same key in their filename will be merged.')

    group = parser.add_argument_group(title="output data")
    group.add_argument(
        "--output-prefix",
        type=str,
        required=True,
        help="Path to binary output file without suffix",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.input):
        raise ValueError(f"{args.input} is not a directory or does not exist")

    if not os.path.isdir(os.path.dirname(args.output_prefix)):
        raise ValueError(f"{os.path.dirname(args.output_prefix)} is not a directory or does not exist")

    return args


def merge_datasets(args):
    prefixes = {key: set() for key in args.keys}
    for key in prefixes:
        for basename in os.listdir(args.input):
            prefix, ext = os.path.splitext(basename)
    
            if prefix in prefixes[key] or key not in prefix:
                continue
    
            if not os.path.isfile(os.path.join(args.input, basename)):
                continue
    
            ext_pair = ".bin" if ext == ".idx" else ".idx"
            if not os.path.isfile(os.path.join(args.input, prefix) + ext_pair):
                raise FileNotFoundError(f"{ext_pair} file not provided for {os.path.join(args.input, prefix)}")
    
            prefixes[key].add(prefix)
    
    for key in prefixes:
        builder = None
        for prefix in sorted(prefixes[key]):
            if builder is None:
                dataset = IndexedDataset(os.path.join(args.input, prefix), multimodal=False)
                builder = IndexedDatasetBuilder(
                    get_bin_path(f'{args.output_prefix}_{key}'), dtype=dataset.index.dtype, multimodal=False
                )
                del dataset
    
            builder.add_index(os.path.join(args.input, prefix))
    
        builder.finalize(get_idx_path(f'{args.output_prefix}_{key}'))


def main():
    args = get_args()
    merge_datasets(args)


if __name__ == '__main__':

    main()
