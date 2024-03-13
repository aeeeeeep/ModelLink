# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import argparse

from atb_llm.utils.convert import convert_files
from atb_llm.utils.hub import weight_files


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
                        help="model and tokenizer path",
                        default='/data/acltransformer_testdata/weights/llama2/llama-2-70b',
                        )
    parser.add_argument('--from_pretrained', type=bool, default=True)
    return parser.parse_args()


def convert_bin2st(model_path):
    local_pt_files = weight_files(model_path, revision=None, extension=".bin")
    local_st_files = [
        p.parent / f"{p.stem.lstrip('pytorch_')}.safetensors"
        for p in local_pt_files
    ]
    convert_files(local_pt_files, local_st_files, discard_names=[])
    found_st_files = weight_files(model_path)


def convert_bin2st_from_pretrained(model_path):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path,
        low_cpu_mem_usage=True,
        torch_dtype="auto")
    model.save_pretrained(model_path, safe_serialization=True)


if __name__ == '__main__':
    args = parse_arguments()

    if args.from_pretrained:
        convert_bin2st_from_pretrained(args.model_path)
    else:
        convert_bin2st(args.model_path)