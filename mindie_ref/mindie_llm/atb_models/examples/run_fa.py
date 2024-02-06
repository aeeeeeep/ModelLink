# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import argparse
import os
import time
import torch

from loguru import logger
from atb_llm.runner import ModelRunner


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
                        help="model and tokenizer path",
                        default='/data/acltransformer_testdata/weights/llama2/llama-2-70b',
                        )
    parser.add_argument(
        '--input_text',
        type=str,
        nargs='+',
        default=["What's deep learning?"])
    parser.add_argument(
        '--input_file',
        type=str,
        help='CSV or Numpy file containing tokenized input. Alternative to text input.',
        default=None)
    parser.add_argument('--max_input_length', type=int, default=512)
    parser.add_argument('--max_output_length', type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument('--is_flash_causal_lm', action='store_true')
    parser.add_argument('--is_bf16', action='store_true')

    parser.add_argument('--num_beams',
                        type=int,
                        help="Use beam search if num_beams >1",
                        default=1)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=1)
    parser.add_argument('--top_p', type=float, default=0.0)
    parser.add_argument('--length_penalty', type=float, default=1.0)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--presence_penalty', type=float, default=0.0)
    parser.add_argument('--frequency_penalty', type=float, default=0.0)
    parser.add_argument('--use_refactor', action='store_true')

    return parser.parse_args()


def warm_up(model, batch_size, max_input_length, max_output_length, rank):
    if rank == 0:
        logger.info("---------------begin warm-up---------------")
    dummy_input_ids_full = torch.randint(0, 32000, [batch_size, max_input_length], dtype=torch.long).npu()
    model.generate(inputs=dummy_input_ids_full, do_sample=False, max_new_tokens=10)
    if rank == 0:
        logger.info("---------------end warm-up---------------")


def infer(model, tokenizer, input_text, batch_size, max_input_length, max_output_length, rank):
    if rank == 0:
        logger.info("---------------begin inference---------------")
    if isinstance(input_text, str):
        input_text = [input_text] * batch_size

    inputs = tokenizer(input_text, return_tensors="pt", padding='max_length',
                       max_length=max_input_length,
                       truncation=True)

    prefill_start_time = time.time()
    with torch.no_grad():
        model.generate(inputs=inputs.input_ids.npu(),
                       attention_mask=inputs.attention_mask.npu(),
                       max_new_tokens=1)
    prefill_end_time = time.time()

    decode_start_time = time.time()
    with torch.no_grad():
        generate_ids = model.generate(inputs=inputs.input_ids.npu(),
                                      attention_mask=inputs.attention_mask.npu(),
                                      max_new_tokens=max_output_length
                                      )
    decode_end_time = time.time()

    generate_text = tokenizer.batch_decode(generate_ids[:, max_input_length:], skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)
    if rank == 0:
        input_tokens_num = len(inputs.input_ids[0])
        generate_tokens_num = len(generate_ids[0]) - len(inputs.input_ids[0])
        logger.info(f'Question: {input_text[0]}')
        logger.info(f'Answer: {generate_text[0][:-generate_tokens_num]}')
        logger.info(f'Generate token num: {generate_tokens_num}')

        logger.info("---------------end inference---------------")

        prefill_time = (prefill_end_time - prefill_start_time)
        e2e_time = (decode_end_time - decode_start_time)
        try:
            decode_average_time = (e2e_time - prefill_time) / (max_output_length - 1)
        except ZeroDivisionError as e:
            raise ZeroDivisionError from e
        logger.info(
            f"Prefill time: {prefill_time}ms, Decode average time: {decode_average_time}ms, E2E time: {e2e_time}ms")


if __name__ == '__main__':
    args = parse_arguments()
    model_path = args.model_path
    input_text = args.input_text
    max_input_length = args.max_input_length
    max_output_length = args.max_output_length
    batch_size = args.batch_size
    is_flash_causal_lm = args.is_flash_causal_lm
    use_refactor = args.use_refactor
    is_bf16 = args.is_bf16

    if is_bf16:
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    model = ModelRunner(model_path, rank=rank, world_size=world_size, quantize=None, dtype=dtype,
                        is_flash_causal_lm=is_flash_causal_lm, use_refactor=use_refactor)

    tokenizer = model.tokenizer
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    model.load_weights()

    warm_up(model, batch_size, max_input_length, max_output_length, rank)

    infer(model, tokenizer, input_text, batch_size, max_input_length, max_output_length, rank)

    if world_size > 1:
        torch.distributed.destroy_process_group()
