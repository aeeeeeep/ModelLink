# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import argparse
import os
import math
import torch

from loguru import logger

from atb_llm.runner import ModelRunner
from examples.server.cache import CacheConfig, ModelConfig, CacheManager
from examples.server.batch import Batch
from examples.server.request import Request, request_from_token_file, request_from_text
from examples.server.generate import generate_token, decode_token, generate_req


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
    parser.add_argument('--max_input_length', type=int, default=1024)
    parser.add_argument('--max_output_length', type=int, default=20)
    parser.add_argument('--max_prefill_tokens', type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument('--is_flash_model', action='store_false')
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


def warm_up(model, max_prefill_tokens, cache_manager, rank):
    block_size = cache_manager.block_size

    input_ids = torch.ones(max_prefill_tokens, dtype=torch.int64)
    position_ids = torch.arange(max_prefill_tokens, dtype=torch.int32)
    cu_seqlen_prefill = torch.tensor([1])
    block_num = math.ceil(max_prefill_tokens / block_size)
    block_tables_tensor = torch.arange(block_num, dtype=torch.int32).view(1, -1)
    slots = torch.arange(max_prefill_tokens, dtype=torch.int32)
    input_lengths_tensor = torch.tensor([max_prefill_tokens], dtype=torch.int64)
    prefill_head_indices = torch.tensor([max_prefill_tokens - 1], dtype=torch.int64)
    if rank == 0:
        logger.info("---------------begin warm-up---------------")
    logits = model.forward(
        input_ids=input_ids.npu(),
        position_ids=position_ids.npu(),
        is_prefill=cu_seqlen_prefill.npu() is not None,
        block_tables=block_tables_tensor.npu(),
        kv_cache=cache_manager.kv_cache,
        slots=slots.npu(),
        input_lengths=input_lengths_tensor.npu(),
        max_seq_len=max_prefill_tokens,
        lm_head_indices=prefill_head_indices.npu()
    )
    if rank == 0:
        logger.info("---------------end warm-up---------------")


def infer(model, tokenizer, input_text, batch_size, max_prefill_tokens, max_output_length, block_size, cache_manager,
          rank):
    if rank == 0:
        logger.info("---------------begin inference---------------")
    input_text = input_text[0]
    req_list = [request_from_text(input_text, tokenizer, max_output_length, block_size, req_idx=i) \
                for i in range(batch_size)]

    generate_req(req_list, model, tokenizer, batch_size, max_prefill_tokens, max_output_length, cache_manager,
                 rank)
    generate_text_list, token_num_list = decode_token(req_list, tokenizer)
    if rank == 0:
        logger.info(f'Question: {input_text}')
        for i, generate_text in enumerate(generate_text_list):
            logger.info(f'Answer: {generate_text}')
            logger.info(f'Generate token num: {token_num_list[i]}')

        logger.info("---------------end inference---------------")


if __name__ == '__main__':
    args = parse_arguments()
    model_path = args.model_path
    input_text = args.input_text
    max_input_length = args.max_input_length
    max_prefill_tokens = args.max_prefill_tokens
    max_output_length = args.max_output_length
    is_flash_model = args.is_flash_model
    batch_size = args.batch_size
    use_refactor = args.use_refactor
    is_bf16 = args.is_bf16

    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    if is_bf16:
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    model = ModelRunner(
        model_path, rank=rank, world_size=world_size, dtype=dtype,
        quantize=None, use_refactor=use_refactor
    )
    tokenizer = model.tokenizer

    model.load_weights()

    cache_config = CacheConfig()
    model_config = ModelConfig(model.num_heads,
                               model.num_kv_heads,
                               model.head_size,
                               model.num_layers,
                               model.device,
                               model.dtype,
                               model.soc_info)

    cache_manager = CacheManager(cache_config, model_config)
    warm_up(model, max_prefill_tokens, cache_manager, rank)

    infer(model, tokenizer, input_text, batch_size, max_prefill_tokens, max_output_length, cache_config.block_size,
          cache_manager, rank)

    if world_size > 1:
        torch.distributed.destroy_process_group()
