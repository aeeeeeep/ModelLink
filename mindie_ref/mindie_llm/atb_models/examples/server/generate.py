# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import torch

from loguru import logger
from .batch import Batch


def next_token_chooser(logits: torch.Tensor):
    return torch.argmax(logits, dim=-1)


def generate_token(model, tokenizer, cache_manager, batch: Batch, max_out_length, rank):
    logits = model.forward(
        input_ids=batch.batch_input_ids.npu(),
        position_ids=batch.batch_position_ids.npu(),
        is_prefill=batch.cu_seqlen_prefill is not None,
        block_tables=batch.batch_block_tables.npu(),
        kv_cache=cache_manager.kv_cache,
        slots=batch.batch_slots_tables[batch.batch_slot_indices].npu(),
        input_lengths=batch.context_length.npu(),
        max_seq_len=batch.max_s,
        lm_head_indices=None if batch.lm_head_indices is None else batch.lm_head_indices.npu()
    )
    next_token = next_token_chooser(logits)

    for i, req in enumerate(batch.req_list):
        req.out_token_list.append(int(next_token[i]))

    batch.batch_input_ids = next_token.to(torch.int64)
    batch.batch_position_ids = batch.context_length.to(torch.long)
    if batch.cu_seqlen_prefill is not None:
        batch.batch_slot_indices = batch.batch_slot_indices[batch.lm_head_indices]
        batch.cu_seqlen_prefill = None
        batch.lm_head_indices = None

    batch.batch_slot_indices += 1
    batch.context_length += 1
    batch.max_s += 1

    eos_token_id = tokenizer.eos_token_id
    return batch.filter(eos_token_id, max_out_length, cache_manager)


def generate_req(req_list, model, tokenizer, max_batch_size, max_prefill_tokens, max_out_length, cache_manager,
                        rank):
    req_num = len(req_list)
    if rank == 0:
        logger.info(f"------total req num: {req_num}, infer start--------")

    req_idx = 0
    total_req_finished = 0
    generate_batch_size = 0

    generate_batches = []

    while total_req_finished < req_num:
        do_generate = True
        if req_idx < req_num and generate_batch_size < max_batch_size:
            prefill_start = req_idx
            free_block = cache_manager.get_free_block_num()
            total_need_blocks = 0
            total_prefill_token = 0
            prefill_batch_size = 0

            while generate_batch_size + prefill_batch_size < max_batch_size:
                if req_idx >= req_num:
                    break
                cur_need_blocks = req_list[req_idx].need_blocks
                cur_context_len = req_list[req_idx].input_length
                if total_need_blocks + cur_need_blocks > free_block:
                    break
                if total_prefill_token + cur_context_len > max_prefill_tokens:
                    do_generate = False
                    break
                total_need_blocks += cur_need_blocks
                total_prefill_token += cur_context_len
                prefill_batch_size += 1
                req_idx += 1

            if prefill_batch_size > 0:
                batch = Batch(req_list[prefill_start:prefill_start + prefill_batch_size])
                cache_manager.allocate(batch)
                req_finished = generate_token(model, tokenizer, cache_manager, batch, max_out_length, rank)

                generate_batches.append(batch)

                if req_finished != (prefill_batch_size - batch.batch_num):
                    logger.error("batch filter error")
                    raise AssertionError
                generate_batch_size += batch.batch_num
                total_req_finished += req_finished

        if do_generate:
            if len(generate_batches) > 1:
                Batch.concatenate(generate_batches)

            generate_batch_size = generate_batches[0].batch_num
            if generate_batch_size > max_batch_size:
                logger.error(f"decode batch size: {generate_batch_size}, max allowed: {max_batch_size}")
                raise AssertionError

            req_finished = generate_token(model, tokenizer, cache_manager, generate_batches[0], max_out_length, rank)

            if req_finished != (generate_batch_size - generate_batches[0].batch_num):
                logger.error(f"batch filter error")
                raise AssertionError
            generate_batch_size = generate_batches[0].batch_num
            total_req_finished += req_finished


def decode_token(req_list, tokenizer):
    decode_text_list = []
    token_num_list = []
    request_id = 0
    token_num = 0
    for req in req_list:
        out_token = len(req.out_token_list)
        token_tensor = torch.tensor(req.out_token_list, dtype=torch.int64)
        decode_text = tokenizer.decode(token_tensor)
        decode_text_list.append(decode_text)
        token_num += out_token
        token_num_list.append((request_id, token_num))
        request_id += 1
    return decode_text_list, token_num_list
