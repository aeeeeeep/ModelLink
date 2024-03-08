import torch
import torch.distributed as dist

from SwissArmyTransformer import mpu, get_tokenizer


def print_rank_0(*args, **kwargs):
    if torch.distributed.get_rank() == 0:
        print(*args, **kwargs)


def build_data_loader(dataset, micro_batch_size, num_workers, drop_last, collate_fn=None):
    # Sampler.
    world_size = mpu.get_data_parallel_world_size()
    rank = mpu.get_data_parallel_rank()
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False
    )

    # Data loader. Note that batch size is the per GPU batch size.
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=micro_batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return data_loader


def gather_result(prediction, total_length, micro_batch_size):
    """
    @param prediction: Local predictions with order defined by distributed sampler
    @param total_length: Total sample num
    @return: [sample_0, sample_1, ..., sample_{total_length-1}]
    """
    res = []
    for p in prediction:
        res.extend(p)
    return res


def get_tokenized_input(item, key):
    if key in item:
        return item[key]
    tokenizer = get_tokenizer()
    pretokenized_key = key + "_pretokenized"
    if pretokenized_key not in item:
        raise(f'{pretokenized_key}not in item')
    if isinstance(item[pretokenized_key], list):
        result = []
        for raw in item[pretokenized_key]:
            result.append(tokenizer.tokenize(raw))
        return result
    else:
        return tokenizer.tokenize(item[pretokenized_key])
