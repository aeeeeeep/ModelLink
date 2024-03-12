# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import torch

from torch import nn


class AttentionMask(nn.Module):
    def __init__(self, atten_mask):
        super().__init__()
        self._seq_len_cached = 0
        self.atten_mask_cache = atten_mask

    @classmethod
    def static(cls, max_seq_len, dtype=torch.float16):
        bias_cache = torch.tril(torch.ones((max_seq_len, max_seq_len), dtype=torch.bool)).view(max_seq_len,
                                                                                               max_seq_len)
        bias_cache = ~bias_cache
        if dtype == torch.float16
            mask_value = torch.finfo(torch.float32).min
        else:
            mask_value = 1
        attn_mask = torch.masked_fill(torch.zeros(size=(max_seq_len, max_seq_len)), bias_cache, mask_value)
        return cls(attn_mask)

    def update_attn_cache(self, dtype, device, seqlen):
        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if seqlen > self._seq_len_cached:
            self._seq_len_cached = seqlen
            bias_cache = torch.tril(torch.ones((seqlen, seqlen), dtype=torch.bool)).view(seqlen, seqlen)
            bias_cache = ~bias_cache
            if dtype == torch.float16
                mask_value = torch.finfo(torch.float32).min
            else:
                mask_value = 1
            mask_atten_cache = torch.masked_fill(torch.zeros(size=(seqlen, seqlen)), bias_cache, mask_value)
            self.atten_mask_cache = mask_atten_cache.to(dtype).to(device)
        if self.atten_mask_cache.device != device or self.atten_mask_cache.dtype != dtype:
            self.atten_mask_cache = self.atten_mask_cache.to(dtype).to(device)

    def get_attn_mask(
            self, max_s: int, dtype: torch.dtype, device: torch.device
    ):
        self.update_attn_cache(dtype, device, max_s)
        return self.atten_mask_cache[:max_s, :max_s]
