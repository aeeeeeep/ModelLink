# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
import torch

from torch import nn


class PositionRotaryEmbedding(nn.Module):
    def __init__(self, inv_freq):
        super().__init__()

        self.inv_freq = inv_freq
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_cached_total = None
        self._sin_cached_total = None
        self._cos_k_cached = None
        self._sin_k_cached = None

    @classmethod
    def static(cls, dim, base, device):
        try:
            inv_freq = 1.0 / (
                    base
                    ** (torch.arange(0, dim, 2, device=device, dtype=torch.double) / dim)
            ).to(torch.float)
        except ZeroDivisionError as e:
            raise ZeroDivisionError from e
        return cls(inv_freq)

    @classmethod
    def load(cls, prefix, weights):
        # Always load this in float32 !
        dtype = weights.dtype
        weights.dtype = torch.float32
        inv_freq = weights.get_tensor(f"{prefix}.inv_freq")
        weights.dtype = dtype
        return cls(inv_freq)

    def update_cos_sin_cache(self, dtype, device, seqlen):
        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if (
                seqlen > self._seq_len_cached
                or self._cos_cached.device != device
                or self._cos_cached.dtype != dtype
        ):
            self._seq_len_cached = seqlen
            t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
            # Don't do einsum, it converts fp32 to fp16 #freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            freqs = torch.outer(t, self.inv_freq.to(device=t.device))
            self._cos_cached = torch.cos(freqs).to(dtype)
            self._sin_cached = torch.sin(freqs).to(dtype)

    def update_cos_sin_cache_total(self, dtype, device, seqlen):
        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if (
                seqlen > self._seq_len_cached
                or self._cos_cached_total.device != device
                or self._cos_cached_total.dtype != dtype
        ):
            self._seq_len_cached = seqlen
            t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
            # Don't do einsum, it converts fp32 to fp16 # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            freqs = torch.outer(t, self.inv_freq.to(device=t.device))
            freqs = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached_total = torch.cos(freqs).to(dtype)
            self._sin_cached_total = torch.sin(freqs).to(dtype)

    def get_cos_sin(
            self, position_ids: torch.Tensor, max_s: int, dtype: torch.dtype
    ):
        """
        Return cos and sin for the asked position ids
        """

        self.update_cos_sin_cache(dtype, position_ids.device, max_s)

        cos = torch.nn.functional.embedding(position_ids, self._cos_cached)
        sin = torch.nn.functional.embedding(position_ids, self._sin_cached)
        
        return cos.unsqueeze(1), sin.unsqueeze(1)

    def get_cos_sin_total(
            self, position_ids: torch.Tensor, max_s: int, dtype: torch.dtype
    ):
        """
        Return cos and sin for the asked position ids
        """

        self.update_cos_sin_cache_total(dtype, position_ids.device, max_s)

        cos = torch.nn.functional.embedding(position_ids, self._cos_cached_total)
        sin = torch.nn.functional.embedding(position_ids, self._sin_cached_total)

        return cos, sin

    def get_cos_cached_total(self):
        return self._cos_cached_total

    def get_sin_cached_total(self):
        return self._sin_cached_total

    def get_cos_sin_cached_total(self, position_ids):
        cos = torch.index_select(self._cos_cached_total, 0, position_ids)
        sin = torch.index_select(self._sin_cached_total, 0, position_ids)
        return cos, sin

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        rotary_dim = cos.shape[-1]
        x1 = x[..., :rotary_dim]
        x2 = x[..., rotary_dim: 2 * rotary_dim]
        x_rope = x[..., :2 * rotary_dim]

        # do original forward
        dtype = x.dtype
        cos_compute = torch.cat((cos, cos), dim=-1)
        sin_compute = torch.cat((sin, sin), dim=-1)
        x_rope = (x_rope * cos_compute) + (torch.cat((-x2, x1), dim=-1) * sin_compute)
        x[..., :2 * rotary_dim] = x_rope
        x = x.to(dtype)
        return x
