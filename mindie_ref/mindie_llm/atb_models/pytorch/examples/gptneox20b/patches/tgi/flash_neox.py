import torch
import torch.distributed
import torch_npu

from opentelemetry import trace
from transformers import AutoTokenizer, AutoConfig
from typing import Optional
from loguru import logger

from text_generation_server.models.flash_causal_lm import FlashCausalLM
from text_generation_server.models.custom_modeling.flash_neox_modeling_ascend import (
    FlashGPTNeoXForCausalLM,
)
from text_generation_server.utils import (
    initialize_torch_distributed,
    weight_files,
    Weights,
)

tracer = trace.get_tracer(__name__)


class FlashNeoXSharded(FlashCausalLM):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        quantize: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
    ):
        self.process_group, rank, world_size = initialize_torch_distributed()
        if torch_npu.npu.is_available():
            device = torch.device(f"npu:{rank}")
            dtype = torch.float16 if dtype is None else dtype

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=trust_remote_code,
        )

        config = AutoConfig.from_pretrained(
            model_id, revision=revision, trust_remote_code=trust_remote_code
        )
        config.quantize = quantize

        torch.distributed.barrier(group=self.process_group)
        filenames = weight_files(model_id, revision=revision, extension=".safetensors")
        weights = Weights(
            filenames, device=device, dtype=dtype, process_group=self.process_group
        )
        if config.quantize == "gptq":
            weights._set_gptq_params(model_id)

        model = FlashGPTNeoXForCausalLM(config, weights)

        torch.distributed.barrier(group=self.process_group)
        super(FlashNeoXSharded, self).__init__(
            model=model.to(device),
            tokenizer=tokenizer,
            num_layers=len(model.gpt_neox.layers),
            num_kv_heads=model.gpt_neox.num_heads,
            head_size=model.gpt_neox.head_size,
            dtype=dtype,
            device=device,
            rank=rank,
            world_size=world_size,
        )
