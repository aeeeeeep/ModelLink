from functools import wraps
import megatron
from megatron.model import GPTModel
from megatron import get_args as megatron_get_args
from .model.module import MegatronModuleForCausalLM


def seq_length_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        self.seq_length = megatron_get_args().seq_length
        return fn(self, *args, **kwargs)

    return wrapper


class BaseModel(GPTModel, MegatronModuleForCausalLM):
    def __init__(self, config, num_tokentypes=0, parallel_output=True, pre_process=True, post_process=True):
        super(BaseModel, self).__init__(config=config, num_tokentypes=num_tokentypes, parallel_output=parallel_output,
                                        pre_process=pre_process, post_process=post_process)


def apply_model_patch():
    megatron.model.GPTModel = BaseModel
    megatron.model.language_model.TransformerLanguageModel.forward = (seq_length_wrapper(
        megatron.model.language_model.TransformerLanguageModel.forward))