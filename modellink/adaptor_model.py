from functools import wraps
import torch
import megatron
from megatron.model import GPTModel
from megatron import get_args
from megatron.core import tensor_parallel
from megatron.model.module import MegatronModule
from megatron.model.enums import AttnMaskType
from megatron.model.language_model import parallel_lm_logits
from megatron.model.language_model import get_language_model
from megatron.model import transformer
from modellink.model.utils import get_inverted_mask
from megatron.core import mpu

from .model.module import MegatronModuleForCausalLM


def _build_alibi_tensor(max_seq_len, num_attention_heads, batch_size, square_alibi_mask, fill_neg_inf):
    """Returns tensor shaped (batch_size * num_attention_heads, 1, max_seq_len)"""

    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = (2 ** (-2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2 * closest_power_of_2)[0::2][
                                                               :n - closest_power_of_2]

    def _fill_with_neg_inf(t):
        """FP16-compatible function that fills a tensor with -inf."""
        return t.float().fill_(float("-inf")).type_as(t)

    def _buffered_future_mask(maxpos, alibi, attn_heads):
        _future_mask = torch.triu(_fill_with_neg_inf(torch.zeros([maxpos, maxpos])), 1)
        _future_mask = _future_mask.unsqueeze(0) + alibi
        return _future_mask[:attn_heads, :maxpos, :maxpos]

    slopes = torch.Tensor(get_slopes(num_attention_heads))

    if square_alibi_mask:
        position_point = torch.arange(max_seq_len) - max_seq_len + 1
        position_point = position_point.unsqueeze(0).unsqueeze(0).expand(num_attention_heads, max_seq_len, -1)
        diag = torch.diag(position_point[0])
        position_point = position_point - diag.unsqueeze(0).unsqueeze(0).transpose(-1, -2)
        alibi = slopes.unsqueeze(1).unsqueeze(1) * position_point
    else:
        alibi = slopes.unsqueeze(1).unsqueeze(1) * torch.arange(max_seq_len).unsqueeze(0).unsqueeze(0).expand(
            num_attention_heads, -1, -1)

    # Select the part of the tensor that corresponds to our tensor parallel index.
    tp_world_size = parallel_state.get_tensor_model_parallel_world_size()
    tp_index = parallel_state.get_tensor_model_parallel_rank()
    alibi = alibi.reshape((tp_world_size, -1, *alibi.shape[1:]))[tp_index]

    if fill_neg_inf:
        return _buffered_future_mask(max_seq_len, alibi, num_attention_heads)
    return alibi


def seq_length_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        self.seq_length = get_args().seq_length
        return fn(self, *args, **kwargs)

    return wrapper


def core_attention_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *arg, **kwargs):
        fn(self, *arg, **kwargs)
        args = get_args()
        self.square_alibi_mask = args.square_alibi_mask
        self.max_seq_length = args.seq_length
        self.fill_neg_inf = args.fill_neg_inf
        self.beta = 1.0
        self.apply_query_key_layer_scaling = arg[1].apply_query_key_layer_scaling
        if self.apply_query_key_layer_scaling:
            self.beta = 1.0 / self.layer_number
        if args.use_alibi_position_embedding:
            self.alibi = _build_alibi_tensor(args.seq_length, args.num_attention_heads,
                                             args.micro_batch_size, args.square_alibi_mask,
                                             args.fill_neg_inf).to(torch.cuda.current_device())
            if args.params_dtype == torch.float16:
                self.alibi = self.alibi.to(torch.float16)
            elif args.params_dtype == torch.bfloat16:
                self.alibi = self.alibi.to(torch.bfloat16)
        else:
            self.alibi = None

    return wrapper


def core_attention_forward_wrapper(fn):
    @wraps(fn)
    def wrapper(self, query_layer, key_layer, value_layer, attention_mask):
        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================

        # [b, np, sq, sk]
        output_size = (query_layer.size(1),
                       query_layer.size(2),
                       query_layer.size(0),
                       key_layer.size(0))

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.reshape(output_size[2],
                                          output_size[0] * output_size[1], -1)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key_layer = key_layer.view(output_size[3],
                                   output_size[0] * output_size[1], -1)

        if self.alibi is None:
            matmul_result = None
        elif self.fill_neg_inf:
            _alibi = self.alibi[:, :output_size[3], :output_size[3]]
            attention_mask = attention_mask.repeat(output_size[0], 1, 1, 1)[:output_size[0], :, :, :]
            matmul_result = get_inverted_mask(attention_mask, _alibi).view(-1, output_size[2],
                                                                           output_size[2]).contiguous()
        else:
            matmul_result = self.alibi[:, :, :output_size[3]].repeat(output_size[0], 1, 1)

        # preallocting input tensor: [b * np, sq, sk]
        matmul_input_buffer = mpu.get_global_memory_buffer().get_tensor(
            (output_size[0] * output_size[1], output_size[2], output_size[3]),
            query_layer.dtype, "mpu")

        # Raw attention scores. [b * np, sq, sk]
        if self.alibi is None:
            matmul_result = torch.baddbmm(
                matmul_input_buffer,
                query_layer.transpose(0, 1),  # [b * np, sq, hn]
                key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
                beta=0.0, alpha=(1.0 / self.norm_factor))
        else:
            q_trans = query_layer.transpose(0, 1).contiguous()
            k_trans = key_layer.transpose(0, 1).transpose(1, 2).contiguous()
            matmul_result = self.beta * matmul_result + torch.bmm(q_trans, k_trans) * (1.0 / self.norm_factor)
        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        if self.square_alibi_mask:
            attention_scores = torch.max(
                attention_scores, torch.tensor(torch.finfo(attention_scores.dtype).min)
            )
            attention_probs = torch.nn.functional.softmax(attention_scores, -1)
        else:
            attention_probs = self.scale_mask_softmax(attention_scores,
                                                      attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        if not self.sequence_parallel:
            with tensor_parallel.get_cuda_rng_tracker().fork():
                attention_probs = self.attention_dropout(attention_probs)
        else:
            attention_probs = self.attention_dropout(attention_probs)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (value_layer.size(1),
                       value_layer.size(2),
                       query_layer.size(0),
                       value_layer.size(3))

        # change view [sk, b * np, hn]
        value_layer = value_layer.view(value_layer.size(0),
                                       output_size[0] * output_size[1], -1)

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1],
                                               output_size[2], -1)

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + \
                                  (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer

    return wrapper


class BaseModel(GPTModel, MegatronModuleForCausalLM):
    def __init__(self, config, num_tokentypes=0, parallel_output=True, pre_process=True, post_process=True):
        super(BaseModel, self).__init__(config=config, num_tokentypes=num_tokentypes, parallel_output=parallel_output,
                                        pre_process=pre_process, post_process=post_process)


def apply_model_patch():
    megatron.model.transformer.CoreAttention.__init__ = core_attention_wrapper(
        megatron.model.transformer.CoreAttention.__init__)
    megatron.model.transformer.CoreAttention.forward = core_attention_forward_wrapper(
        megatron.model.transformer.CoreAttention.forward)
        
    megatron.model.GPTModel = GPTModel
    megatron.model.language_model.TransformerLanguageModel.forward = (seq_length_wrapper(
        megatron.model.language_model.TransformerLanguageModel.forward))


def post_language_model_processing(lm_output, labels, logit_weights,
                                   parallel_output,
                                   fp16_lm_cross_entropy):
    # Output. Format [s b h]
    output = parallel_lm_logits(
        lm_output,
        logit_weights,
        parallel_output)

    if labels is None:
        # [s b h] => [b s h]
        return output.transpose(0, 1).contiguous()
    else:
        # [b s] => [s b]
        labels = labels.transpose(0, 1).contiguous()
        if fp16_lm_cross_entropy:
            if output.dtype != torch.half:
                raise ValueError("Wrong output dtype when fp16_lm_cross_entropy.")
            loss = tensor_parallel.vocab_parallel_cross_entropy(output, labels)
        else:
            loss = tensor_parallel.vocab_parallel_cross_entropy(output.float(), labels)

        # [s b] => [b, s]
        loss = loss.transpose(0, 1).contiguous()
        return loss


class GPTModel(MegatronModule, MegatronModuleForCausalLM):
    """GPT-2 Language model."""

    def __init__(self,
                 config,
                 num_tokentypes=0,
                 parallel_output=True,
                 pre_process=True,
                 post_process=True):
        args = get_args()
        super().__init__(config=config,
                         share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights)

        self.parallel_output = parallel_output
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy
        self.untie_embeddings_and_output_weights = args.untie_embeddings_and_output_weights

        self.language_model, self._language_model_key = get_language_model(
            config=config,
            num_tokentypes=num_tokentypes,
            add_pooler=False,
            encoder_attn_mask_type=AttnMaskType.causal,
            pre_process=self.pre_process,
            post_process=self.post_process)

        if not args.untie_embeddings_and_output_weights:
            self.initialize_word_embeddings()

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.language_model.set_input_tensor(input_tensor)

    def forward(self, input_ids, position_ids, attention_mask,
                retriever_input_ids=None,
                retriever_position_ids=None,
                retriever_attn_mask=None,
                labels=None, tokentype_ids=None, inference_params=None):

        lm_output = self.language_model(
            input_ids,
            position_ids,
            attention_mask,
            retriever_input_ids=retriever_input_ids,
            retriever_position_ids=retriever_position_ids,
            retriever_attn_mask=retriever_attn_mask,
            inference_params=inference_params)

        if self.post_process:
            return post_language_model_processing(
                lm_output, labels,
                self.language_model.output_layer.weight if self.untie_embeddings_and_output_weights else self.shared_embedding_or_output_weight(),
                self.parallel_output,
                self.fp16_lm_cross_entropy)
        else:
            return lm_output

    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):

        state_dict_ = {}
        state_dict_[self._language_model_key] \
            = self.language_model.state_dict_for_save_checkpoint(
            prefix=prefix, keep_vars=keep_vars)
        # Save word_embeddings.
        if self.post_process and not self.pre_process and not self.untie_embeddings_and_output_weights:
            state_dict_[self._word_embeddings_for_head_key] \
                = self.word_embeddings.state_dict(prefix=prefix,
                                                  keep_vars=keep_vars)
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        # Load word_embeddings.
        if self.post_process and not self.pre_process and not self.untie_embeddings_and_output_weights:
            self.word_embeddings.load_state_dict(
                state_dict[self._word_embeddings_for_head_key], strict=strict)
        if self._language_model_key in state_dict:
            state_dict = state_dict[self._language_model_key]
        self.language_model.load_state_dict(state_dict, strict=strict)
