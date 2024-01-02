# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GPT-2 model."""

import torch
from modellink import get_args
from modellink.core import tensor_parallel
from modellink.core.enums import AttnMaskType
from modellink.error_utils import check_equal
from .language_model import parallel_lm_logits
from .language_model import get_language_model

from .module import MegatronModule, MegatronModuleForCausalLM


def post_language_model_processing(lm_output, labels, logit_weights,
                                   parallel_output,
                                   fp16_lm_cross_entropy,
                                   z_loss_weight,
                                   keep_last_token):
    # Output. Format [s b h]
    output = parallel_lm_logits(
        lm_output,
        logit_weights,
        parallel_output)
    if labels is None:

        return output
    else:
        if keep_last_token:
            output = output[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()

        if fp16_lm_cross_entropy:
            check_equal(output.dtype, torch.half)
            loss = tensor_parallel.vocab_parallel_cross_entropy(output, labels)
        else:
            loss = tensor_parallel.vocab_parallel_cross_entropy(output.float(), labels)

        if z_loss_weight is not None:
            softmax_normalizer = output.max(-1).values ** 2
            z_loss = z_loss_weight * softmax_normalizer
            loss = loss + z_loss

        return loss


class GPTModel(MegatronModule, MegatronModuleForCausalLM):
    """GPT-2 Language model."""

    def __init__(self,
                 config,
                 num_tokentypes=0,
                 parallel_output=True,
                 pre_process=True,
                 post_process=True,
                 return_moe_loss=True):
        args = get_args()
        super().__init__(config=config,
                         share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights)

        self.parallel_output = parallel_output
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy
        self.return_moe_loss = return_moe_loss
        self.untie_embeddings_and_output_weights = args.untie_embeddings_and_output_weights
        self.z_loss_weight = args.z_loss_weight
        self.keep_last_token = args.keep_last_token

        self.language_model, self._language_model_key = get_language_model(
            config=config,
            num_tokentypes=num_tokentypes,
            add_pooler=False,
            encoder_attn_mask_type=AttnMaskType.causal,
            pre_process=self.pre_process,
            post_process=self.post_process,
            num_experts=args.num_experts)

        if not args.untie_embeddings_and_output_weights:
            self.initialize_word_embeddings()

    def set_input_tensor(self, input_tensor):
        """See modellink.model.transformer.set_input_tensor()"""
        self.language_model.set_input_tensor(input_tensor)

    def forward(self, input_ids, position_ids, attention_mask,
                retriever_input_ids=None,
                retriever_position_ids=None,
                retriever_attn_mask=None,
                labels=None, tokentype_ids=None, inference_params=None,
                curriculum_seqlen=None):
        args = get_args()
        if curriculum_seqlen is not None:
            args.curriculum_seqlen = curriculum_seqlen
            if curriculum_seqlen < input_ids.size()[1]:
                # seqlen-based curriculum learning
                # input_ids, position_ids, labels have size [batch size, seqlen]
                input_ids = input_ids[:, :curriculum_seqlen].contiguous()
                position_ids = position_ids[:, :curriculum_seqlen].contiguous()
                if labels is not None:
                    labels = labels[:, :curriculum_seqlen].contiguous()

                # attention_mask has size [1, 1, seqlen, seqlen]
                attention_mask = attention_mask[:, :, :curriculum_seqlen, :curriculum_seqlen].contiguous()
        else:
            if args.curriculum_learning_legacy:
                # If got a None input, need to reset curriculum_seqlen on user side
                args.curriculum_seqlen = args.seq_length

        lm_output, *moe_losses = self.language_model(
            input_ids,
            position_ids,
            attention_mask,
            retriever_input_ids=retriever_input_ids,
            retriever_position_ids=retriever_position_ids,
            retriever_attn_mask=retriever_attn_mask,
            inference_params=inference_params)
        if self.post_process:
            lm_output = post_language_model_processing(
                lm_output, labels,
                self.language_model.output_layer.weight if self.untie_embeddings_and_output_weights else self.shared_embedding_or_output_weight(),
                self.parallel_output,
                self.fp16_lm_cross_entropy,
                self.z_loss_weight,
                self.keep_last_token)

        return lm_output, moe_losses if self.return_moe_loss else lm_output

    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):

        state_dict_ = {}
        language_model_state_dict = self.language_model.state_dict_for_save_checkpoint(
            prefix=prefix, keep_vars=keep_vars)
        # MoE states need to be handled separately by DeepSpeed engine, thus
        # moving them to the top level dictionary
        if "moe_state_dict" in language_model_state_dict:
            for key in list(language_model_state_dict["moe_state_dict"].keys()):
                state_dict_[key] = language_model_state_dict["moe_state_dict"].pop(key)
            del language_model_state_dict["moe_state_dict"]
        state_dict_[self._language_model_key] = language_model_state_dict
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
        # Gather MoE states and move under language model
        moe_state_dict = {}
        for key in list(state_dict.keys()):
            if 'expert' in key and 'moe.gate.wg.weight' not in key:
                moe_state_dict[key] = state_dict.pop(key)
        if self._language_model_key in state_dict:
            state_dict = state_dict[self._language_model_key]
        if len(moe_state_dict) > 0:
            state_dict["moe_state_dict"] = moe_state_dict
        self.language_model.load_state_dict(state_dict, strict=strict)


def get_cross_entropy(is_prefix: bool):
    def CrossEntropy(output, labels):
        labels, loss_mask = labels[0], labels[1]

        args = get_args()

        losses = tensor_parallel.vocab_parallel_cross_entropy(output.contiguous().float(), labels)

        if is_prefix:
            micro_batch_size, sequence_length = loss_mask.shape
            average_tokens_per_sample: torch.Tensor
            if args.loss_on_targets_only:
                # HACK: This is useful when we obtain loss masks that are microbatch dependent.
                #   Consequently, if we want to
                #   preserve the notion that all tokens have the same impact on the loss,
                #   we can only normalise using a
                #   microbatch independent value. It should be expected weight over a microbatch.
                #   Here we still use `sequence_length`, that's batch size dependent,
                #   in order to be backwards compatible with
                #   current experiment on vanilla gpt.
                if args.reweight_loss_based_on_position_frequency:
                    reweight = torch.arange(
                        sequence_length, 0, -1, dtype=torch.float, device=loss_mask.device
                    ) / (sequence_length + 1) * 2
                    average_tokens_per_sample = reweight.flip(-1).cumsum(-1).mean()
                else:
                    average_tokens_per_sample = (sequence_length + 1) / 2
            else:
                average_tokens_per_sample = sequence_length
            expected_number_of_tokens = average_tokens_per_sample * micro_batch_size
        else:
            expected_number_of_tokens = loss_mask.sum()

        loss_mask = loss_mask.view(-1)
        loss = torch.sum(losses.view(-1) * loss_mask) / expected_number_of_tokens
        return loss

    return CrossEntropy
