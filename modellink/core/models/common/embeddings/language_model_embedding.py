from torch import Tensor

from megatron.core import tensor_parallel
from megatron.training import get_args


def language_model_embedding_forward(self, input_ids: Tensor, position_ids: Tensor, tokentype_ids: int = None):
    """Forward pass of the embedding module.

    Args:
        input_ids (Tensor): The input tokens
        position_ids (Tensor): The position id's used to calculate position embeddings
        tokentype_ids (int): The token type ids. Used when args.bert_binary_head is set to True. Defaults to None

    Returns:
        Tensor: The output embeddings
    """
    args = get_args()
    word_embeddings = self.word_embeddings(input_ids) * args.embedding_multiplier_scale
    if self.add_position_embedding:
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = word_embeddings + position_embeddings
    else:
        embeddings = word_embeddings

    # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
    embeddings = embeddings.transpose(0, 1).contiguous()

    if tokentype_ids is not None:
        assert self.tokentype_embeddings is not None
        # [b s h] -> [s b h] (So that it can be added with embeddings)
        tokentype_embedding = self.tokentype_embeddings(tokentype_ids).permute(1, 0, 2)
        embeddings = embeddings + tokentype_embedding
    else:
        assert self.tokentype_embeddings is None

    # If the input flag for fp32 residual connection is set, convert for float.
    if self.config.fp32_residual_connection:
        embeddings = embeddings.float()

    # Dropout.
    if self.config.sequence_parallel:
        embeddings = tensor_parallel.scatter_to_sequence_parallel_region(embeddings)
        # `scatter_to_sequence_parallel_region` returns a view, which prevents
        # the original tensor from being garbage collected. Clone to facilitate GC.
        # Has a small runtime cost (~0.5%).
        if self.config.clone_scatter_output_in_embedding:
            embeddings = embeddings.clone()
        with tensor_parallel.get_cuda_rng_tracker().fork():
            embeddings = self.embedding_dropout(embeddings)
    else:
        embeddings = self.embedding_dropout(embeddings)

    return embeddings
