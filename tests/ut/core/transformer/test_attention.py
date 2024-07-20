import math
import pytest
import torch
import torch_npu
import torch.distributed as dist

# To activate modellink.patches.__init__
import modellink
from megatron.training.global_vars import set_args
from megatron.training.arguments import parse_args
from megatron.legacy.model.transformer import FlashSelfAttention
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.dot_product_attention import DotProductAttention
import megatron.core.parallel_state as mpu
from mindspeed.core.context_parallel.ulysses_context_parallel import UlyssesContextAttention
from mindspeed.core.parallel_state import get_context_parallel_group_for_hybrid_ulysses

from tests.common import DistributedTest
from tests.ut.utils import initialize_model_parallel
from modellink.core.transformer import get_attention_mask, MUST_COMPRESS
from modellink.core.transformer.mask_generator import set_attention_mask
from modellink.utils import seed_all


def get_data_on_this_cp_rank(data, r_size, u_size, cp_rank, dim=0):
    """ Slice data along sequence dimension into multiple chunks,
        which are parallelized across GPUs in a context parallel group.
        Dispatch data in a striped way for load-balance.
    """
    cp_size = r_size * u_size
    if r_size == 1:
        data = data.chunk(cp_size, dim=dim)[cp_rank]
    elif u_size == 1:
        data = data.view(*data.shape[0:dim], 2 * cp_size, data.shape[dim] // (2 * cp_size), *data.shape[dim + 1:])
        index = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)], device=data.device)
        data = data.index_select(dim, index)
        data = data.view(*data.shape[0:dim], -1, *data.shape[dim + 2:])
    else:
        r_rank = cp_rank // u_size
        u_rank = cp_rank % u_size
        data = data.view(*data.shape[0:dim], 2 * r_size, data.shape[dim] // (2 * r_size), *data.shape[dim + 1:])
        index = torch.tensor([r_rank, (2 * r_size - r_rank - 1)], device=data.device)
        data = data.index_select(dim, index)
        data = data.view(*data.shape[0:dim], -1, *data.shape[dim + 2:])
        data = data.chunk(u_size, dim=dim)[u_rank]
    return data


def run_attention_module(test_args, use_mcore, use_cp, cp_size, u_size):
    bs, seq_len, dtype = test_args
    r_size = cp_size // u_size
    args = parse_args(None, True)
    args.use_cp_send_recv_overlap = True
    args.cp_attention_mask_type = 'causal'

    # currently we always use FA in context parallel.
    args.use_flash_attn = True
    if u_size == 1:
        args.context_parallel_algo = 'megatron_cp_algo'
    elif u_size == 8:
        args.context_parallel_algo = 'ulysses_cp_algo'
    else:
        args.context_parallel_algo = 'hybrid_cp_algo'

    args.context_parallel_size = cp_size
    args.ulysses_degree_in_cp = u_size
    args.seq_length = seq_len
    set_args(args)
    # clear global attention mask set by last test case
    set_attention_mask(None)
    initialize_model_parallel(context_parallel_size=cp_size)
    seed_all(1234)

    rank = dist.get_rank()
    b, n, s, d = bs, 32, seq_len, 128
    scale = 1.0 / math.sqrt(d)

    q = torch.randn(s, b, n * d, dtype=dtype, device='npu', requires_grad=True)
    k = torch.randn(s, b, n * d, dtype=dtype, device='npu', requires_grad=True)
    v = torch.randn(s, b, n * d, dtype=dtype, device='npu', requires_grad=True)
    dout = torch.randn(s, b, n * d, dtype=dtype, device='npu', requires_grad=True)

    attn_mask = get_attention_mask(mode=MUST_COMPRESS)
    out = torch_npu.npu_fusion_attention( \
        q, k, v, n, 'SBH', \
        pse=None, \
        padding_mask=None, \
        atten_mask=attn_mask, \
        scale=scale, \
        pre_tockens=seq_len, \
        next_tockens=0, \
        keep_prob=1., \
        inner_precise=0, \
        sparse_mode=4 if attn_mask is not None else 0
    )[0]
    out.backward(dout)

    if use_cp:
        out_ref = get_data_on_this_cp_rank(out.clone().detach(), r_size, u_size, rank)
        k_grad_ref = get_data_on_this_cp_rank(k.grad.clone().detach(), r_size, u_size, rank)
        v_grad_ref = get_data_on_this_cp_rank(v.grad.clone().detach(), r_size, u_size, rank)

        q_ = get_data_on_this_cp_rank(q.clone().detach(), r_size, u_size, rank)
        k_ = get_data_on_this_cp_rank(k.clone().detach(), r_size, u_size, rank)
        v_ = get_data_on_this_cp_rank(v.clone().detach(), r_size, u_size, rank)
        dout_ = get_data_on_this_cp_rank(dout.clone().detach(), r_size, u_size, rank)
    else:
        out_ref = out.clone().detach()
        k_grad_ref = k.grad.clone().detach()
        v_grad_ref = v.grad.clone().detach()

        q_ = q.clone().detach()
        k_ = k.clone().detach()
        v_ = v.clone().detach()
        dout_ = dout.clone().detach()

    for x in [q_, k_, v_]:
        x.requires_grad = True

    if use_mcore:
        # test core branch, which uses core.transformer.DotProductAttention as core attention
        config = TransformerConfig(num_layers=2, hidden_size=n * d, num_attention_heads=n, use_cpu_initialization=True)
        local_attn = DotProductAttention(config=config, layer_number=1,
                                         attn_mask_type=None, attention_type='self',
                                         attention_dropout=0.)
    else:
        # test legacy branch, which uses legacy.model.transformer.FlashSelfAttention as core attention
        local_attn = FlashSelfAttention(causal=True, softmax_scale=scale, attention_dropout=0.)

    attn = local_attn
    if args.context_parallel_algo != "megatron_cp_algo":
        ulysses_group = get_context_parallel_group_for_hybrid_ulysses() \
            if args.context_parallel_algo == 'hybrid_cp_algo' else mpu.get_context_parallel_group()
        attn = UlyssesContextAttention(local_attn, ulysses_group)


    if use_mcore:
        # global mask for core branch is generated at DotProductAttention forward
        out_ = attn(q_.reshape(-1, b, n, d), k_.reshape(-1, b, n, d), v_.reshape(-1, b, n, d), None, None, None)
    else:
        out_ = attn(q_.reshape(-1, b, n, d), k_.reshape(-1, b, n, d), v_.reshape(-1, b, n, d), None)

    out_.backward(dout_)

    # same as transformer_engine
    tols = dict(atol=5e-3, rtol=5e-3)
    if dtype == torch.bfloat16:
        tols = dict(atol=2.5e-2, rtol=2.5e-2)

    # compare results
    assert torch.allclose(out_ref, out_, **tols)
    assert torch.allclose(k_grad_ref, k_.grad, **tols)
    assert torch.allclose(v_grad_ref, v_.grad, **tols)


class TestAttention(DistributedTest):
    """
    Test attention module, including DotProductAttention in megatron-core and FlashSelfAttention in legacy.
    """
    world_size = 8
    _device_name = torch_npu.npu.get_device_name(0)[:10]

    @pytest.mark.skipif(_device_name != 'Ascend910B', reason='device type is not supported, skip this UT!')
    @pytest.mark.parametrize("use_mcore", [True, False])
    def test_no_context_parallel_seq8192_bs2_bf16(self, use_mcore):
        run_attention_module((2, 8192, torch.bfloat16), use_mcore, False, 1, 1)

    @pytest.mark.skipif(_device_name != 'Ascend910B', reason='device type is not supported, skip this UT!')
    @pytest.mark.parametrize("use_mcore", [True, False])
    def test_hybrid_context_parallel_seq8192_bs2_bf16(self, use_mcore):
        run_attention_module((2, 8192, torch.bfloat16), use_mcore, True, self.world_size, 2)


    @pytest.mark.skipif(_device_name != 'Ascend910B', reason='device type is not supported, skip this UT!')
    @pytest.mark.parametrize("use_mcore", [True, False])
    def test_ring_context_parallel_seq8192_bs2_bf16(self, use_mcore):
        run_attention_module((2, 8192, torch.bfloat16), use_mcore, True, self.world_size, 1)


    @pytest.mark.skipif(_device_name != 'Ascend910B', reason='device type is not supported, skip this UT!')
    @pytest.mark.parametrize("use_mcore", [True, False])
    def test_ulysses_context_parallel_seq8192_bs2_bf16(self, use_mcore):
        run_attention_module((2, 8192, torch.bfloat16), use_mcore, True, self.world_size, 8)

