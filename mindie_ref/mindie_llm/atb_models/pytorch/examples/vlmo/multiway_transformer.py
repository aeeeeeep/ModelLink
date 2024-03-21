""" Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale' - https://arxiv.org/abs/2010.11929

The official jax code is released and available at https://github.com/google-research/vision_transformer

Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert

DeiT model defs and weights from https://github.com/facebookresearch/deit,
paper `DeiT: Data-efficient Image Transformers` - https://arxiv.org/abs/2012.12877

Hacked together by / Copyright 2020 Ross Wightman
"""

from functools import partial
import os
import json
import numpy as np
import torch
import torch_npu
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from pytorch_lightning.utilities.distributed import rank_zero_info


# def load_ascend_transformer():
#     ATB_SPEED_HOME_PATH = os.environ.get("ATB_SPEED_HOME_PATH")
#     if ATB_SPEED_HOME_PATH is None:
#         raise RuntimeError("env ATB_SPEED_HOME_PATH not exist, source set_env.sh")
#     LIB_PATH = os.path.join(ATB_SPEED_HOME_PATH, "lib/libatb_speed_torch.so")
#     torch.classes.load_library(LIB_PATH)


# load_ascend_transformer()


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        # self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        x = self.fc2(x)
        # x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None

        # self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        # self.proj_drop = proj_drop

    def forward(self, x, mask=None, relative_position_bias=None):
        # print("multiway transformer attention forward")
        # print("x.shape ",x.shape)
        # x.shape  torch.Size([1, 941, 768])

        B, N, C = x.shape

        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (
                    self.q_bias, # 768
                    torch.zeros_like(self.v_bias, requires_grad=False),
                    self.v_bias, # 768
                )
            )
        # x.shape  torch.Size([1, 941, 768])
        # qkv.w   torch.Size([2304, 768])
        # qkv bias 768*3 = 2304
        
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)

        # torch.Size([1, 941, 2304])
        # 1 941 2304
        # 1 941 3 12 64
        # 3 1 12 941 64
        
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(
            2, 0, 3, 1, 4
        )  # 3 batchSize numHeads N -1
        # print(torch.allclose(qkvc, qkv, rtol=1e-4))
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        # q.shape 1 12 941 64
        # 相当于有个rope操作。

        # print("q  k  v.shape ",q.shape)
        q = q * self.scale
        # 941  768
        # 768 941
        #  941 941
        #  1 12 941 64
        #   1 12 64 941
        # 1 12 941 941

        attn = q.float() @ k.float().transpose(-2, -1)
        # print("1 attn shape",attn.shape)
        # 移动到外面，加到mask上面
        # if relative_position_bias is not None:
        #     attn = attn + relative_position_bias.unsqueeze(0)
        #     # 1  12  941 941
        # if mask is not None:
        #     mask = mask.bool()  # 1 1 1 941
        #     # print("~mask[:, None, None, :]",~mask[:, None, None, :])
        #     attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))

        attn = attn + relative_position_bias
        attn = attn.softmax(dim=-1).type_as(x)
        # attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # 768 384
        # 768 
        # proj: [768,768]
        x = self.proj(x)
        # print("final x shape ",x.shape)
        # x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        with_vlffn=False,
        layer_scale_init_values=0.1,
        max_text_len=40,
        layerid=0,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2_text = norm_layer(dim)
        self.norm2_imag = norm_layer(dim)
        self.dim = dim
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_text = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.mlp_imag = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.mlp_vl = None
        if with_vlffn:
            self.mlp_vl = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=drop,
            )
            self.norm2_vl = norm_layer(dim)

        self.gamma_1 = (
            nn.Parameter(
                layer_scale_init_values * torch.ones((dim)), requires_grad=True
            )
            if layer_scale_init_values is not None
            else 1.0
        )
        self.gamma_2 = (
            nn.Parameter(
                layer_scale_init_values * torch.ones((dim)), requires_grad=True
            )
            if layer_scale_init_values is not None
            else 1.0
        )

        self.max_text_len = max_text_len

    def forward(self, x, mask=None, modality_type=None, relative_position_bias=None):

        norm = self.norm1(x)

        orgattn = self.attn(
            norm, mask=mask, relative_position_bias=relative_position_bias
        )
        x = x + self.gamma_1 * orgattn

        if modality_type == "image":
            x = x + self.gamma_2 * self.mlp_imag(self.norm2_imag(x))
        elif modality_type == "text":
            x = x + self.gamma_2 * self.mlp_text(self.norm2_text(x))
        else:
            if self.mlp_vl is None:
                # print("self.mlp_vl is None")
                x_text = x[:, : self.max_text_len]
                x_imag = x[:, self.max_text_len :]
                x_text = x_text + self.gamma_2 * self.mlp_text(self.norm2_text(x_text))
                x_imag = x_imag + self.gamma_2 * self.mlp_imag(self.norm2_imag(x_imag))
                x = torch.cat([x_text, x_imag], dim=1)
            else:
                x = x + self.gamma_2 * self.mlp_vl(self.norm2_vl(x))
        # print("org x shape",x.shape)

        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        no_patch_embed_bias=False,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False if no_patch_embed_bias else True,
        )

    def forward(self, x):
        # print("multiway transformer PatchEmbed forward")

        B, C, H, W = x.shape
        x = self.proj(x)
        return x


class MultiWayTransformer(nn.Module):
    """Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=None,
        need_relative_position_embed=True,
        use_abs_pos_emb=False,
        layer_scale_init_values=0.1,
        vlffn_start_layer_index=10,
        config=None,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
            need_relative_position_embed (bool): enable relative position bias on self-attention
            use_abs_pos_emb (bool): enable abs pos emb
            layer_scale_init_values (float or None): layer scale init values, set None to disable
            vlffn_start_layer_index (int): vl-ffn start index
            config: (dict): other hyper from pytorch-lighting
        """
        super().__init__()
        drop_path_rate = drop_path_rate if config is None else config["drop_path_rate"]
        rank_zero_info("drop path rate: {}".format(drop_path_rate))
        self.use_abs_pos_emb = use_abs_pos_emb
        self.need_relative_position_embed = need_relative_position_embed

        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.depth = depth
        self.embed_dim = embed_dim
        self.vlffn_start_layer_index = vlffn_start_layer_index
        if config["loss_names"]["textmlm"] > 0:
            self.vlffn_start_layer_index = depth
            rank_zero_info(
                "Set vlffn_start_layer_index={} for text-only pretraining".format(
                    self.vlffn_start_layer_index
                )
            )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = (
            nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
            if self.use_abs_pos_emb
            else None
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        # print("self.vlffn_start_layer_index ",self.vlffn_start_layer_index)
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    layerid=i,
                    norm_layer=norm_layer,
                    with_vlffn=(i >= self.vlffn_start_layer_index),
                    layer_scale_init_values=layer_scale_init_values,
                    max_text_len=config["max_text_len"],
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def visual_embed(self, _x):

        x = self.patch_embed(_x)
        x = x.flatten(2).transpose(1, 2)
        B, L, _ = x.shape

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        if self.pos_embed is not None:
            x = x + self.pos_embed
        # x = self.pos_drop(x)

        x_mask = torch.ones(x.shape[0], x.shape[1])

        return x, x_mask


# VLMo base/p16
@register_model
def vlmo_base_patch16(pretrained=False, **kwargs):
    img_size = kwargs.pop("img_size", 224)
    model = MultiWayTransformer(
        img_size=img_size,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        vlffn_start_layer_index=10,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


# VLMo large/p16
@register_model
def vlmo_large_patch16(pretrained=False, **kwargs):
    img_size = kwargs.pop("img_size", 224)
    model = MultiWayTransformer(
        img_size=img_size,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        vlffn_start_layer_index=21,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


# VLMo base+/p16
@register_model
def vlmo_base_plus_patch16(pretrained=False, **kwargs):
    img_size = kwargs.pop("img_size", 224)
    model = MultiWayTransformer(
        img_size=img_size,
        patch_size=16,
        embed_dim=544,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        vlffn_start_layer_index=21,
        use_abs_pos_emb=True,
        need_relative_position_embed=False,
        layer_scale_init_values=None,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model
