from functools import partial

import einops
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any
from torchvision.models import VisionTransformer


class MultiHeadAttention(nn.Module):
    dim: int
    heads: int
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        b, *_ = x.shape

        x = nn.Dense(3 * self.dim, dtype=self.dtype, use_bias=False)(x)
        q, k, v = tuple(einops.rearrange(x, 'b t (d k h) -> k b h t d ', k=3, h=self.heads))
        # q = nn.LayerNorm()(q)
        # k = nn.LayerNorm()(k)
        scaled_dot_prod = jnp.einsum('b h i d , b h j d -> b h i j', q, k) * q.shape[-1] ** -0.5
        attn = nn.softmax(scaled_dot_prod, axis=-1)
        out = jnp.einsum('b h i j , b  h j d -> b h i d', attn, v)
        out = einops.rearrange(out, 'b h i d-> b i ( h d)')
        out = nn.Dense(self.dim, dtype=self.dtype)(out)
        return out


class MLP(nn.Module):
    dim: int
    expand_ratio: int = 4
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        x = nn.Dense(self.dim * self.expand_ratio, dtype=self.dtype)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.dim, dtype=self.dtype)(x)
        return x


class Block(nn.Module):
    dim: int
    norm: Any
    nums_head: int
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        y = self.norm()(x)
        y = MultiHeadAttention(self.dim, self.nums_head, self.dtype)(y)
        x = y + x
        y = self.norm()(x)
        y = MLP(self.dim)(y)
        return x + y


"""


class Block(nn.Module):
    dim: int
    norm: Any
    nums_head: int
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        y = self.norm()(x)
        return MultiHeadAttention(self.dim, self.nums_head, self.dtype)(y) + MLP(self.dim)(y) + x
"""


class Embedding(nn.Module):
    din: int
    patch_size: int

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        i = 0
        res = 0
        while i < self.patch_size:
            i += 2
            res += nn.Conv(self.din, (i, i), (self.patch_size, self.patch_size), padding='same')(x)
        return res


class ViT(nn.Module):
    dim: int
    nums_head: int
    patch_size: int
    depth: int
    num_classes: int = 1000
    classifier: str = 'token'
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        b, *_ = x.shape
        norm = partial(nn.LayerNorm, dtype=self.dtype)

        x=Embedding(self.dim,self.patch_size)(x)

        # x = nn.Conv(self.dim, (self.patch_size, self.patch_size), (self.patch_size, self.patch_size), dtype=self.dtype)(
        #     x)
        x = einops.rearrange(x, 'b h w c->b (h w) c')

        if self.classifier == 'token':
            cls = self.param('cls', nn.initializers.zeros, (1, 1, self.dim), self.dtype)
            cls = jnp.tile(cls, [b, 1, 1])
            x = jnp.concatenate([cls, x], axis=1)
        for i in range(self.depth):
            x = Block(self.dim, norm, self.nums_head, self.dtype)(x)

        x = norm()(x)
        if self.classifier == 'token':
            x = x[:, 0]

        x = nn.Dense(self.num_classes, dtype=self.dtype)(x)
        return x


ViT_T = partial(ViT, dim=192, nums_head=3, patch_size=16, depth=12)
ViT_S = partial(ViT, dim=384, nums_head=6, patch_size=16, depth=12)
ViT_B = partial(ViT, dim=768, nums_head=12, patch_size=16, depth=12)
