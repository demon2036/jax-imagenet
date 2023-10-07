import einops
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Sequence
from functools import partial

from model.attention import MultiHeadSelfAttention


class MLP(nn.Module):
    dim: int
    expand_ratios: int
    dtype : Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        x = nn.Dense(self.dim * self.expand_ratios, dtype=self.dtype)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.dim, dtype=self.dtype)(x)
        return x


class Block(nn.Module):
    dim: int
    norm: Any
    num_heads: int
    expand_ratios: int
    dtype : Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        y = self.norm()(x)
        y = MultiHeadSelfAttention(dim=self.dim, num_heads=self.num_heads, dtype=self.dtype
                                   ,attention_type='memory_efficient')(y)
        x = x + y
        y = self.norm()(x)
        y = MLP(dim=self.dim, expand_ratios=self.expand_ratios, dtype=self.dtype)(y)
        x = x + y
        return x


class Stage(nn.Module):
    dim: int
    norm: Any
    patch_size: int
    num_heads: int
    expand_ratios: int
    num_blocks : int
    dtype = Any = jnp.bfloat16

    @nn.compact
    def __call__(self,x, *args, **kwargs):
        x = einops.rearrange(x,'b (n p) d ->b n (p d)', p=self.patch_size ** 2)
        x = nn.Dense(self.dim, dtype=self.dtype)(x)

        for _ in range(self.num_blocks):
            x = Block(dim=self.dim,norm=self.norm,
                      num_heads=self.num_heads, expand_ratios=self.expand_ratios, dtype=self.dtype)(x)

        return x


class PyramidViT(nn.Module):
    dims: Sequence[int]
    patch_sizes: Sequence[int]
    num_heads: Sequence[int]
    expand_ratios: Sequence[int]
    num_blocks: Sequence[int]
    num_classes: int = 1000
    dtype = Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        x=einops.rearrange(x,'b h w c->b (h w) c')

        norm=partial(nn.LayerNorm)

        for dim, patch_size, num_head, expand_ratio, num_block in zip(self.dims, self.patch_sizes, self.num_heads,
                                                                      self.expand_ratios, self.num_blocks):
            x = Stage(dim=dim,norm=norm, patch_size=patch_size, num_heads=num_head, expand_ratios=expand_ratio,
                      num_blocks=num_block)(x)

        x = jnp.mean(x, 1)
        x = nn.Dense(self.num_classes, dtype=self.dtype)(x)
        return x


PVT_Tiny = partial(PyramidViT, dims=[64, 128, 320, 512], patch_sizes=[4, 2, 2, 2], num_heads=[1, 2, 5, 8],
                   expand_ratios=[8, 8, 4, 4], num_blocks=[2, 2, 2, 2])
