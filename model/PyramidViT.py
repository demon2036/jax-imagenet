import einops
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Sequence
from functools import partial

from model.attention import MultiHeadSelfAttention


def posemb_sincos_2d(h, w, width, temperature=10_000., dtype=jnp.float32):
    """Follows the MoCo v3 logic."""
    y, x = jnp.mgrid[:h, :w]

    assert width % 4 == 0, "Width must be mult of 4 for sincos posemb"
    omega = jnp.arange(width // 4) / (width // 4 - 1)
    omega = 1. / (temperature ** omega)
    y = jnp.einsum("m,d->md", y.flatten(), omega)
    x = jnp.einsum("m,d->md", x.flatten(), omega)
    pe = jnp.concatenate([jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)], axis=1)
    return jnp.asarray(pe, dtype)[None, :, :]


def get_posemb(self, typ, seqshape, width, name, dtype=jnp.float32):
    if typ == "learn":
        return self.param(name, nn.initializers.normal(stddev=1 / np.sqrt(width)),
                          (1, np.prod(seqshape), width), dtype)
    elif typ == "sincos2d":
        return posemb_sincos_2d(*seqshape, width, dtype=dtype)
    else:
        raise ValueError(f"Unknown posemb type: {typ}")


class MLP(nn.Module):
    dim: int
    expand_ratios: int
    dtype: Any = jnp.bfloat16

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
    reduction_ratio: int
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        y = self.norm()(x)
        y = MultiHeadSelfAttention(dim=self.dim, num_heads=self.num_heads, dtype=self.dtype
                                   , attention_type='math', reduction_ratio=self.reduction_ratio)(y)
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
    reduction_ratio: int
    num_blocks: int
    posemb: str = 'sincos2d'

    dtype = Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        x = einops.rearrange(x, 'b (n p) d ->b n (p d)', p=self.patch_size ** 2)
        x = nn.Dense(self.dim, dtype=self.dtype)(x)
        b, n, d = x.shape
        x = self.norm()(x)
        x = x + get_posemb(self, self.posemb, (n ** 0.5, n ** 0.5), d, "pos_embedding", x.dtype)

        for _ in range(self.num_blocks):
            x = Block(dim=self.dim, norm=self.norm, num_heads=self.num_heads, expand_ratios=self.expand_ratios,
                      dtype=self.dtype, reduction_ratio=self.reduction_ratio)(x)

        return x


class PyramidViT(nn.Module):
    dims: Sequence[int]
    patch_sizes: Sequence[int]
    num_heads: Sequence[int]
    expand_ratios: Sequence[int]
    num_blocks: Sequence[int]
    reduction_ratios: Sequence[int]
    num_classes: int = 1000
    dtype = Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        x = einops.rearrange(x, 'b h w c->b (h w) c')

        norm = partial(nn.LayerNorm, dtype=self.dtype)

        for dim, patch_size, num_head, expand_ratio, num_block, reduction_ratio in zip(self.dims, self.patch_sizes,
                                                                                       self.num_heads,
                                                                                       self.expand_ratios,
                                                                                       self.num_blocks,
                                                                                       self.reduction_ratios):
            x = Stage(dim=dim, norm=norm, patch_size=patch_size, num_heads=num_head, expand_ratios=expand_ratio,
                      num_blocks=num_block, reduction_ratio=reduction_ratio)(x)

        x = norm()(x)
        x = jnp.mean(x, 1)
        x = nn.Dense(self.num_classes, dtype=self.dtype)(x)
        return x


PVT_Tiny = partial(PyramidViT, dims=[64, 128, 320, 512], patch_sizes=[4, 2, 2, 2], num_heads=[1, 2, 5, 8],
                   expand_ratios=[8, 8, 4, 4], num_blocks=[2, 2, 2, 2], reduction_ratios=[8, 4, 2, 1])
PVT_S = partial(PyramidViT, dims=[64, 128, 320, 512], patch_sizes=[4, 2, 2, 2], num_heads=[1, 2, 5, 8],
                expand_ratios=[8, 8, 4, 4], num_blocks=[3, 3, 6, 3], reduction_ratios=[8, 4, 2, 1])
PVT_M = partial(PyramidViT, dims=[64, 128, 320, 512], patch_sizes=[4, 2, 2, 2], num_heads=[1, 2, 5, 8],
                expand_ratios=[8, 8, 4, 4], num_blocks=[3, 3, 18, 3], reduction_ratios=[8, 4, 2, 1])
PVT_L = partial(PyramidViT, dims=[64, 128, 320, 512], patch_sizes=[4, 2, 2, 2], num_heads=[1, 2, 5, 8],
                expand_ratios=[8, 8, 4, 4], num_blocks=[3, 8, 27, 3], reduction_ratios=[8, 4, 2, 1])
