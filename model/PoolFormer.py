import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any
from functools import partial


class MLP(nn.Module):
    dim: int
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        x = nn.Dense(self.dim * 4, dtype=self.dtype)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.dim, dtype=self.dtype)(x)
        return x


class Block(nn.Module):
    dim: int
    norm: Any
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        y = self.norm()(x)
        y = nn.max_pool(y, (3, 3), padding='same')
        x = y
        y = self.norm()(x)
        y = MLP(self.dim, self.dtype)(y)
        return x + y


class PoolFormer(nn.Module):
    dim: int
    patch_size: int
    depth: int
    num_classes: int = 1000
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        norm = partial(nn.LayerNorm, epsilon=1e-6, dtype=self.dtype)
        x = nn.Conv(self.dim, (self.patch_size, self.patch_size), (self.patch_size, self.patch_size), dtype=self.dtype)(
            x)

        for _ in range(self.depth):
            x = Block(dim=self.dim, norm=norm, dtype=self.dtype)(x)
        x = norm()(x)
        x = jnp.mean(x, [1,2])
        x = nn.Dense(self.num_classes, dtype=self.dtype)(x)
        return x


PoolFormer_T = partial(PoolFormer, dim=192, patch_size=16, depth=12)
PoolFormer_S = partial(PoolFormer, dim=384, patch_size=16, depth=12)
PoolFormer_B = partial(PoolFormer, dim=768, patch_size=16, depth=12)
