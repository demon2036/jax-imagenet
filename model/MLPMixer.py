import einops
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any
from functools import partial


class MLPBlock(nn.Module):
    dim: int
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        out_dim = x.shape[-1]
        x = nn.Conv(self.dim, (1, 1), dtype=self.dtype)(x)
        x = nn.gelu(x)
        x = nn.Conv(out_dim, (1, 1), dtype=self.dtype)(x)
        return x


class MixerBlock(nn.Module):
    tokens_mlp_dim: int
    channels_mlp_dim: int
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        y = nn.LayerNorm(dtype=self.dtype)(x)
        y = einops.rearrange(y, 'b n d -> b d n')
        y = MLPBlock(self.tokens_mlp_dim, dtype=self.dtype)(y)
        y = einops.rearrange(y, 'b d n->b n d ')
        x = x + y
        y = nn.LayerNorm()(x)
        return MLPBlock(self.channels_mlp_dim, dtype=self.dtype)(y) + x


class MLPMixer(nn.Module):
    dim: int = 512
    tokens_mlp_dim: int = 512
    channels_mlp_dim: int = 2048
    patch_size: int = 32
    num_blocks: int = 8
    num_classes: int = 1000
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        x = nn.Conv(self.dim, (self.patch_size, self.patch_size), (self.patch_size, self.patch_size), dtype=self.dtype)(
            x)
        x = einops.rearrange(x, 'b h w c->b (h w) c')

        for _ in range(self.num_blocks):
            x = MixerBlock(self.tokens_mlp_dim, self.channels_mlp_dim, dtype=self.dtype)(x)
        x = nn.LayerNorm()(x)
        x = jnp.mean(x, axis=[1])
        x = nn.Dense(self.num_classes, dtype=self.dtype)(x)
        return x


MLPMixer_S_32 = partial(MLPMixer, tokens_mlp_dim=256, channels_mlp_dim=2048, num_blocks=8, dim=512, patch_size=32)
MLPMixer_S_16 = partial(MLPMixer, tokens_mlp_dim=256, channels_mlp_dim=2048, num_blocks=8, dim=512, patch_size=16)
MLPMixer_B_32 = partial(MLPMixer, tokens_mlp_dim=384, channels_mlp_dim=3072, num_blocks=12, dim=768, patch_size=32)
MLPMixer_B_16 = partial(MLPMixer, tokens_mlp_dim=384, channels_mlp_dim=3072, num_blocks=12, dim=768, patch_size=16)
