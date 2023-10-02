from torchvision.models import ConvNeXt
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Sequence
from functools import partial


class Block(nn.Module):
    channels: int
    norm: Any
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        c = x.shape[-1]
        hidden = nn.Conv(self.channels, (7, 7), padding='same', dtype=self.dtype, feature_group_count=c)(x)
        hidden = self.norm()(hidden)
        hidden = nn.Conv(self.channels * 4, (1, 1), padding='same', dtype=self.dtype, )(hidden)
        hidden = nn.gelu(hidden)
        hidden = nn.Conv(self.channels, (1, 1), padding='same', dtype=self.dtype)(hidden)
        x = x + hidden
        return x


class ConvNext(nn.Module):
    num_classes: int = 1000
    out_channels: Sequence[int] = (64, 128, 256, 512)
    num_blocks: Sequence[int] = (3, 3, 9, 3)  # (2, 4, 14, 1)
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        norm = partial(nn.LayerNorm, dtype=self.dtype)
        # Stem
        x = nn.Conv(self.out_channels[0], (4, 4), (4, 4), dtype=self.dtype)(x)

        for out_channel, num_block in zip(self.out_channels, self.num_blocks):
            x = norm()(x)
            x = nn.Conv(out_channel, (2, 2), (2, 2), padding='same', dtype=self.dtype)(x)
            for _ in range(num_block):
                x = Block(out_channel, norm=norm, dtype=self.dtype)(x)

        x = jnp.mean(x, axis=[1, 2])
        x = norm()(x)
        x = nn.Dense(self.num_classes, dtype=self.dtype)(x)

        return x


ConvNext_T = partial(ConvNext, num_blocks=[3, 3, 9, 3], out_channels=[96, 192, 384, 768])
ConvNext_S = partial(ConvNext, num_blocks=[3, 3, 27, 3], out_channels=[96, 192, 384, 768])
ConvNext_B = partial(ConvNext, num_blocks=[3, 3, 27, 3], out_channels=[128, 256, 512, 1024])
ConvNext_L = partial(ConvNext, num_blocks=[3, 3, 27, 3], out_channels=[192, 384, 768, 1536])
ConvNext_XL = partial(ConvNext, num_blocks=[3, 3, 27, 3], out_channels=[256, 512, 1024, 2048])
