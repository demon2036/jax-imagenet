from typing import Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn


class ConvBlock(nn.Module):
    out_channels: int
    kernel_size: int
    strides: int = 1
    use_norm: bool = False
    use_act: bool = False
    dtype: str = 'bfloat16'

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        x = nn.Conv(self.out_channels, (self.kernel_size, self.kernel_size), (self.strides, self.strides),
                    padding='same', dtype=self.dtype)(x)
        """
        if self.use_norm:
            x = nn.BatchNorm()(x)
        """

        if self.use_norm:
            x = nn.GroupNorm()(x)

        if self.use_act:
            x = nn.relu(x)
        return x


class ResBlock(nn.Module):
    out_channel: int
    strides: int = 1
    dtype: str = 'bfloat16'

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        b, h, w, c = x.shape
        hidden = ConvBlock(c, 1, dtype=self.dtype, use_norm=True, use_act=True)(x)  # True
        hidden = ConvBlock(c, 3, dtype=self.dtype, use_norm=True, use_act=True)(hidden)  # True
        hidden = ConvBlock(self.out_channel, 1, dtype=self.dtype, use_norm=True, use_act=False)(hidden)  # True

        if self.out_channel != c:
            x = ConvBlock(self.out_channel, 1, self.strides, dtype=self.dtype, use_norm=True,  # True
                          use_act=False)(x)

        return x + hidden


class GlobalAveragePooling(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = jnp.mean(x, [1, 2])
        return x


class ResNet(nn.Module):
    out_channels: Sequence = (64, 256, 1024, 2048)
    num_res_blocks: Sequence = (3, 3, 5, 2)
    down_samples: Sequence = (False, True, True, True)
    dtype: str = 'bfloat16'

    # down_samples: Sequence = (False, False, False, True)

    @nn.compact
    def __call__(self, x, *args, **kwargs):

        # ResNet Stem
        x = nn.Conv(64, (7, 7), (2, 2), 'same')(x)
        # x = nn.Conv(64, (3, 3), 2, 'same')(x)
        x = nn.max_pool(x, (3, 3), (2, 2), 'same')

        for out_channel, num_res_block, down_sample in zip(self.out_channels, self.num_res_blocks, self.down_samples):
            if down_sample:
                # x = ResBlock(out_channel=out_channel, strides=2, dtype='bfloat16')(x)
                x = ConvBlock(out_channel, 1, 2, True, dtype=self.dtype)(x)

            for _ in range(num_res_block):
                x = ResBlock(out_channel=out_channel, dtype='bfloat16')(x)

        x = GlobalAveragePooling()(x)
        x = nn.Dense(1000, dtype='float32')(x)
        # x = nn.softmax(x,axis=1)
        return x
