from functools import partial
from flax.linen.initializers import constant
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from typing import Any, Sequence


class RepVGGBlock(nn.Module):
    channels: int = 1
    strides: int = 1
    norm: Any = None  # partial(nn.BatchNorm, use_running_average=not True, )
    deploy: bool = False
    dtype: Any = jnp.bfloat16

    def setup(self) -> None:
        self.conv_3x3 = nn.Conv(self.channels, (3, 3), (self.strides, self.strides), padding='valid', name='conv_3x3',use_bias=False,
                                dtype=self.dtype)
        self.conv_1x1 = nn.Conv(self.channels, (1, 1), (self.strides, self.strides), padding='same', name='conv_1x1',use_bias=False,
                                dtype=self.dtype)
        self.bn_3x3 = self.norm(name='bn_3x3', )
        self.bn_1x1 = self.norm(name='bn_1x1', )
        # self.identity_bn = self.norm(name='identity_bn') if self.strides == 1 else None

        self.conv_deploy = nn.Conv(self.channels, (3, 3), (self.strides, self.strides), padding='valid',
                                   name='conv_deploy', dtype=self.dtype)

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        padding = ((0, 0), (1, 1), (1, 1), (0, 0))
        if self.deploy:
            x = jnp.pad(x, padding, constant_values=0)
            x = self.conv_deploy(x)
        else:
            b, h, w, c = x.shape
            identity_out = self.norm(name='identity_bn')(x) if self.strides == 1 and c == self.channels else 0
            x = self.bn_3x3(self.conv_3x3(jnp.pad(x, padding, constant_values=0)), ) + self.bn_1x1(
                self.conv_1x1(x)) + identity_out

        return  nn.relu(x)


class RepVGG(nn.Module):
    deploy: bool = False
    out_channels: Sequence[int] = (64, 128, 256, 512)
    width_multiplier: Sequence[int] = (0.75, 0.75, 0.75, 2.5)
    num_blocks: Sequence[int] = (2,)  # (2, 4, 14, 1)
    num_classes: int = 1000
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x, train=True, *args, **kwargs):
        norm = partial(
            nn.BatchNorm,
            use_running_average=not train,
            momentum=0.9,
            epsilon=1e-5,
            dtype=self.dtype,
            axis_name='batch',
        )

        x = RepVGGBlock(channels=min(64, int(64 * self.width_multiplier[0])), strides=2, norm=norm, deploy=self.deploy,
                        dtype=self.dtype)(x)

        for width_mul, num_blocks, out_channel in zip(self.width_multiplier, self.num_blocks, self.out_channels):
            out_channels = int(out_channel * width_mul)
            x = RepVGGBlock(channels=out_channels, strides=2, norm=norm, deploy=self.deploy, dtype=self.dtype)(x)
            for _ in range(num_blocks - 1):
                x = RepVGGBlock(channels=out_channels, strides=1, norm=norm, deploy=self.deploy, dtype=self.dtype)(x)

        x = jnp.mean(x, axis=[1, 2])
        x = nn.Dense(self.num_classes, name='classifier', dtype=self.dtype)(x)
        return x


RepVGG_A0 = partial(RepVGG, num_blocks=[2, 4, 14, 1], width_multiplier=[0.75, 0.75, 0.75, 2.5])
RepVGG_A1 = partial(RepVGG, num_blocks=[2, 4, 14, 1], width_multiplier=[1, 1, 1, 2.5])
RepVGG_A2 = partial(RepVGG, num_blocks=[2, 4, 14, 1], width_multiplier=[1.5, 1.5, 1.5, 2.75])
