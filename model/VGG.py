import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
from torchvision.models import VisionTransformer
from typing import Any
from functools import partial


class MaxPool2D(nn.Module):

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        return nn.max_pool(x, (2, 2), (2, 2))


class VGG(nn.Module):
    num_classes: int = 1000
    cfg: Any = None
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        x = make_features(self.cfg, self.dtype)(x)
        x = einops.rearrange(x, 'b h w c-> b (h w c)')

        x = nn.Sequential([
            nn.Dense(4096, dtype=self.dtype),
            nn.relu,
            nn.Dense(4096, dtype=self.dtype),
            nn.relu,
            nn.Dense(self.num_classes, dtype=self.dtype)
        ])(x)

        return x

        pass


cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def make_features(cfg, dtype):
    layer = []
    for v in cfg:
        if v == 'M':
            layer.append(MaxPool2D())
        else:
            layer.append(nn.Conv(v, (3, 3), padding='same', dtype=dtype))

    return nn.Sequential(layer)


VGG11 = partial(VGG, cfg=cfgs['vgg11'])
VGG13 = partial(VGG, cfg=cfgs['vgg13'])
VGG16 = partial(VGG, cfg=cfgs['vgg16'])
VGG19 = partial(VGG, cfg=cfgs['vgg19'])
