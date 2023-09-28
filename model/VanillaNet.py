import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Sequence
from functools import partial


class Block(nn.Module):
    dim: int
    norm: Any
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        y=nn.Conv(self.dim, (1, 1), dtype=self.dtype)(nn.relu(x))
        x = nn.Conv(self.dim, (1, 1), dtype=self.dtype)(x)
        x = self.norm()(x)
        x = nn.Conv(self.dim, (1, 1), dtype=self.dtype)(x)
        return x+y


class VanillaNet(nn.Module):
    dims: Sequence[int]
    num_blocks: Sequence[int]
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

        x = nn.Conv(512, (4, 4), (4, 4), dtype=self.dtype)(x)

        for i, (dim, num_block) in enumerate(zip(self.dims, self.num_blocks)):
            for _ in range(num_block):
                x = Block(dim, norm, dtype=self.dtype)(x)

            if i != 3:
                x = nn.max_pool(x, (2, 2), (2, 2))

        x = jnp.mean(x, [1, 2])
        x = nn.Dense(self.num_classes, dtype=self.dtype)(x)
        return x


VanillaNet5 = partial(VanillaNet, dims=[1024, 2048, 4096], num_blocks=[1, 1, 1])
VanillaNet6 = partial(VanillaNet, dims=[1024, 2048, 4096, 4096], num_blocks=[1, 1, 1, 1])
VanillaNet7 = partial(VanillaNet, dims=[1024, 2048, 4096, 4096], num_blocks=[2, 1, 1, 1])
VanillaNet8 = partial(VanillaNet, dims=[1024, 2048, 4096, 4096], num_blocks=[2, 1, 2, 1])
VanillaNet9 = partial(VanillaNet, dims=[1024, 2048, 4096, 4096], num_blocks=[2, 1, 3, 1])
VanillaNet10 = partial(VanillaNet, dims=[1024, 2048, 4096, 4096], num_blocks=[2, 1, 4, 1])
VanillaNet11 = partial(VanillaNet, dims=[1024, 2048, 4096, 4096], num_blocks=[2, 1, 5, 1])
VanillaNet12 = partial(VanillaNet, dims=[1024, 2048, 4096, 4096], num_blocks=[2, 1, 6, 1])
VanillaNet13 = partial(VanillaNet, dims=[1024, 2048, 4096, 4096], num_blocks=[2, 1, 7, 1])
