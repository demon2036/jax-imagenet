import flax.linen as nn
import jax
import jax.numpy as jnp


class VGG(nn.Module):
    @nn.compact
    def __call__(self, x, *args, **kwargs):
        pass
