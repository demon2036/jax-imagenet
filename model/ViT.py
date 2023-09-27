import jax
import jax.numpy as jnp
import flax.linen as nn
from torchvision.models import MaxVit


class MultiHeadAttention(nn.Module):
    @nn.compact
    def __call__(self, x, *args, **kwargs):
        pass


class Transformer(nn.Module):

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        pass


class ViT(nn.Module):
    @nn.compact
    def __call__(self, x, *args, **kwargs):
        pass
