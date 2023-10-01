import einops
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any

import optax



# from torchvision.models import AlexNet

class AlexNet(nn.Module):
    num_classes: int = 1000
    dtype: Any = 'bfloat16'

    def setup(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        self.classifier = nn.Sequential([
            # nn.Dropout(p=dropout),
            nn.Dense(4096, dtype=self.dtype),
            nn.relu,
            # nn.Dropout(p=dropout),
            nn.Dense(4096, dtype=self.dtype),
            nn.relu,
            nn.Dense(num_classes),
        ]
        )

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        x = nn.Conv(64, kernel_size=(11, 11), strides=(4, 4), padding='SAME', dtype=self.dtype)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding='SAME')
        x = nn.Conv(192, kernel_size=(5, 5), padding="SAME", dtype=self.dtype)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding='SAME')
        x = nn.Conv(384, kernel_size=(3, 3), padding="SAME", dtype=self.dtype)(x)
        x = nn.relu(x)
        x = nn.Conv(256, kernel_size=(3, 3), padding="SAME", dtype=self.dtype)(x)
        x = nn.relu(x)
        x = nn.Conv(256, kernel_size=(3, 3), padding="SAME", dtype=self.dtype)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding='SAME')

        # x = self.features(x)
        x = nn.avg_pool(x, window_shape=(6, 6))
        x = einops.rearrange(x, 'b h w c ->b (h w c)')
        # x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
