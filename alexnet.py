import einops
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any


class AlexNet(nn.Module):
    num_classes: int = 1000
    dtype: Any = 'bfloat16'

    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv(64, kernel_size=(11, 11), stride=(4, 4), padding='SAME', dtype=self.dtype),
            nn.relu(),
            nn.max_pool(window_shape=(3, 3), strides=(2, 2), padding='SAME'),
            nn.Conv(192, kernel_size=(5, 5), padding="SAME", dtype=self.dtype),
            nn.relu(),
            nn.max_pool(window_shape=(3, 3), strides=(2, 2), padding='SAME'),
            nn.Conv(384, kernel_size=(3, 3), padding="SAME", dtype=self.dtype),
            nn.relu(),
            nn.Conv(256, kernel_size=(3, 3), padding="SAME", dtype=self.dtype),
            nn.relu(),
            nn.Conv(256, kernel_size=(3, 3), padding="SAME", dtype=self.dtype),
            nn.relu(),
            nn.max_pool(window_shape=(3, 3), strides=(2, 2), padding='SAME'),
        )
        self.avg_pool = nn.avg_pool(window_shape=(6, 6))
        self.classifier = nn.Sequential(
            # nn.Dropout(p=dropout),
            nn.Dense(4096, dtype=self.dtype),
            nn.relu(),
            # nn.Dropout(p=dropout),
            nn.Dense(4096, dtype=self.dtype),
            nn.relu(),
            nn.Dense(num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = einops.rearrange(x, 'b h w c ->b (h w c)')
        # x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
