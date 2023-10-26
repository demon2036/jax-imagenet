import io
import os

import numpy
import numpy as np
import torch
import webdataset
import webdataset as wds

from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import albumentations as A
import jax
import jax.numpy as jnp
from temp import get_dl


def collect_fn(x):
    #print(x)
    # print(x)
    return x


class MyDataSet(Dataset):
    def __init__(self, cache_data):
        self.cache_data = cache_data

        self.transform = A.Resize(256, 256)

    def __len__(self):
        return len(self.cache_data)

    def _preprocess(self, x):
        x = Image.open(io.BytesIO(x['jpg'])).convert('RGB')
        x = np.array(x)
        x = self.transform(image=x)['image']
        return x

    def __getitem__(self, index):
        x = self._preprocess(self.cache_data[index])
        return x





if __name__ == "__main__":
    urls = 'pipe:gcloud alpha storage cat gs://luck-eu/data/imagenet_train_shards/imagenet_train_shards-{00000..00073}.tar '

    # urls = 'pipe: cat /home/john/data/imagenet_train_shards/imagenet_train_shards-{00000..00073}.tar'

    dl = get_dl()
    # dl = map(prepare_tf_data,dl, )

    for _ in range(100):
        for data in tqdm(dl):
            #print(data)
            pass

        # print(data)
        # print(data)

    """
    temp = []

    for data in tqdm(dl):
        temp.extend(data)

    print(len(temp))

    dataset = MyDataSet(temp)

    dl = DataLoader(dataset, num_workers=jax.device_count() * 2, prefetch_factor=16, batch_size=1024)

    for data in tqdm(dl):
        # print(data.shape)
        pass
    """
