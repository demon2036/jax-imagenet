import io
import os

import einops
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

MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]

mean = jnp.array(MEAN_RGB).reshape(1, 1, 3)
std = jnp.array(STDDEV_RGB).reshape(1, 1, 3)

# mean = torch.Tensor(MEAN_RGB).reshape(1, 1, 3)
# std = torch.Tensor(STDDEV_RGB).reshape(1, 1, 3)


def test(x):
    cls = int(x['cls'].decode('utf-8'))
    x = Image.open(io.BytesIO(x['jpg'])).convert('RGB')
    x = np.array(x)
    x = A.HorizontalFlip()(image=x)['image']
    x = A.Resize(224, 224)(image=x)['image']

    # x = x / 255.0

    return {'images': x, 'labels': torch.nn.functional.one_hot(torch.Tensor(np.array(cls).reshape(-1)).to(torch.int64),
                                                               1000).float().reshape(-1)}


def normalize(images):
    # images = images.float()
    # print(images.dtype)
    images -= mean
    images /= std
    return images


def prepare_torch_data(xs):
    """Convert a input batch from tf Tensors to numpy arrays."""
    local_device_count = jax.local_device_count()


    def _prepare(x):
        # Use _numpy() for zero-copy conversion between TF and NumPy.
        # x = {'img': x['img'], 'cls': x['cls']}
        x = numpy.asarray(x)

        return x.reshape((local_device_count, -1) + x.shape[1:])

    xs = jax.tree_util.tree_map(_prepare, xs)

    xs['images'] = jax.pmap(normalize)(xs['images'])

    return xs


def create_input_pipeline(*args, **kwargs):
    urls = 'pipe:gcloud alpha storage cat gs://luck-eu/data/imagenet_train_shards/imagenet_train_shards-{00073..00073}.tar '
    urls = 'pipe:gcloud alpha storage cat gs://luck-eu/data/imagenet_train_shards/imagenet_train_shards-{00000..00073}.tar '

    # urls = 'pipe: cat /home/john/data/imagenet_train_shards/imagenet_train_shards-{00073..00073}.tar'


    dataset = wds.WebDataset(
        urls=urls,
        shardshuffle=False).mcached().map(test)  # .batched(1024,collation_fn=default_collate).map(temp)

    dataloader = DataLoader(dataset, num_workers=48, prefetch_factor=8, batch_size=1024, drop_last=True,
                            persistent_workers=True)

    while True:
        for xs in dataloader:
            del xs['__key__']
            yield xs


if __name__ == "__main__":
    dl = create_input_pipeline()
    dl = map(prepare_torch_data, dl)
    data = next(dl)
    print(data['images'], )
    # print(jnp.argmax(data['labels'],axis=-1))
    print(data['labels'].shape)

    images = np.asarray(data['images'])
    images = torch.Tensor(images)

    from torchvision.utils import save_image

    images = einops.rearrange(images, 'n b h w c->(n b ) c h w  ')
    print(images.shape)

    save_image(images, 'test.png')
