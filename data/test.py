import io
import os

import einops
import numpy
import numpy as np
import torch

from torch.nn.functional import scaled_dot_product_attention
import torchvision.transforms
import webdataset
import webdataset as wds
from timm.data import RandAugment, create_transform
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import albumentations as A
import jax
import jax.numpy as jnp

MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]

mean = jnp.array(MEAN_RGB, dtype=np.float32).reshape(1, 1, 3)
std = jnp.array(STDDEV_RGB, dtype=np.float32).reshape(1, 1, 3)

# mean = np.array(MEAN_RGB,dtype=np.float32).reshape(1, 1, 3)
# std = np.array(STDDEV_RGB,dtype=np.float32).reshape(1, 1, 3)
# mean = torch.Tensor(MEAN_RGB).reshape(1, 1, 3)
# std = torch.Tensor(STDDEV_RGB).reshape(1, 1, 3)


MEAN_RGB = [0.485, 0.456, 0.406]
STDDEV_RGB = [0.229, 0.224, 0.225]


def test(x):
    # print(x['cls'])
    # cls = int(x['cls'].decode('utf-8'))
    # x = Image.open(io.BytesIO(x['jpg'])).convert('RGB')
    cls = x['cls']
    cls = torch.nn.functional.one_hot(torch.Tensor(np.array(cls).reshape(-1)).to(torch.int64),
                                      1000).float().reshape(-1)
    x = x['jpg']

    x = np.asarray(x)
    x = A.OneOf([A.Resize(256, 256), A.Resize(480, 480)],p=1)(image=x)['image']
    x = A.RandomCrop(224, 224)(image=x)['image']
    x = A.HorizontalFlip()(image=x)['image']

    # x= x/255.0
    # x = x / 255.0

    return {'image': x, 'label': cls}


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
        x = numpy.asarray(x)
        return x.reshape((local_device_count, -1) + x.shape[1:])

    xs = jax.tree_util.tree_map(_prepare, xs)

    xs['image'] = jax.pmap(normalize)(xs['image'])

    return xs


def create_input_pipeline(*args, **kwargs):
    urls = 'pipe:gcloud alpha storage cat gs://luck-eu/data/imagenet_train_shards/imagenet_train_shards-{00073..00073}.tar '
    urls = 'pipe:gcloud alpha storage cat gs://luck-eu/data/imagenet_train_shards/imagenet_train_shards-{00000..00073}.tar '
    # urls = 'gs://luck-eu/data/imagenet_train_shards/imagenet_train_shards-{00000..00073}.tar '

    # urls = 'pipe: cat /home/john/data/imagenet_train_shards/imagenet_train_shards-{00073..00073}.tar'

    dataset = wds.WebDataset(
        urls=urls,
        shardshuffle=True).mcached().shuffle(20000).decode('pil').map(
        test)  # .batched(1024,collation_fn=default_collate).map(temp)

    dataloader = DataLoader(dataset, num_workers=48, prefetch_factor=2, batch_size=1024, drop_last=True,
                            persistent_workers=True)

    while True:
        for xs in dataloader:
            del xs['__key__']
            yield xs


if __name__ == "__main__":
    dl = create_input_pipeline()
    dl = map(prepare_torch_data, dl)
    data = next(dl)
    print(data['image'], )

    print(data['label'].shape)

    images = np.asarray(data['image'])
    images = torch.Tensor(images)

    from torchvision.utils import save_image

    images = einops.rearrange(images, 'n b h w c->(n b ) c h w  ')
    print(images.shape)

    save_image(images, 'test.png')
