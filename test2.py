from input_pipeline import create_split
import tensorflow_datasets as tfds
from torchvision.utils import save_image
import einops
import torch
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    dataset_builder = tfds.builder('imagenet2012', )  # try_gcs=True,data_dir='gs://jtitor-eu/data/tensorflow_datasets'
    ds = create_split(dataset_builder, batch_size=64, train=True, cache=True)

    for data in tqdm(ds):
        x, y = data['image'], data['label']
        print(x.shape)
        # x=np.array(x)
        # x=torch.Tensor(x)
        # x=einops.rearrange(x,'b h w c->b c h w')
        # save_image(x,'test.png')
        # break
