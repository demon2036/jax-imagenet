from pathlib import Path
import cv2
import jax
import tqdm
import webdataset as wds
import albumentations as A
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np


class ImagePreprocessor():
    def __init__(self, image_size=224):
        self.resize = A.Resize(image_size, image_size)
        self.center_crop = A.CenterCrop(image_size, image_size, p=0.5)
        self.random_horizontal_flip = A.HorizontalFlip()
        self.normalize = A.Normalize()

    def preprocess(self, img):
        # print(img.max(), img.min())
        img = self.resize(image=img)['image']
        # img = self.center_crop(image=img)['image']
        img = self.random_horizontal_flip(image=img)['image']

        # img=img/255.0
        img = self.normalize(image=img)['image']
        return img


def transfer_data(x, image_size=224):
    # x = A.center_crop(x, image_size, image_size)
    # x = A.resize(x, image_size, image_size, interpolation=cv2.INTER_CUBIC)

    preprocessor = ImagePreprocessor(image_size=image_size)
    x = preprocessor.preprocess(x)

    return x


def create_input_pipeline(dataset_root='./imagenet_train_shards', batch_size=128, num_workers=8, pin_memory=True,
                          drop_last=True, shuffle_size=10000):
    shards_urls = [str(path) for path in Path(dataset_root).glob('*')]

    # print(shards_urls)
    dataset = wds.WebDataset(shards_urls, shardshuffle=True).shuffle(shuffle_size).decode('rgb').to_tuple('jpg',
                                                                                                          'cls').map_tuple(
        transfer_data)
    dl = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, pin_memory=pin_memory, drop_last=drop_last)
    return dl
    # for sample in tqdm.tqdm(dl, total=10000):
    #     data, cls = sample
    # print(data.shape, cls)


if __name__ == '__main__':
    dl = create_input_pipeline(dataset_root='pipe:gsutil cat gs://somebucket/dataset-000.tar', batch_size=16)
    for data in dl:
        x, y = data
        print(x.shape, y.shape)
        break

    # create_shards()

# dataset=wds.WebDataset(shardshuffle=True).shuffle(1000).decode('rgb')
