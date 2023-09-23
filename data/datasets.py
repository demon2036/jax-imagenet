from pathlib import Path
import cv2
import jax
import tqdm
import webdataset as wds
import albumentations as A
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader,Dataset
import numpy as np
from tqdm import tqdm

image_size = 224


class ImagePreprocessor():
    def __init__(self, image_size=224):
        self.resize = A.Resize(image_size, image_size)
        self.center_crop = A.CenterCrop(image_size, image_size, p=0.5)
        self.random_horizontal_flip = A.HorizontalFlip()
        self.normalize = A.Normalize()

    def preprocess(self, img):
        img = self.resize(image=img)['image']
        # img = self.center_crop(image=img)['image']
        img = self.random_horizontal_flip(image=img)['image']
        img = self.normalize(image=img)['image']
        return img


def transfer_data(x):
    # x = A.center_crop(x, image_size, image_size)
    # x = A.resize(x, image_size, image_size, interpolation=cv2.INTER_CUBIC)
    # A.normalize()
    preprocessor = ImagePreprocessor(image_size=image_size)
    x = preprocessor.preprocess(x)

    return x


class MyWebDataSet(Dataset):
    def __init__(self):
        self.preprocessor = ImagePreprocessor()

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


    def _preprocess(self, img):
        return self.preprocessor.preprocess(img)


def create_input_pipeline(dataset_root='./imagenet_train_shards', batch_size=128, num_workers=8, pin_memory=True,
                          drop_last=True, shuffle_size=10000):
    shards_urls = [str(path) for path in Path(dataset_root).glob('*')]
    # print(shards_urls)
    dataset = (wds.WebDataset(shards_urls, shardshuffle=True).shuffle(shuffle_size).decode('rgb').to_tuple('jpg',
                                                                                                           'cls').map_tuple(
        lambda img: transfer_data(img)
    ))

    dataset.with_length(1000)
    print(len(dataset))

    dl = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, pin_memory=pin_memory, drop_last=drop_last,
                    persistent_workers=True)
    return dl
    # for sample in tqdm.tqdm(dl, total=10000):
    #     data, cls = sample
    # print(data.shape, cls)


if __name__ == '__main__':

    dl = create_input_pipeline(dataset_root='/root/fused_bucket/data/imagenet_train_shards', batch_size=32)
    for data in tqdm(dl, total=10000):
        x, y = data
        print(x.min(), x.max())
        # print(x.shape, y.shape)

    # create_shards()

# dataset=wds.WebDataset(shardshuffle=True).shuffle(1000).decode('rgb')
