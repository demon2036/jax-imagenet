import json
from pathlib import Path
import cv2
import jax
import tqdm
import webdataset as wds
import albumentations as A
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm


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


def cycle(dataset):
    for data in dataset:
        yield data






#imagenet_train_shards-{00000..00012}.tar
class MyWebDataSet(Dataset):
    def __init__(self, dataset_root='./imagenet_train_shards', shuffle_size=10000, cache=False, web_dataset=None):
        self.preprocessor = ImagePreprocessor()
        self.dataset_root = dataset_root
        self.dataset_size = 1281167  # self.info_from_json()
        self.cache = cache

        # self.web_dataset = wds.WebDataset(shards_urls, shardshuffle=True).shuffle(shuffle_size).decode(
        # 'rgb').to_tuple( 'jpg', 'cls') self.web_dataset = wds.WebDataset(shards_urls, shardshuffle=True,
        # ).shuffle(shuffle_size).decode( 'rgb')  # .to_tuple( # 'jpg', 'cls')
        #
        # self.web_dataset = cycle(self.web_dataset)
        self.web_dataset = web_dataset

        self.cached_data = []

    def info_from_json(self):
        with open(Path(self.dataset_root) / 'dataset-size.json', 'r') as f:
            info_dic = json.load(f)
        return info_dic['dataset_size']

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):

        if len(self.cached_data) == self.dataset_size:
            # return self._preprocess(self.cached_data[idx])
            return self.cached_data[idx]
        else:
            data = next(self.web_dataset)
            preprocess_img = self._preprocess(data['jpg'])
            if self.cache:
                self.cached_data.append(preprocess_img)
            # if self.cache:
            #     self.cached_data.append(data['jpg'])
            return preprocess_img

    def _preprocess(self, img):
        return self.preprocessor.preprocess(img)


def create_input_pipeline(dataset_root='./imagenet_train_shards', batch_size=128, num_workers=8, pin_memory=True,
                          drop_last=True, shuffle_size=10000):
    # print(shards_urls)
    shards_urls = [str(path) for path in Path(dataset_root).glob('*.tar')]
    web_dataset = wds.WebDataset(shards_urls, shardshuffle=True, ).shuffle(shuffle_size).decode(
        'rgb')  # .to_tuple(
    # 'jpg', 'cls')

    web_dataset = cycle(web_dataset)

    dataset = MyWebDataSet(dataset_root, cache=True, web_dataset=web_dataset)

    # for x in dataset:
    #     print(x)

    dl = DataLoader(dataset, num_workers=12, batch_size=batch_size, pin_memory=False, drop_last=False,
                    persistent_workers=False)
    return dl
    # for sample in tqdm.tqdm(dl, total=10000):
    #     data, cls = sample
    # print(data.shape, cls)


if __name__ == '__main__':

    # dl = create_input_pipeline(dataset_root='/home/john/data/imagenet_train_shards', batch_size=32)
    # dl = create_input_pipeline(dataset_root='/home/john/data/ffhq_shards', batch_size=32, num_workers=8)
    dl = create_input_pipeline(dataset_root='/root/fused_bucket/data/imagenet_train_shards', batch_size=1024,
                               num_workers=0)
    for _ in range(10):
        for datas in tqdm(dl, total=10000):
            # print(datas.shape)
            pass
        print(1)
        # break
        # x, y = datas
        # print(x.min(), x.max())
        # print(x.shape, y.shape)

    # create_shards()

# dataset=wds.WebDataset(shardshuffle=True).shuffle(1000).decode('rgb')
