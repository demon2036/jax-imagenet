import json
from pathlib import Path
from tqdm import tqdm
import webdataset as wds
from random import shuffle


def create_shards(shard_path='imagenet_train_shards', dataset_root='/home/john/data/imagenet_raw/train',
                  shard_size=int(1 * 1000 ** 3), use_shuffle=False):
    # dataset=ImageFolder()
    file_paths = [path for path in Path(dataset_root).glob('*/*')]
    print(len(file_paths))
    if use_shuffle:
        for _ in range(10):
            shuffle(file_paths)

    category_list = sorted([path.name for path in Path(dataset_root).glob('*') if path.is_dir()])
    category_index = {category_name: i for i, category_name in enumerate(category_list)}
    print(category_index)

    shard_dir_path = Path(shard_path)
    shard_dir_path.mkdir(exist_ok=True)
    shard_filename = str(shard_dir_path / 'imagenet_train_shards-%05d.tar')
    # maxsize=shard_size
    with wds.ShardWriter(shard_filename, maxcount=10000, maxsize=3e10) as sink, tqdm(file_paths) as pbar:
        for file_path in pbar:
            category_name = file_path.parent.name
            label = category_index[category_name]
            key_str = category_name + '/' + file_path.stem
            with open(file_path, 'rb') as stream:
                img = stream.read()

            sink.write({
                "__key__": key_str,
                "jpg": img,
                "cls": label
            })


def create_shards_no_label(shard_path='imagenet_train_shards', dataset_root='/home/john/data/imagenet_raw/train',
                           shard_size=int(1 * 1000 ** 3), use_shuffle=False):
    # dataset=ImageFolder()
    file_paths = [path for path in Path(dataset_root).glob('*')]

    if use_shuffle:
        for _ in range(10):
            shuffle(file_paths)

    shard_dir_path = Path(shard_path)
    shard_dir_path.mkdir(exist_ok=True)
    shard_filename = str(shard_dir_path / 'data_shards-%05d.tar')
    dataset_size = 0
    # maxsize=shard_size
    with wds.ShardWriter(shard_filename, maxcount=10000, maxsize=3e9) as sink, tqdm(file_paths) as pbar:
        for file_path in pbar:
            dataset_size += 1
            category_name = file_path.parent.name

            key_str = category_name + '/' + file_path.stem
            with open(file_path, 'rb') as stream:
                img = stream.read()

            sink.write({
                "__key__": key_str,
                "jpg": img,
            })
    print(f'dataset_size:{dataset_size}')
    with open(shard_dir_path / 'dataset-size.json', 'w') as f:
        json.dump({
            "dataset_size": dataset_size,
        }, f)


if __name__ == "__main__":
    create_shards('/home/john/data/imagenet_train_shards', dataset_root='/home/john/imagenet/train',
                  shard_size=int(200 * 1000 ** 2), use_shuffle=True)

    create_shards_no_label('/home/john/data/ffhq_shards', dataset_root='/home/john/imagenet/train',
                           shard_size=int(200 * 1000 ** 2), use_shuffle=True)
