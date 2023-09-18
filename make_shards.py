from pathlib import Path
from tqdm import tqdm
import webdataset as wds

def create_shards(dataset_root='/home/john/data/imagenet_raw/train'):
    # dataset=ImageFolder()
    file_paths = [path for path in Path(dataset_root).glob('*/*')]
    category_list = sorted([path.name for path in Path(dataset_root).glob('*') if path.is_dir()])
    category_index = {category_name: i for i, category_name in enumerate(category_list)}
    print(category_index)
    shard_path = 'imagenet_train_shards'
    shard_dir_path = Path(shard_path)
    shard_dir_path.mkdir(exist_ok=True)
    shard_filename = str(shard_dir_path / 'imagenet_train_shards-%05d.tar')
    shard_size = int(1 * 1000 ** 3)
    with wds.ShardWriter(shard_filename, maxsize=shard_size) as sink, tqdm(file_paths) as pbar:
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
