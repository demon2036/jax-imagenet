from data.datasets import create_input_pipeline
import jax
import tensorflow_datasets as tfds

from input_pipeline import create_split
from modules.utils import create_checkpoint_manager


class Trainer:
    def __init__(self,
                 image_size=224,
                 batch_size=1,
                 data_path='./imagenet_train_shards',
                 num_workers=8,
                 shuffle_size=10000,
                 drop_last=True,
                 seed=43,
                 total_epoch=90,
                 model_path='check_points/Diffusion',
                 ckpt_max_to_keep=5
                 ):
        dataset_builder = tfds.builder('imagenet2012', try_gcs=True,
                                       data_dir='gs://jtitor-eu/data/tensorflow_datasets')  # try_gcs=True,data_dir='gs://jtitor-eu/data/tensorflow_datasets'
        self.dl = create_split(dataset_builder, batch_size=batch_size, train=True, cache=True)

        #self.dl = create_input_pipeline(batch_size=batch_size, num_workers=num_workers, dataset_root=data_path,drop_last=drop_last,shuffle_size=shuffle_size)
        self.rng = jax.random.PRNGKey(seed)
        self.total_epoch = total_epoch
        self.model_path = model_path
        self.checkpoint_manager = create_checkpoint_manager(model_path, max_to_keep=ckpt_max_to_keep)
        self.finished_steps = 0

