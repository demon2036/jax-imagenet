import flax.jax_utils
from tqdm import tqdm

from data.datasets import create_input_pipeline
import jax
import tensorflow_datasets as tfds

from input_pipeline import create_split
from modules.utils import create_checkpoint_manager


def prepare_tf_data(xs):
    """Convert a input batch from tf Tensors to numpy arrays."""
    local_device_count = jax.local_device_count()

    def _prepare(x):
        # Use _numpy() for zero-copy conversion between TF and NumPy.
        x = x#._numpy()  # pylint: disable=protected-access

        # reshape (host_batch_size, height, width, 3) to
        # (local_devices, device_batch_size, height, width, 3)
        return x.reshape((local_device_count, -1) + x.shape[1:])

    return jax.tree_util.tree_map(_prepare, xs)


class Trainer:
    def __init__(self,
                 image_size=224,
                 batch_size=1,
                 data_path='./imagenet_train_shards',
                 num_workers=8,
                 shuffle_size=250000,
                 drop_last=True,
                 warmup_epoch=5,
                 try_gcs=False,
                 seed=43,
                 total_epoch=90,
                 model_path='check_points/Diffusion',
                 ckpt_max_to_keep=5,
                 cut_mix=False,
                 cache=True
                 ):
        # 'gs://jtitor-eu/data/tensorflow_datasets'
        dataset_builder = tfds.builder('imagenet2012', try_gcs=try_gcs,
                                       data_dir=data_path)  # try_gcs=True,data_dir='gs://jtitor-eu/data/tensorflow_datasets'
        # ds_train = create_split(dataset_builder, batch_size=batch_size, train=True, cache=cache, cutmix=cut_mix,shuffle_buffer_size=shuffle_size)
        # ds_eval = create_split(dataset_builder, batch_size=batch_size, train=False, cache=cache)

        from data.datasets import create_input_pipeline

        ds_train = create_input_pipeline(dataset_builder, batch_size=batch_size, )
        ds_eval = create_input_pipeline(dataset_builder, batch_size=batch_size, )

        self.dl = map(prepare_tf_data, ds_train)
        self.dl = flax.jax_utils.prefetch_to_device(self.dl, 2)
        self.dl_eval = map(prepare_tf_data, ds_eval)

        num_validation_examples = dataset_builder.info.splits[
            'validation'
        ].num_examples
        self.steps_per_eval = num_validation_examples // batch_size

        self.rng = jax.random.PRNGKey(seed)
        self.total_epoch = total_epoch
        self.model_path = model_path
        self.checkpoint_manager = create_checkpoint_manager(model_path, max_to_keep=ckpt_max_to_keep)
        self.finished_steps = 0
        self.steps_per_epoch = dataset_builder.info.splits['train'].num_examples // batch_size
        self.total_steps = self.steps_per_epoch * self.total_epoch
        self.warmup_epoch = warmup_epoch
        self.pbar = tqdm(total=self.total_epoch * self.steps_per_epoch)
