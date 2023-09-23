import tensorflow_datasets as tfds
import tensorflow as tf
import jax

IMAGE_SIZE = 224
CROP_PADDING = 32
MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]


def test(x):
    print(x)


def temp(train=True):
    dataset_builder = tfds.builder('imagenet2012')
    if train:
        train_examples = dataset_builder.info.splits['train'].num_examples
        split_size = train_examples // jax.process_count()
        start = jax.process_index() * split_size
        split = f'train[{start}:{start + split_size}]'

    else:
        validate_examples = dataset_builder.info.splits['validation'].num_examples
        split_size = validate_examples // jax.process_count()
        start = jax.process_index() * split_size
        split = f'validation[{start}:{start + split_size}]'
    print(split)
    ds = dataset_builder.as_dataset(
        split=split,
        decoders={
            'image': tfds.decode.SkipDecoding(),
        },
    )
    ds.map()
    #ds.map()



if __name__ == "__main__":
    # tfds.builder('imagenet2012').download_and_prepare(
    #     download_config=tfds.download.DownloadConfig(
    #         manual_dir='/home/john/data/test'))
    #dataset_builder = tfds.builder('imagenet2012')
    #print(type(dataset_builder))
    temp()



    # options = tf.data.Options()
    # ds = dataset.with_options(options)
