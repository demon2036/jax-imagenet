# Copyright 2023 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ImageNet input pipeline."""
from collections import abc
import random

import jax
import keras_cv
import matplotlib
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_models

import autoaugment

IMAGE_SIZE = 224
CROP_PADDING = 32
MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]

# MEAN_RGB = [0, 0, 0]
# STDDEV_RGB = [1, 1, 1]
#
# MEAN_RGB = [0.5*255, 0.5*255, 0.5*255]
# STDDEV_RGB = [0.5*255, 0.5*255, 0.5*255]

def distorted_bounding_box_crop(
        image_bytes,
        bbox,
        min_object_covered=0.1,
        aspect_ratio_range=(0.75, 1.33),
        area_range=(0.05, 1.0),
        max_attempts=100,
):
    """Generates cropped_image using one of the bboxes randomly distorted.

  See `tf.image.sample_distorted_bounding_box` for more documentation.

  Args:
    image_bytes: `Tensor` of binary image data.
    bbox: `Tensor` of bounding boxes arranged `[1, num_boxes, coords]`
        where each coordinate is [0, 1) and the coordinates are arranged
        as `[ymin, xmin, ymax, xmax]`. If num_boxes is 0 then use the whole
        image.
    min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
        area of the image must contain at least this fraction of any bounding
        box supplied.
    aspect_ratio_range: An optional list of `float`s. The cropped area of the
        image must have an aspect ratio = width / height within this range.
    area_range: An optional list of `float`s. The cropped area of the image
        must contain a fraction of the supplied image within this range.
    max_attempts: An optional `int`. Number of attempts at generating a cropped
        region of the image of the specified constraints. After `max_attempts`
        failures, return the entire image.
  Returns:
    cropped image `Tensor`
  """
    shape = tf.io.extract_jpeg_shape(image_bytes)
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        shape,
        bounding_boxes=bbox,
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=True,
    )
    bbox_begin, bbox_size, _ = sample_distorted_bounding_box

    # Crop the image to the specified bounding box.
    offset_y, offset_x, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = tf.unstack(bbox_size)
    crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
    image = tf.io.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)

    return image


def _resize(image, image_size):
    return tf.image.resize(
        [image], [image_size, image_size], method=tf.image.ResizeMethod.BICUBIC
    )[0]


def _at_least_x_are_equal(a, b, x):
    """At least `x` of `a` and `b` `Tensors` are equal."""
    match = tf.equal(a, b)
    match = tf.cast(match, tf.int32)
    return tf.greater_equal(tf.reduce_sum(match), x)


def _decode_and_random_crop(image_bytes, image_size):
    """Make a random crop of image_size."""
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    image = distorted_bounding_box_crop(
        image_bytes,
        bbox,
        min_object_covered=0.1,
        aspect_ratio_range=(3.0 / 4, 4.0 / 3.0),
        area_range=(0.08, 1.0),
        max_attempts=10,
    )
    original_shape = tf.io.extract_jpeg_shape(image_bytes)
    bad = _at_least_x_are_equal(original_shape, tf.shape(image), 3)

    image = tf.cond(
        bad,
        lambda: _decode_and_center_crop(image_bytes, image_size),
        lambda: _resize(image, image_size),
    )

    return image


def _decode_and_center_crop(image_bytes, image_size):
    """Crops to center of image with padding then scales image_size."""
    shape = tf.io.extract_jpeg_shape(image_bytes)
    image_height = shape[0]
    image_width = shape[1]

    padded_center_crop_size = tf.cast(
        (
                (image_size / (image_size + CROP_PADDING))
                * tf.cast(tf.minimum(image_height, image_width), tf.float32)
        ),
        tf.int32,
    )

    offset_height = ((image_height - padded_center_crop_size) + 1) // 2
    offset_width = ((image_width - padded_center_crop_size) + 1) // 2
    crop_window = tf.stack([
        offset_height,
        offset_width,
        padded_center_crop_size,
        padded_center_crop_size,
    ])
    image = tf.io.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
    image = _resize(image, image_size)

    return image


def normalize_image(image):
    image -= tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=image.dtype)
    image /= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=image.dtype)
    return image


def preprocess_for_train(image_bytes, dtype=tf.float32, image_size=IMAGE_SIZE):
    """Preprocesses the given image for training.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    dtype: data type of the image.
    image_size: image size.

  Returns:
    A preprocessed image `Tensor`.
  """
    image = _decode_and_random_crop(image_bytes, image_size)
    image = tf.reshape(image, [image_size, image_size, 3])
    image = tf.image.random_flip_left_right(image)
    image = normalize_image(image)
    image = tf.image.convert_image_dtype(image, dtype=dtype)
    return image


def preprocess_for_eval(image_bytes, dtype=tf.float32, image_size=IMAGE_SIZE):
    """Preprocesses the given image for evaluation.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    dtype: data type of the image.
    image_size: image size.

  Returns:
    A preprocessed image `Tensor`.
  """
    # image=tf.io.decode_image(image_bytes)
    image = _decode_and_center_crop(image_bytes, image_size)
    image = tf.reshape(image, [image_size, image_size, 3])
    image = normalize_image(image)
    image = tf.image.convert_image_dtype(image, dtype=dtype)
    return image


def get_resize_small(images, smaller_size, method="area", antialias=False):
    """Resizes the smaller side to `smaller_size` keeping aspect ratio.

  Args:
    smaller_size: an integer, that represents a new size of the smaller side of
      an input image.
    method: the resize method. `area` is a meaningful, bwd-compat default.
    antialias: see tf.image.resize. Ideally set to True for all new configs.

  Returns:
    A function, that resizes an image and preserves its aspect ratio.

  Note:
    backwards-compat for "area"+antialias tested here:
    (internal link)
  """

    def _resize_small(image):  # pylint: disable=missing-docstring
        h, w = tf.shape(image)[0], tf.shape(image)[1]

        # Figure out the necessary h/w.
        ratio = (
                tf.cast(smaller_size, tf.float32) /
                tf.cast(tf.minimum(h, w), tf.float32))
        h = tf.cast(tf.round(tf.cast(h, tf.float32) * ratio), tf.int32)
        w = tf.cast(tf.round(tf.cast(w, tf.float32) * ratio), tf.int32)

        dtype = image.dtype
        image = tf.image.resize(image, (h, w), method=method, antialias=antialias)
        return tf.cast(image, dtype)

    return _resize_small(images)


def maybe_repeat(arg, n_reps):
    if not isinstance(arg, abc.Sequence):
        arg = (arg,) * n_reps
    return arg


def get_central_crop(images, crop_size):
    """Makes central crop of a given size.

  Args:
    crop_size: either an integer H, where H is both the height and width of the
      central crop, or a list or tuple [H, W] of integers, where H and W are
      height and width of the central crop respectively.

  Returns:
    A function, that applies central crop.
  """
    crop_size = maybe_repeat(crop_size, 2)

    def _crop(image):
        h, w = crop_size[0], crop_size[1]
        dy = (tf.shape(image)[0] - h) // 2
        dx = (tf.shape(image)[1] - w) // 2
        return tf.image.crop_to_bounding_box(image, dy, dx, h, w)

    return _crop(images)


def preprocess_for_eval_test(image_bytes, dtype=tf.float32, image_size=IMAGE_SIZE):
    """Preprocesses the given image for evaluation.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    dtype: data type of the image.
    image_size: image size.

  Returns:
    A preprocessed image `Tensor`.
  """
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)

    image = get_resize_small(image, 256)
    image = get_central_crop(image, 224)
    image = tf.cast(image, tf.uint8)
    image = normalize_image(image)
    image = tf.image.convert_image_dtype(image, dtype=dtype)
    return image


def preprocess_for_eval(image_bytes, dtype=tf.float32, image_size=IMAGE_SIZE):
    """Preprocesses the given image for evaluation.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    dtype: data type of the image.
    image_size: image size.

  Returns:
    A preprocessed image `Tensor`.
  """
    image = _decode_and_center_crop(image_bytes, image_size)
    image = tf.reshape(image, [image_size, image_size, 3])
    image = normalize_image(image)
    image = tf.image.convert_image_dtype(image, dtype=dtype)
    return image


def one_hot(sample):
    sample['label'] = tf.one_hot(tf.cast(sample['label'], tf.int32), 1000)
    return sample


def get_randaug(num_layers: int = 2, magnitude: int = 10):
    """Creates a function that applies RandAugment.

  RandAugment is from the paper https://arxiv.org/abs/1909.13719,

  Args:
    num_layers: Integer, the number of augmentation transformations to apply
      sequentially to an image. Represented as (N) in the paper. Usually best
      values will be in the range [1, 3].
    magnitude: Integer, shared magnitude across all augmentation operations.
      Represented as (M) in the paper. Usually best values are in the range
      [5, 30].

  Returns:
    a function that applies RandAugment.
  """

    def _randaug(image):
        return autoaugment.distort_image_with_randaugment(
            image, num_layers, magnitude)

    return _randaug


rand_augment = get_randaug()


def preprocess_for_train_test(image_bytes, dtype=tf.float32, image_size=IMAGE_SIZE):
    """Preprocesses the given image for training.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    dtype: data type of the image.
    image_size: image size.

  Returns:
    A preprocessed image `Tensor`.
  """
    image = _decode_and_random_crop(image_bytes, image_size)
    image = tf.reshape(image, [image_size, image_size, 3])

    image = tf.image.random_flip_left_right(image)
    image = tf.cast(image, tf.uint8)
    image = rand_augment(image)
    image = tf.cast(image, dtype)
    image = normalize_image(image)
    image = tf.image.convert_image_dtype(image, dtype=dtype)
    return image


def create_split(
        dataset_builder,
        batch_size,
        train,
        dtype=tf.bfloat16,
        image_size=IMAGE_SIZE,
        cache=False,
        shuffle_buffer_size=16 * 1024,  # 16 * 1024,
        prefetch=10,
        cutmix=False
):
    """Creates a split from the ImageNet dataset using TensorFlow Datasets.

  Args:
    dataset_builder: TFDS dataset builder for ImageNet.
    batch_size: the batch size returned by the data pipeline.
    train: Whether to load the train or evaluation split.
    dtype: data type of the image.
    image_size: The target size of the images.
    cache: Whether to cache the dataset.
    shuffle_buffer_size: Size of the shuffle buffer.
    prefetch: Number of items to prefetch in the dataset.
  Returns:
    A `tf.data.Dataset`.
  """
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

    def decode_example(example):
        if train:
            image = preprocess_for_train_test(example['image'], dtype, image_size)

        else:
            image = preprocess_for_eval(example['image'], dtype, image_size)
            example['label'] = tf.one_hot(example['labels'], 1000)
        return {'image': image, 'label': example['label']}

    ds = dataset_builder.as_dataset(
        split=split,
        decoders={
            'image': tfds.decode.SkipDecoding(),
        },
    )
    options = tf.data.Options()
    options.threading.private_threadpool_size = 96
    # print(options.threading.private_threadpool_size)
    ds = ds.with_options(options)

    if cache:
        ds = ds.cache()

    if train:
        ds = ds.repeat()
        ds = ds.shuffle(shuffle_buffer_size, seed=0)

    ds = ds.map(decode_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)

    if not train:
        ds = ds.repeat()

    cut_mix = tensorflow_models.vision.augment.MixupAndCutmix(num_classes=1000, prob=0.2, switch_prob=0.5,
                                                              mixup_alpha=0.8)

    def cut_mix_and_mix_up(samples):
        samples['image'], samples['label'] = cut_mix(samples['image'], samples['label'])
        return samples

    if train and cutmix:
        # ds = ds.unbatch().batch(batch_size//16, drop_remainder=True)
        ds = ds.map(cut_mix_and_mix_up, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # ds = ds.unbatch().batch(batch_size , drop_remainder=True)
    elif train:
        ds = ds.map(one_hot)

    ds = ds.prefetch(prefetch)

    return ds


if __name__ == "__main__":

    import keras
    import matplotlib.pyplot as plt

    matplotlib.use('TkAgg')

    dataset_builder = tfds.builder('imagenet2012', data_dir='/home/john/tensorflow_datasets')
    ds = create_split(dataset_builder, 64, True, cutmix=True)


    # def decode_example(example):
    #
    #     image = preprocess_for_train(example['image'], tf.float32, 224)
    #
    #     return {'images': image, 'labels': example['label']}
    #
    #
    # ds = dataset_builder.as_dataset(
    #     split=split,
    #     decoders={
    #         'image': tfds.decode.SkipDecoding(),
    #     },
    # )
    # options = tf.data.Options()
    # options.threading.private_threadpool_size = 96
    # # print(options.threading.private_threadpool_size)
    # ds = ds.with_options(options)
    # ds = ds.map(decode_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #
    # cutmix = tensorflow_models.vision.augment.MixupAndCutmix(num_classes=1000, prob=0.2, switch_prob=1.0)
    #
    #
    # def cut_mix_and_mix_up(samples):
    #     # samples['labels'] = tf.cast(samples['labels'], tf.float32)
    #     # samples = cut_mix(samples, training=True)
    #     samples['images'], samples['labels'] = cutmix(samples['images'], samples['labels'])
    #
    #     # samples = mix_up(samples, training=True)
    #     return samples
    #
    #
    # ds = ds.batch(64, drop_remainder=True)
    # ds = ds.map(cut_mix_and_mix_up, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # # ds = ds.batch(64, drop_remainder=True)

    def visualize_dataset(dataset, title):
        plt.figure(figsize=(20, 20)).suptitle(title, fontsize=18)
        for i, samples in enumerate(iter(dataset.take(64))):
            print(samples['labels'])
            # print(samples)
            images = samples["images"]
            plt.subplot(8, 8, i + 1)
            plt.imshow(images[0].numpy().astype("uint8"))
            plt.axis("off")
        plt.show()


    visualize_dataset(ds, title="After CutMix and MixUp")
    # visualize_dataset(ds, title="After CutMix and MixUp")
    # visualize_dataset(ds, title="After CutMix and MixUp")
