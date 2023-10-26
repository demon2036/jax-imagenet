from input_pipeline import create_split
import tensorflow_datasets as tfds

if __name__ == "__main__":
    data_path = '/home/john/tensorflow_datasets'

    dataset_builder = tfds.builder('imagenet2012', try_gcs=False,
                                   data_dir=data_path)
    ds_eval = create_split(dataset_builder, batch_size=64, train=False, cache=False)

    for batch in ds_eval:
        print(batch)
