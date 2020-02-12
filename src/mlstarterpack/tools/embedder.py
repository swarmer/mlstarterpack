from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
from typing import *

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
try:
    import tensorflow as tf
    from tensorflow import keras as tfk
except ImportError as exc:
    raise ImportError('This module requires tensorflow extra feature') from exc

from mlstarterpack.vision.images import find_images, load_img


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('images_dir', type=str, help='Path to a directory with images to check')
    parser.add_argument(
        '--data-root', type=Path, required=False, default=Path('data/'),
        help='Path to the data root directory',
    )
    parser.add_argument('--output', type=str, help='Path to the output parquet file')
    return parser.parse_args()


def get_model() -> tfk.Model:
    assert tfk.backend.image_data_format() == 'channels_last'
    model = tfk.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        pooling='avg',
    )
    return model


def compute_embeddings(model: tfk.Model, image_paths: List[Path]):
    def generator_factory():
        return (
            np.array(load_img(path), np.float32)
            for path in image_paths
        )
    dataset = (
        tf.data.Dataset.from_generator(generator_factory, tf.float32)
        .map(partial(tf.image.resize_with_crop_or_pad, target_height=224, target_width=224))
        .map(tfk.applications.resnet50.preprocess_input)
        .batch(32)
        .prefetch(3)
    )
    embeddings = model.predict(dataset, verbose=1)
    return embeddings


def main():
    args = parse_arguments()
    images_dir = args.images_dir

    model = get_model()

    image_paths = list(find_images(images_dir))
    embeddings = compute_embeddings(model, image_paths)

    df = pd.DataFrame({
        'image_paths': [str(path.relative_to(args.data_root)) for path in image_paths],
        'embeddings': list(embeddings),
    })
    table = pa.Table.from_pandas(df)
    pq.write_table(table, args.output)


if __name__ == '__main__':
    main()
