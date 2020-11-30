from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
from typing import *
import multiprocessing as mp
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from plant_analysis.models.species.model import SpeciesClassifier
from plant_analysis.models.species.data import SpeciesInferenceDataset
from tensorflow.keras.models import Model


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
    parser.add_argument ('--model-path', type=Path, required=False, help='Path to the model')
    return parser.parse_args()


def get_model(model_path) -> tfk.Model:
    assert tfk.backend.image_data_format() == 'channels_last'
    model = SpeciesClassifier.load_from_dir(model_path)
    def call(inputs, training=None, mask=None):
        return model.backbone(inputs)
    model.call = call
    return model


def compute_embeddings(model: tfk.Model, image_paths: List[Path]):
    samples = SpeciesInferenceDataset(image_paths)
    dataset = (
        samples.to_tf_dataset(2048)
        .batch(128)
        .prefetch(3)
    )    
    embeddings = model.predict(dataset, verbose=1)
    return embeddings


def main():
    mp.set_start_method('spawn')
    args = parse_arguments()
    images_dir = args.images_dir

    model = get_model(args.model_path)

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
