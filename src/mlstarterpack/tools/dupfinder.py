from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
try:
    import tensorflow as tf
except ImportError as exc:
    raise ImportError('This module requires tensorflow extra feature') from exc

from mlstarterpack.vision.images import find_images
from .dupbrowser import DuplicateBrowser, ImagePair
from .embedder import compute_embeddings, get_model


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--images-dir', type=str, required=False, default=None,
        help='Path to a directory with images to check',
    )
    parser.add_argument(
        '--data-root', type=Path, required=False, default=Path('data/'),
        help='Path to the data root directory',
    )
    parser.add_argument(
        '--embeddings-file', type=str, required=False, default=None,
        help='Path to a file produced by the embedder tool',
    )
    parser.add_argument(
        '--output-file', type=str, required=True,
        help='Path to a the file where removed images will be saved',
    )
    return parser.parse_args()


def row_pairwise_distances(m):
    dists = tf.sqrt(
        -2 * m @ tf.transpose(m)  # square
        + tf.math.reduce_sum(m ** 2, axis=1)  # broadcasted over rows
        + tf.expand_dims(tf.math.reduce_sum(m ** 2, axis=1), 0)  # broadcasted over columns
    )
    diag_len = tf.size(tf.linalg.diag_part(dists))
    tf.linalg.set_diag(dists, tf.broadcast_to(0.0, [diag_len]))
    return dists


def main():
    args = parse_arguments()

    if Path(args.output_file).exists():
        blacklist = pd.read_csv(args.output_file, header=None, names=['image_name'])
        blacklisted = set(blacklist.image_name)
    else:
        blacklisted = set()

    if args.images_dir:
        images_dir = args.images_dir

        model = get_model()

        image_paths = list(find_images(images_dir))
        embeddings = compute_embeddings(model, image_paths)
        image_names = [path.relative_to(args.data_root) for path in image_paths]
    elif args.embeddings_file:
        data_df = pq.read_pandas(args.embeddings_file).to_pandas()

        data_df = data_df[~data_df.image_paths.isin(blacklisted)]

        image_names = list(data_df.image_paths)
        embeddings = tf.constant(list(data_df.embeddings))
    else:
        raise RuntimeError('Must pass either images dir or embeddings file')

    distances = row_pairwise_distances(embeddings)

    # Don't show images as similar to themselves
    diag_len = tf.size(tf.linalg.diag_part(distances))
    distances = tf.linalg.set_diag(distances, tf.broadcast_to(np.inf, [diag_len]))

    pct_to_show = 0.25
    number_to_show = int(len(image_names) ** 2 * pct_to_show * 2)
    lowest_distances, lowest_indexes = tf.math.top_k(
        tf.reshape(-distances, [-1]),
        min(3 * len(image_names), number_to_show),
    )
    lowest_distances = -lowest_distances

    image_pairs = []
    for i, distance in zip(lowest_indexes, lowest_distances):
        img1, img2 = np.unravel_index(i, distances.shape)
        if img1 > img2:
            # Only show upper triangle of distance matrix
            continue

        image_pairs.append(ImagePair(
            image1_path=Path(image_names[img1]),
            image2_path=Path(image_names[img2]),
            distance=distance,
        ))

    duplicate_browser = DuplicateBrowser(image_pairs, args.data_root, args.output_file)
    duplicate_browser.run()


if __name__ == '__main__':
    main()
