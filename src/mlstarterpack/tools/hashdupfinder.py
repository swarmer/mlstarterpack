from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
from pathlib import Path
import random

import pandas as pd

from mlstarterpack.vision.images import find_images


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-root', type=Path, required=False, default=Path('data/'),
        help='Path to the data root directory',
    )
    parser.add_argument(
        '--images-dir', type=str, required=False, default='data/source/images/',
        help='Path to the directory with images to check',
    )
    parser.add_argument(
        '--output-file', type=str, required=True,
        help='Path to the file where removed images will be saved',
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    if Path(args.output_file).exists():
        blacklist = pd.read_csv(args.output_file, header=None, names=['image_name'])
        blacklisted = set(blacklist.image_name)
    else:
        blacklisted = set()

    images_dir = Path(args.images_dir)
    image_paths = list(find_images(images_dir))
    random.shuffle(image_paths)

    hash_map = defaultdict(list)  # hash -> List[image_path]
    for image_path in image_paths:
        image_name = image_path.relative_to(args.data_root)
        with open(image_path, 'rb') as infile:
            image_data = infile.read()

        image_hash = hashlib.blake2b(image_data).hexdigest()
        hash_map[image_hash].append(image_name)

    for image_hash, hash_paths in hash_map.items():
        if len(hash_paths) <= 1:
            continue

        print(f'Hash {image_hash} is shared by multiple images:')
        for hash_path in hash_paths:
            print(f'  {hash_path}')
        print('Blacklisting all except the first one', end='\n\n')
        blacklisted.update(hash_paths[1:])

    blacklist_series = pd.Series(list(blacklisted))
    blacklist_series.to_csv(args.output_file, index=False)


if __name__ == '__main__':
    main()
