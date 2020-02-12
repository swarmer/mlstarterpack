"""Remove images in blacklist from all other datasets"""
import argparse
from pathlib import Path
from typing import *

import pandas as pd


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=str, default='data/datasets/')
    parser.add_argument('blacklist', type=str)
    return parser.parse_args()


def load_dataset(path):
    return list(pd.read_csv(path, header=None, names=['image_name']).image_name)


def save_dataset(image_names: List[str], path):
    images_df = pd.DataFrame({'image_name': image_names})
    images_df.to_csv(path, index=False, header=False)


def main():
    args = parse_arguments()
    dataset_dir = Path(args.dataset_dir)
    blacklist_path = Path(args.blacklist)

    blacklist_names = list(
        pd.read_csv(blacklist_path, header=None, names=['image_name'])
        .image_name
    )

    for dataset_path in dataset_dir.glob('*.csv'):
        dataset = load_dataset(dataset_path)

        filtered_dataset = [
            name
            for name in dataset
            if name not in blacklist_names
        ]

        save_dataset(filtered_dataset, dataset_path)
        print(f'{len(dataset) - len(filtered_dataset)} images removed')


if __name__ == '__main__':
    main()
