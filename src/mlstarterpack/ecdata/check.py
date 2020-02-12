import argparse
from pathlib import Path

import pandas as pd

from mlstarterpack.vision.images import find_images
from ._tabular import find_table_files, load_dataframe
from .utils import (
    find_dataset_csvs,
    find_metadata_csvs,
)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-root', type=Path, required=False, default=Path('data/'),
        help='Data root to check',
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    print('Reading datasets...')
    dataset_keys = set()
    for dataset_csv_path in find_dataset_csvs(args.data_root / 'datasets'):
        input_df = pd.read_csv(dataset_csv_path, header=None)
        dataset_keys.update(input_df[0])

    print('Checking metadata...')
    metadata_keys = set()
    for metadata_csv_path in find_metadata_csvs(args.data_root / 'source/metadata'):
        df = pd.read_csv(metadata_csv_path, header=None)
        metadata_keys.update(df[0])

    not_in_metadata = dataset_keys - metadata_keys
    if not_in_metadata:
        print('WARNING: Keys in dataset csvs, but not in metadata csvs:')
        for key in not_in_metadata:
            print(f'  {key}')
        print()

    print('Checking object files...')
    object_keys = frozenset(
        str(image_path.relative_to(args.data_root))
        for image_path in find_images(args.data_root / 'source/')
    )

    print('Checking tabular files...')
    table_keys = set()
    for table_path in find_table_files(args.data_root / 'source/'):
        df = load_dataframe(table_path)
        table_keys.update(df[0])

    missing_keys = (dataset_keys | metadata_keys) - (object_keys | table_keys)
    if missing_keys:
        print('ERROR: Keys in csvs, but not among object files:')
        for key in missing_keys:
            print(f'  {key}')
        print()

    print('Done')


if __name__ == '__main__':
    main()
