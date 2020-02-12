import argparse
import os
from pathlib import Path
import shutil

import pandas as pd

from ._tabular import TABLE_EXTENSIONS
from .utils import (
    create_minimal_data_dirs,
    find_dataset_csvs,
    find_metadata_csvs,
)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--fraction', type=float, required=True,
        help='Approximate fraction of samples to keep, ex. 0.1',
    )
    parser.add_argument(
        '--min-per-dataset', type=int, required=False, default=10,
        help='Keep at least this many samples from each dataset',
    )
    parser.add_argument(
        '--input-data-root', type=Path, required=False, default=Path('data/'),
        help='Input dataset',
    )
    parser.add_argument(
        '--output-data-root', type=Path, required=True,
        help='Where to store resulting dataset',
    )
    return parser.parse_args()


def downsample_dataset(df: pd.DataFrame, fraction: float, min_count: int) -> pd.DataFrame:
    count_by_fraction = len(df) * fraction
    if count_by_fraction < min_count:
        return df.sample(n=min_count)
    else:
        return df.sample(frac=fraction)


def main():  # pylint: disable=too-many-locals
    args = parse_arguments()

    if args.output_data_root.exists():
        raise RuntimeError('Output directory must not exist')
    create_minimal_data_dirs(args.output_data_root)

    print('Sampling datasets...')
    remaining_object_names = set()
    for dataset_csv_path in find_dataset_csvs(args.input_data_root / 'datasets'):
        input_df = pd.read_csv(dataset_csv_path, header=None)
        downsampled_df = downsample_dataset(input_df, args.fraction, args.min_per_dataset)

        relative_path = dataset_csv_path.relative_to(args.input_data_root)
        downsampled_path = args.output_data_root / relative_path
        downsampled_df.to_csv(downsampled_path, header=None, index=False)

        remaining_object_names.update(downsampled_df[0])

    print('Copying metadata...')
    for metadata_csv_path in find_metadata_csvs(args.input_data_root / 'source/metadata'):
        df = pd.read_csv(metadata_csv_path, header=None)
        downsampled_df = df[df[0].isin(remaining_object_names)]

        relative_path = metadata_csv_path.relative_to(args.input_data_root)
        downsampled_path = args.output_data_root / relative_path
        downsampled_df.to_csv(downsampled_path, header=None, index=False)

    print('Copying source data...')
    for dirpath, _dirnames, filenames in os.walk(args.input_data_root / 'source/'):
        if Path(dirpath).relative_to(args.input_data_root) == Path('source/metadata/'):
            continue

        for filename in filenames:
            file_path = Path(dirpath) / filename
            file_name = file_path.relative_to(args.input_data_root)
            new_file_path = args.output_data_root / file_name

            if (
                file_path.suffix.lower() in TABLE_EXTENSIONS
                or str(file_name) in remaining_object_names
            ):
                shutil.copy(file_path, new_file_path)

    print('Done')


if __name__ == '__main__':
    main()
