"""Import a folder with additions to the datasets"""
import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
from typing import *
import uuid

from ._tabular import append_table, TABLE_EXTENSIONS
from .utils import (
    create_minimal_data_dirs,
    find_dataset_csvs,
    find_metadata_csvs,
)


@dataclass
class PathParameters:
    data_drop_dir: Path
    output_dir: Path


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_drop_dir', type=str)
    parser.add_argument('--output-data-root', type=str, default='data/', required=False)
    return parser.parse_args()


def import_source_data(params: PathParameters) -> Dict[str, str]:
    """Copy renamed objects to output and return mapping
    from source names to renamed ones
    """
    object_name_map = {}
    for dirpath, _dirnames, filenames in os.walk(str(params.data_drop_dir / 'source')):
        rel_dirpath = str(Path(dirpath).relative_to(params.data_drop_dir / 'source'))
        if rel_dirpath in ('.', 'metadata'):
            continue

        for filename in filenames:
            if filename.startswith('.'):
                continue

            file_path = Path(dirpath) / filename

            extension = file_path.suffix.lower()
            original_name = str(file_path.relative_to(params.data_drop_dir))

            if extension in TABLE_EXTENSIONS:
                append_table(file_path, params.output_dir / original_name)
            else:
                uid = str(uuid.uuid4())
                new_name = (
                    file_path.parent.relative_to(params.data_drop_dir)
                    / f'{uid}{extension}'
                )
                new_path = params.output_dir / new_name

                object_name_map[original_name] = str(new_name)

                shutil.copy(str(file_path), str(new_path))

    return object_name_map


def main():
    args = parse_arguments()
    output_dir = Path(args.output_data_root)
    params = PathParameters(
        data_drop_dir=Path(args.data_drop_dir),
        output_dir=output_dir,
    )

    create_minimal_data_dirs(params.output_dir)

    print('Importing source files...')
    object_key_map = import_source_data(params)

    print('Importing metadata...')
    for metadata_csv_path in find_metadata_csvs(params.data_drop_dir):
        append_table(
            metadata_csv_path,
            output_dir / 'source/metadata' / metadata_csv_path.name,
            object_key_map,
        )

    print('Importing datasets...')
    for dataset_csv_path in find_dataset_csvs(params.data_drop_dir):
        append_table(
            dataset_csv_path,
            output_dir / 'datasets' / dataset_csv_path.name,
            object_key_map
        )

    print('Done')


if __name__ == '__main__':
    main()
