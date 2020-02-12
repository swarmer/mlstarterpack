import os
import re
from pathlib import Path
from typing import *

import mlstarterpack as mlsp


DATA_DIR_STRUCTURE = {
    'datasets': {},
    'source': {
        'images': {},
        'metadata': {},
    }
}


def create_minimal_data_dirs(data_dir: Path):
    mlsp.dirs.prepare_directory(
        data_dir,
        DATA_DIR_STRUCTURE,
        on_exists=mlsp.dirs.DirectoryExistsStrategy.ignore,
    )


def find_metadata_csvs(where: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(str(where)):
        for filename in filenames:
            file_path = Path(dirpath) / filename

            if not re.match(r'^[a-zA-Z0-9]+\.csv$', filename):
                continue

            yield file_path


def find_dataset_csvs(where: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(str(where)):
        for filename in filenames:
            file_path = Path(dirpath) / filename

            if not re.match(r'^[a-zA-Z0-9]+_[a-zA-Z0-9]+\.csv$', filename):
                continue

            yield file_path
