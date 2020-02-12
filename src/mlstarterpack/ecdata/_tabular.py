import os
from pathlib import Path
from typing import *

import pandas as pd


TABLE_EXTENSIONS = ('.csv',)


def load_dataframe(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == '.csv':
        return pd.read_csv(path, header=None)
    else:
        raise ValueError(f'Unknown table file format: {path.suffix}')


def save_dataframe(df: pd.DataFrame, path: Path):
    if path.suffix.lower() == '.csv':
        df.to_csv(path, header=None, index=False)
    else:
        raise ValueError(f'Unknown table file format: {path.suffix}')


def find_table_files(where: Path) -> Iterable[Path]:
    for dirpath, _dirnames, filenames in os.walk(str(where)):
        for filename in filenames:
            file_path = Path(dirpath) / filename
            if not file_path.suffix or file_path.suffix.lower() not in TABLE_EXTENSIONS:
                continue

            yield file_path


def append_table(
    source_path: Path,
    target_path: Path,
    key_map: Optional[Dict[str, str]] = None,
):
    source_df = load_dataframe(source_path)

    if key_map:
        # map image names in the first column to the new ones
        source_df.iloc[:, 0] = source_df.apply(
            lambda row: key_map.get(row[0], row[0]),
            axis=1,
        )

    if target_path.exists():
        target_df = load_dataframe(target_path)
        result_df = pd.concat([target_df, source_df])
    else:
        result_df = source_df

    save_dataframe(result_df, target_path)
