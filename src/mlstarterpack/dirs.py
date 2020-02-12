import enum
import os
from pathlib import Path
from shutil import rmtree
from typing import *


class DirectoryExistsError(Exception):
    pass


class DirectoryCleanRefusedError(Exception):
    pass


class DirectoryExistsStrategy(enum.Enum):
    error = enum.auto()
    ignore = enum.auto()
    clean = enum.auto()


def default_dir_clean_verify(dirname: Union[str, os.PathLike]):
    print(f'Directory {dirname} has files, proceed deleting?')
    response = input('Enter yes to proceed: ')
    if response.lower() not in ('y', 'yes'):
        raise DirectoryCleanRefusedError()


def prepare_directory(
    dirname: Path,
    structure: Optional[Dict[str, dict]] = None,
    on_exists: DirectoryExistsStrategy = DirectoryExistsStrategy.error,
    dir_clean_verify: Callable[[Union[str, os.PathLike]], None] = default_dir_clean_verify,
):
    if structure is None:
        structure = {}

    if os.path.exists(dirname):
        if on_exists == DirectoryExistsStrategy.error:
            raise DirectoryExistsError()
        elif on_exists == DirectoryExistsStrategy.clean:
            if os.listdir(dirname):
                dir_clean_verify(dirname)
            rmtree(dirname)
        elif on_exists == DirectoryExistsStrategy.ignore:
            pass
        else:
            raise RuntimeError()

    os.makedirs(dirname, exist_ok=True)

    for subdir_name, subdir_structure in structure.items():
        prepare_directory(
            dirname / subdir_name,
            subdir_structure,
            on_exists=DirectoryExistsStrategy.ignore,
        )
