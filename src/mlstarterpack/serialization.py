import abc
from contextlib import ExitStack
from dataclasses import asdict
import json
import os
from pathlib import Path
from typing import *
from typing import TextIO

import yaml


class SerializableDataclass(abc.ABC):
    @classmethod
    def get_serializer(cls, file: Union[os.PathLike, TextIO]) -> Any:
        if not isinstance(file, os.PathLike):
            raise ValueError(f'Cannot guess file serializer for {file}')

        path = Path(file)
        if path.suffix == '.json':
            return json
        elif path.suffix in ('.yml', '.yaml'):
            return yaml
        else:
            raise ValueError(f'Cannot guess file serializer for {file}')

    @classmethod
    def from_file(
        cls,
        file: Union[os.PathLike, TextIO],
        serializer: Any = None,
        collate: bool = False,
    ):
        with ExitStack() as stack:
            if serializer is None:
                serializer = cls.get_serializer(file)

            if isinstance(file, os.PathLike):
                file = stack.enter_context(open(file))

            if serializer is yaml:
                serializer_params = {'Loader': yaml.SafeLoader}
            else:
                serializer_params = {}

            hp_data = serializer.load(file, **serializer_params)

            if collate:
                for key, value in hp_data.copy().items():
                    type_ = cls.__dataclass_fields__[key].type  # type: ignore
                    hp_data[key] = type_(value)

            return cls(**hp_data)  # type: ignore

    def to_file(self, file: Union[os.PathLike, TextIO], serializer: Any = None):
        with ExitStack() as stack:
            if serializer is None:
                serializer = self.get_serializer(file)

            if isinstance(file, os.PathLike):
                file = stack.enter_context(open(file, 'x'))

            serializer.dump(asdict(self), file)
