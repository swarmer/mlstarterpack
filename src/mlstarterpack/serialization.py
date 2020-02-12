import abc
from contextlib import ExitStack
from dataclasses import asdict
import os
from typing import *
from typing import TextIO

import yaml


class SerializableDataclass(abc.ABC):
    @classmethod
    def from_yml_file(cls, yml_file: Union[os.PathLike, TextIO]):
        with ExitStack() as stack:
            if isinstance(yml_file, os.PathLike):
                yml_file = stack.enter_context(open(yml_file))

            hp_data = yaml.load(yml_file, Loader=yaml.SafeLoader)
            return cls(**hp_data)  # type: ignore

    def to_yml_file(self, yml_file: Union[os.PathLike, TextIO]):
        with ExitStack() as stack:
            if isinstance(yml_file, os.PathLike):
                yml_file = stack.enter_context(open(yml_file, 'x'))

            yaml.dump(asdict(self), yml_file)
