from __future__ import annotations

import abc
from pathlib import Path
from typing import *

import yaml


M = TypeVar('M', bound='DirSerializableModel')  # pylint: disable=invalid-name


class DirSerializableModel(abc.ABC):
    MODEL_CONFIG_FILENAME = 'model_config.yml'

    def get_config(self) -> dict:
        return {}

    @classmethod
    def from_config(cls: Type[M], config: dict) -> M:
        return cls(**config)  # type: ignore

    @classmethod
    def load_config_from_dir(cls, model_dir: Path) -> dict:
        with open(str(model_dir / cls.MODEL_CONFIG_FILENAME)) as infile:
            return yaml.load(infile, Loader=yaml.SafeLoader)

    @classmethod
    def save_config_to_dir(cls, model_dir: Path, config: dict):
        with open(str(model_dir / cls.MODEL_CONFIG_FILENAME), 'x') as outfile:
            return yaml.dump(config, outfile)

    def save_model_config(self, model_dir: Path):
        self.save_config_to_dir(model_dir, self.get_config())

    @classmethod
    def create_uninitialized_from_dir(cls: Type[M], model_dir: Path) -> M:
        """Create a new instance of a model with config taken from the model_dir.

        The result will not have weights initialized, but will have the correct structure
        (shapes, classes etc.).
        """
        config = cls.load_config_from_dir(model_dir)
        return cls.from_config(config)

    def _check_configs_match(self, model_dir: Path):
        config = self.load_config_from_dir(model_dir)
        if config != self.get_config():
            raise RuntimeError('Current model\'s config doesn\'t match saved config')

    @abc.abstractmethod
    def load_weights_from_dir(self, model_dir: Path) -> Tuple[Optional[int], Any]:
        """Set the instance's weights to values saved in the model_dir.

        :returns: A tuple of the epoch number of the loaded snapshot (if any) and
            framework-specific loading status.
        """
        ...

    @classmethod
    @abc.abstractmethod
    def load_from_dir(cls: Type[M], model_dir: Path) -> M:  # type: ignore
        ...
