from __future__ import annotations

from pathlib import Path
from typing import *

try:
    import tensorflow.keras as tfk
except ImportError as exc:
    raise ImportError('This module requires tensorflow extra feature') from exc

from .checkpoints import find_latest_checkpoint
from .dir_serializable_model import DirSerializableModel


class TfDirSerializableModel(DirSerializableModel, tfk.Model):
    CHECKPOINT_RE: Pattern

    def load_weights_from_dir(self, model_dir: Path) -> Tuple[Optional[int], Any]:
        self._check_configs_match(model_dir)

        epoch, checkpoint_name = (
            find_latest_checkpoint(model_dir, self.CHECKPOINT_RE)
        )
        status = self.load_weights(str(model_dir / checkpoint_name))
        return epoch, status

    @classmethod
    def load_from_dir(  # pylint: disable=arguments-differ
        cls,
        model_dir: Path,
        training_mode: bool = False,
    ) -> TfDirSerializableModel:
        model = cls.create_uninitialized_from_dir(model_dir)

        _, status = model.load_weights_from_dir(model_dir)
        if not training_mode:
            status.expect_partial()

        return model
