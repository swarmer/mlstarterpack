import abc
import enum


class ImageLayout(enum.Enum):
    HWC = enum.auto()
    CHW = enum.auto()

    @property
    def keras_name(self) -> str:
        if self is self.HWC:
            return 'channels_last'
        elif self is self.CHW:
            return 'channels_first'
        else:
            raise RuntimeError()


class ImageLayoutAnnotated(abc.ABC):
    IMAGE_LAYOUT: ImageLayout
