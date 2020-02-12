import abc
from typing import *


class IterableDataset(abc.ABC):
    def __iter__(self):
        for item in self.source_iterable():
            yield self.process_source_item(item)

    @abc.abstractmethod
    def source_iterable(self) -> Iterable[Any]:
        ...

    def process_source_item(self, item: Any) -> Any:
        return item


class RandomAccessDataset(IterableDataset, abc.ABC):
    def __getitem__(self, item):
        return self.process_source_item(self.source_sequence[item])

    def __len__(self):
        return len(self.source_sequence)

    @property
    @abc.abstractmethod
    def source_sequence(self) -> Sequence[Any]:
        ...

    def source_iterable(self) -> Iterable[Any]:
        return self.source_sequence
