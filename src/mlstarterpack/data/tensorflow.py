import abc
from collections import deque
from concurrent import futures
import multiprocessing as mp
from typing import *

try:
    import tensorflow as tf
except ImportError as exc:
    raise ImportError('This module requires tensorflow extra feature') from exc

from .datasets import IterableDataset


class PreloadingIterator:
    def __init__(
        self,
        iterable: Iterable,
        executor: futures.Executor,
        buf_size: int,
        resolve_func: Callable,
    ):
        self.inner_iterator = iter(iterable)
        self.executor = executor
        self.resolve_func = resolve_func
        self.buf_size = buf_size

        self._future_buffer = deque(maxlen=buf_size)
        for _ in range(buf_size):
            self._enqueue_one()

    def _enqueue_one(self):
        try:
            self._future_buffer.append(
                self.executor.submit(self.resolve_func, next(self.inner_iterator))
            )
        except StopIteration:
            pass

    def __next__(self):
        try:
            future: futures.Future = self._future_buffer.popleft()
        except IndexError:
            raise StopIteration()
        self._enqueue_one()
        return future.result()

    def __iter__(self):
        return self


class TensorflowConvertibleDataset(IterableDataset, abc.ABC):
    _PROCESS_POOL_EXECUTOR = None

    @property
    @abc.abstractmethod
    def tf_dataset_dtype(self) -> Union[tf.DType, Collection[tf.DType]]:
        ...

    @property
    @abc.abstractmethod
    def tf_dataset_shape(self) -> Union[tf.TensorShape, Collection[tf.TensorShape]]:
        ...

    def to_tf_dataset(self, preload_buf_size: int) -> tf.data.Dataset:
        assert mp.get_start_method() == 'spawn'
        return (
            tf.data.Dataset.from_generator(
                lambda: PreloadingIterator(
                    self.source_iterable(),
                    self._get_process_pool_executor(),
                    preload_buf_size,
                    self.process_source_item,
                ),
                self.tf_dataset_dtype,
                self.tf_dataset_shape,
            )
        )

    @classmethod
    def _get_process_pool_executor(cls) -> futures.ProcessPoolExecutor:
        if cls._PROCESS_POOL_EXECUTOR is None:
            cls._PROCESS_POOL_EXECUTOR = futures.ProcessPoolExecutor()

        return cls._PROCESS_POOL_EXECUTOR
