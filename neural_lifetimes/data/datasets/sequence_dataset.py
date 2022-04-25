import numbers
from abc import ABC, abstractmethod
from typing import Dict, Sequence, Union

import numpy as np
from torch.utils.data import Dataset, Subset


class SliceableDataset(Dataset, ABC):
    """
    The SliceableDataset is an extension around the `pytorch.utils.data.Dataset` enabling bulk loading of batches.

    When loading datasets from databases, single requests per item are often too slow. Thus, the sliceable
    dataset allows loading of multiple data points with one query.

    The SliceableDataset implements two key changes over the base Dataset class:
     - The `__getitem__` method allows for subsetting using: integers (`dataset[0]`), slicing (`dataset[0:5]`) and
        iterables (`dataset[[0,3,6]]`).
     - The `__getitem__` forwards forwards the request, by default to `_load_batch`. Use this function, to implement the
        querying of the entire batch.

    The SliceableDataset supports the same subsetting as a `List` object with the addition of processing `Sequence`
    datatypes.

    Bases:
        `pytorch.utils.data.Dataset`

    Attributes:
        _item_getter (str): Sets the method that is called by `__getitem__`.

    Warning:
        Negative slicing `dataset[-1:2:-3]` not yet supported.
    """

    _item_getter = '_load_batch'

    @abstractmethod
    def __len__(self):
        pass

    # TODO is Iterable or Sequence correct?
    # TODO make it sliceable like a list with negative arguments
    # TODO rename _get_bulk in load_batch
    def __getitem__(self, item: Union[numbers.Integral, slice, Sequence[numbers.Integral]]) -> Dict[str, np.ndarray]:
        """
        TODO docstring.

        Args:
            item (Union[numbers.Integral, slice, Sequence[numbers.Integral]]): _description_

        Raises:
            ValueError: When negative slicing is used, e.g. dataset[-1:2:-3]
            TypeError: When argument is not a supported data type.

        Returns:
            Dict[str, np.ndarray]: _description_
        """
        item_getter = getattr(self, self._item_getter)
        if isinstance(item, slice):
            start = item.start if item.start is not None else 0
            stop = item.stop if item.stop is not None else len(self)
            step = item.step or 1
            if start < 0 or stop < 0 or step < 0:
                raise ValueError(f"Only supporting positive slices. Got {item}")
            return item_getter(list(range(start, stop, item.step or 1)))
        elif isinstance(item, numbers.Integral):
            return item_getter([item])
        # Sequence[numbers.Integral] is not allowed.
        # TODO empty set
        elif isinstance(item, Sequence) and (len(item) == 0 or isinstance(item[0], numbers.Integral)):
            return item_getter(item)
        else:
            raise TypeError(
                f'{type(self).__name__} indices must be integers, slices or iterable over integers, not'
                f' {type(item).__name__}.'
            )

    @abstractmethod
    def _load_batch(self, s: Sequence[int]) -> Sequence[Dict[str, np.ndarray]]:
        """Get the actual feature vectors for several IDs at once."""


class SliceableSubset(Subset):
    """
    A Pytorch Subset for the `SliceableDataset`.

    Bases:
        `pytorch.utils.data.Subset`
    """

    def __getitem__(self, item: Union[numbers.Integral, slice, Sequence[numbers.Integral]]) -> Dict[str, np.ndarray]:
        """
        Forward the item getter to the underlying dataset.

        Args:
            item (Union[numbers.Integral, slice, Sequence[numbers.Integral]]): identifier of  item

        Returns:
            Dict[str, np.ndarray]: batch
        """
        orig_inds = [self.indices[i] for i in item]
        return self.dataset[orig_inds]

    @property
    def asof_time(self):
        return self.dataset.asof_time


class SequenceSubset(SliceableSubset):
    """
    Subsets a `SequenceDataset`.

    Base:
        `SliceableSubset`
    """

    def get_seq_len(self, idx: int) -> int:
        """
        Forward the sequence length of `self.dataset`.

        Args:
            idx (int): _description_

        Returns:
            int: _description_
        """
        return self.dataset.get_seq_len(idx)


class SequenceDataset(SliceableDataset):
    """
    For variable-length sequences with a homogenous batch size and where data points should not be queried individually.

    The workflow is:
        - Load all data once to retrieve sequence per data point and retrieve these with the `get_seq_len` method.
        - The sampling of batches is based done on the data point identifiers and their sequence length.
        - Once all identifiers are sampled by the dataloader, the `_load_batch` method will query the batch in one go.

    Bases:
        `SliceableDataset`
    """

    @abstractmethod
    def get_seq_len(idx: int) -> int:
        """
        Return the length of a sequence given the index of the data point.

        Args:
            idx (int): The index / identifier of the datapoint.

        Returns:
            int: The length of the sequence of events for this data point.
        """


#################### Attempt at implementing with negative slicing ############  # noqa

# def __getitem__(self, item: Union[numbers.Integral, slice, Sequence[numbers.Integral]]) -> Dict[str, np.ndarray]:
#     """
#     Gets the full sequence for one ID.
#     """
#     if isinstance(item, slice):
#         # start = item.start or 0
#         # stop = item.stop or len(self)

#         # min_, max_ = min(start, stop), max(start, stop)
#         # item = list(range(min_, max_+1))[item]
#         # return self._get_bulk(item)
#         step = item.step if item.step is not None else 1
#         start = item.start if item.start is not None else 0
#         if item.start is not None:
#             start = start if start >= 0 else len(self) + start
#         else:
#             start = start if step < 0 else len(self)
#         stop = item.stop if item.stop is not None else len(self)
#         if item.stop is not None:
#             stop = stop if stop >= 0 else len(self) + stop
#         else:
#             stop = stop if step < 0 else 0
#         if step < 0 and (start < 0 or stop < 0):
#             start, stop = stop-1, start-1
#         return self._get_bulk(list(range(start,stop,step)))

#     elif isinstance(item, numbers.Integral):
#         return self._get_bulk([item])
#     elif isinstance(item, Sequence) and (len(item) == 0 or isinstance(item[0], numbers.Integral)):
#         return self._get_bulk(item)
#     else:
#         raise TypeError(
#             f'{type(self).__name__} indices must be integers, slices or iterable over integers, not'
#             f' {type(item).__name__}.'
#         )
