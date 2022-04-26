# flake8: noqa
import math
import random
from typing import Callable, Dict, Sequence, Union

import numpy as np
import torch
from pandas.api.types import is_float_dtype, is_integer_dtype, is_numeric_dtype
from torch.utils.data import Dataset, SubsetRandomSampler

# # doesn't seem like these are used anywhere
# class DatasetFromTimeSeries(Dataset):
#     """
#     Take a time series of arrays, assuming time is the leftmost dimension, get samples with specified number of
#     past_lags.

#     Args:
#         data (Dict[str, torch.Tensor]): a dictionary of tensors, where the keys are the names of the features
#         past_lags (int): the number of past lags to include. Must be a non-negative integer. Defaults to 1.
#         days_ahead (int): the number of days ahead to include. Defaults to 1.
#         transform (Callable): a function to transform the data before returning it. Defaults to ``lambda x: x``.
#         meta (Dict): a dictionary of meta data. Defaults to ``{}``.

#     Attributes:
#         data (Dict[str, torch.Tensor]): a dictionary of tensors, where the keys are the names of the features
#         lags (int): the number of past lags to include. Must be a non-negative integer.
#         days_ahead (int): the number of days ahead to include.
#         transform (Callable): a function to transform the data before returning it.
#         meta (Dict): a dictionary of meta data.
#         output_cache (Dict[int, Dict]): a cache of the output of the dataset. Initialized to ``{}``.
#     """

#     def __init__(
#         self,
#         data: Dict[str, torch.Tensor],
#         lags: int = 1,
#         days_ahead: int = 1,
#         transform: Callable = lambda x: x,
#         meta: Dict = {},
#     ) -> Dict[str, Any]:
#         assert lags >= 1, "past lags must be a non-negative integer!"
#         self.data = data
#         self.lags = lags
#         self.days_ahead = days_ahead
#         self.transform = transform
#         self.meta = meta
#         self.output_cache = {}
#         logging.info(
#             f"""Created a DatasetFromTimeseries,
#                             first date {self.ind_to_date(0)},
#                             last date (inclusive) {self.ind_to_date(len(self))},
#                         """
#         )

#     def __getitem__(self, index):
#         if index not in self.output_cache:
#             this_index = index + self.lags
#             pre_out = {
#                 "orig_index": this_index,
#                 "past": {
#                     key: value[index:this_index] for key, value in self.data.items()
#                 },
#                 "future": {
#                     key: value[this_index : (this_index + self.days_ahead)]
#                     for key, value in self.data.items()
#                 },
#             }

#             self.output_cache[index] = self.transform(pre_out)
#         return self.output_cache[index]

#     def ind_to_date(self, index: int) -> datetime.date:
#         """
#         Finds the last observed date for a given index.

#         Args:
#             index (int): the index of the customer

#         Returns:
#             datetime.date: the last observed date
#         """

#         return self.meta["dates"][index + self.lags - 1]

#     def __len__(self):
#         return len(list(self.data.values())[0]) - self.lags - self.days_ahead + 1

#     def date_range_subset(self, start_date: datetime.date, end_date: datetime.date):
#         """
#         Get a subset of the dataset referring to certain dates

#         Args:
#             start_date (datetime.date): the first date to include
#             end_date (datetime.date): the last date to include (not inclusive)

#         Returns:
#             DatasetFromTimeSeries: a subset of the dataset in the range
#         """
#         assert start_date >= self.ind_to_date(
#             0
#         ), "start_date must be within the date range of the dataset"
#         assert end_date <= self.ind_to_date(
#             len(self)
#         ), "end_date must be within the date range of the dataset"

#         inds = [
#             i
#             for i in range(len(self))
#             if self.ind_to_date(i) >= start_date and self.ind_to_date(i) < end_date
#         ]
#         slice_ds = Subset(self, inds)
#         return slice_ds


# class DatasetFromDict(Dataset):
#     def __init__(self, data: dict):
#         self.data = data

#     def __getitem__(self, index):
#         return {key: value[index] for key, value in self.data.items()}

#     def __len__(self):
#         return len(list(self.data.values())[0])

# TODO move to data.
def train_valid_loaders(dataset, valid_size=0.1, **kwargs):
    """
    Split a dataset into train and validation sets.

    Args:
        dataset (Dataset): the dataset to split
        valid_size (float): the proportion of the dataset to use for validation. Defaults to ``0.1``.
        If the input is greater than 1, instead it is treated as a number of entries to use for validation.
        **kwargs: additional arguments to pass to the ``torch.utils.data.DataLoader`` constructor.

    Returns:
        (torch.utils.data.DataLoader, torch.utils.data.DataLoader): the train and validation loaders
    """
    # num_workers
    # batch_size
    num_train = len(dataset)
    if valid_size > 1:
        valid_size /= num_train

    indices = list(range(num_train))
    split = int(math.floor((1 - valid_size) * num_train))

    if not ("shuffle" in kwargs and not kwargs["shuffle"]):
        # np.random.seed(random_seed)
        np.random.shuffle(indices)
        kwargs.pop("shuffle")

    train_idx, valid_idx = indices[:split], indices[split:]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(dataset, sampler=train_sampler, **kwargs)
    valid_loader = torch.utils.data.DataLoader(dataset, sampler=valid_sampler, **kwargs)

    return train_loader, valid_loader


class SequenceSamplingDataset(Dataset):
    """
    A dataset that samples sequences from a given dataset.

    Args:
        data (Dict[str, Union[torch.Tensor, Sequence[str]]]): the data to sample from.
        transform (Callable): a function to transform the data before returning it.
        uid_name (str): the name of the unique identifier in the data.
        max_item_len (int): the maximum length of the sequences to sample. Defaults to 100.

    Attributes:
        data (Dict[str, Union[torch.Tensor, Sequence[str]]]): the data to sample from.
        transform (Callable): a function to transform the data before returning it.
        uid_name (str): the name of the unique identifier in the data.
        uids (torch.Tensor): the unique identifiers in the data.
        _uid_to_ind (Dict[str, int]): a mapping from unique identifiers to indices. For caching in ``uid_to_inds``.
        max_item_len (int): the maximum length of the sequences to sample.
    """

    def __init__(
        self,
        data: Dict[str, Union[torch.Tensor, Sequence[str]]],
        transform: Callable,
        uid_name: str,
        max_item_len: int = 100,
    ):
        self.data = data
        self.transform = transform
        self.uid_name = uid_name
        # create an index so can sample by UID later
        self.uids = torch.unique(data[uid_name])
        self._uid_to_ind = {}
        self.max_item_len = max_item_len

    def __len__(self):
        """
        Get the number of unique user ids in the dataset.

        Returns:
            int: number of uids.
        """
        return len(self.uids)

    def uid_to_inds(self, uid):
        """
        Cache the lookup by ID for reuse in next epoch.

        Args:
            uid (str): the unique identifier of the user

        Returns:
            torch.Tensor: that user's data.
        """
        #
        if uid not in self._uid_to_ind:
            self._uid_to_ind[uid] = torch.nonzero(self.data[self.uid_name] == uid).reshape(-1)

        return self._uid_to_ind[uid]

    def __getitem__(self, uid: str):
        """
        Get a sequence for a given user id.

        Args:
            uid (str): Unique id of a user in the dataset.

        Returns:
            torch.Tensor: Sequence for the given user.
        """
        # TODO: implement caching of transform results?
        these_inds = self.uid_to_inds(self.uids[uid])
        these_inds_np = these_inds.cpu().detach().numpy()
        pre_out = {}
        for key, value in self.data.items():
            if isinstance(value, torch.Tensor):
                pre_out[key] = value[these_inds][-self.max_item_len :]
            else:
                pre_out[key] = value[these_inds_np][-self.max_item_len :]

        return self.transform(pre_out)


# TODO move to data.dataloaders
class SequenceLoader:
    """
    Sample random sequences, up to a total of max_size points.

    Args:
        data (SequenceSamplingDataset): the dataset to sample from
        max_size (int): the maximum number of points to sample
        shuffle (bool): whether to shuffle the dataset. Defaults to ``True``.

    Its attributes are the same as the above arguments.
    """

    #
    def __init__(self, data: SequenceSamplingDataset, max_points: int, shuffle: bool = True):
        self.data = data
        self.max_points = max_points
        self.shuffle = shuffle

    def __iter__(self):
        """
        Construct a random sequence iterator over the dataset.

        Returns:
            RandomSequenceIterator: iterator over the dataset.
        """
        return RandomSequenceIterator(self.data, self.max_points, self.shuffle)


# TODO move to data.dataloaders
class RandomSequenceIterator:
    """
    An iterator that samples from a dataset of sequences.

    Takes a dataset where items are variable-length sequences
    Randomly samples those into a batch with a maximum number of points

    Args:
        data (SequenceSamplingDataset): the dataset to sample from
        max_size (int): the maximum number of points to sample
        shuffle (bool): whether to shuffle the dataset. Defaults to ``True``.

    Attributes:
        data (SequenceSamplingDataset): the dataset to sample from
        max_points (int): the maximum number of points to sample
        inds (torch.Tensor): the indices of the items in the dataset. If ``shuffle==True``, this will be shuffled
            initially.
        next_ind (int): the index of the next item to sample. Initialised to 0.

    Yields:
        Dict[str,Union[np.array,torch.Tensor]]: a dictionary of the sampled data, as well as the offsets of the data.
    """

    def __init__(self, data: SequenceSamplingDataset, max_points: int, shuffle: bool):
        self.data = data
        self.max_points = max_points
        self.inds = list(range(len(self.data)))
        if shuffle:
            random.shuffle(self.inds)
        self.next_ind = 0

    def __next__(self):
        offsets = [0]
        total_size = 0
        pre_out = []
        while True:
            if self.next_ind >= len(self.data):
                raise StopIteration

            next_item = self.data[self.next_ind]
            # for now, items with only one transaction per user are ignored
            if next_item is None:
                continue
            next_len = len(next(iter(next_item.values())))
            if total_size + next_len <= self.max_points:
                total_size += next_len
                offsets.append(total_size)
                pre_out.append(self.data[self.next_ind])
                self.next_ind += 1
            else:
                break

        out = {}
        for key, val in pre_out[0].items():
            if isinstance(val, torch.Tensor):
                out[key] = torch.cat([x[key] for x in pre_out], dim=0)
            else:
                out[key] = np.concatenate([x[key] for x in pre_out], axis=0)

        out["offsets"] = offsets

        return out


# TODO move to data
def torchify(x: Dict[str, np.ndarray], device: torch.device) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
    """
    Cast all numerical elements to tensors, forcing float64 to float32.

    Args:
        x (Dict[str, np.ndarray]): the dictionary to cast
        device (torch.device): the device to cast to

    Returns:
        Dict[str, Union[np.ndarray, torch.Tensor]]: the dictionary with all numerical elements cast to tensors
    """
    out = {}
    for k, v in x.items():
        if is_numeric_dtype(v.dtype):
            nice_type = normalize_types(v.dtype)
            out[k] = torch.tensor(v.astype(nice_type), device=device)
        else:
            out[k] = v
    return out


def normalize_types(x: np.dtype):
    """
    Find the type to be cast to for a given dtype: np.float32 for floats, np.int64 for ints.

    Note the conversion of ints is required for pytorch loss functions.

    Args:
        x (np.dtype): the dtype for which to determine cast type.
    """
    if is_float_dtype(x):
        return np.float32
    elif is_integer_dtype(x):
        return np.int64  # torch loss functions require that
    else:
        return x
