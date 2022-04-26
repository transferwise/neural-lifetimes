import datetime
from typing import Any, Dict, List, Sequence, Union

import numpy as np
import pandas as pd
import pyarrow
import vaex as vx
import vaex.dataframe

from .sequence_dataset import SequenceDataset

DataFrameT = Union[pd.DataFrame, vaex.dataframe.DataFrame]


# TODO there are multiple Pandas datasets in this repo. Maybe group them?
# TODO: implement filtering for min_items? -- Mark: Seems this might be done?
class PandasSequenceDataset(SequenceDataset):
    """
    SequenceDataset interface for Pandas/Vaex DataFrames.

    Args:
        df (DataFrameT): Pandas/Vaex DataFrame to use as a sequence dataset.
        uid_col (str): Name of the column to use as the UID.
        time_col (str): Name of the column to use as the time.
        asof_time (datetime.datetime): Time to use as the asof time. If None, the
            entire dataset is used. Otherwise, only data before or at this time is
            used.
        min_items_per_uid (int): Minimum number of items per UID to include in the
            dataset. Defaults to ``1``.

    Attributes:
        df (DataFrameT): Pandas/Vaex DataFrame to use as a sequence dataset. Only
            UIDs with at least min_items_per_uid items are included.
        id_column (str): Name of the column to use as the UID.
        time_col (str): Name of the column to use as the time.
        asof_time (datetime.datetime): Time to of the last transaction.
        uid_sizes (Dict[str, int]): Dictionary of UIDs and their frequencies. UIDs
            are included if the frequency is greater than or equal to
            min_items_per_uid.
        ids (List[str]): Sorted list of UIDs to use in the dataset.
        dtypes (Dict[str, type]): Dictionary of column names and their types.
        id_index (Dict[str, tuple]): Dictionary of uids and the index of their first
            and last occurrence in df (sorted)

    Note:
        We assume the dataframe is already sorted by timestamp.
    """

    def __init__(
        self,
        df: DataFrameT,
        uid_col: str,
        time_col: str,
        asof_time: datetime.datetime = None,
        min_items_per_uid: int = 1,
    ):
        if asof_time is not None:
            if isinstance(df, vx.dataframe.DataFrame):
                df = df[df[time_col] <= np.datetime64(asof_time)]
            else:
                df = df[df[time_col] <= asof_time]

        self.df = df
        self.id_column = uid_col
        self.time_col = time_col
        self.asof_time = asof_time
        self.uid_sizes = dict(self.df[uid_col].value_counts()[self.df[uid_col].value_counts() >= min_items_per_uid])
        self.ids = sorted(list(self.uid_sizes.keys()))
        self.df = _dataframe_sort(
            self.df[self.df[uid_col].isin(self.ids)],
            by=[uid_col, time_col],
        )
        self.dtypes = self.df.dtypes

        # For each uid we find the index it first occurred in dataset and the index of
        # its last occurrence (note df is sorted). To do this we load the array of all
        # uids (for all events) into memory.
        self.id_index = {}

        tmp_index = 0

        uuids_list = self.df[self.id_column].values

        tmp_id = uuids_list[0]
        dataframe_length = _dataframe_length(self.df)
        for i in range(dataframe_length):
            # TODO Mark: What's this for? Was it supposed to be a continue or something
            #   for debugging?
            if i % 1000 == 0:
                pass
            if uuids_list[i] != tmp_id:
                self.id_index[_scalar_to_py(tmp_id)] = (tmp_index, i - 1)
                tmp_id = uuids_list[i]
                tmp_index = i

        self.id_index[_scalar_to_py(tmp_id)] = (tmp_index, dataframe_length - 1)

    def __len__(self) -> int:
        """
        Get number of unique uids in the dataset post filtering.

        Returns:
            int: number of unique uids.
        """
        return len(self.ids)

    def get_seq_len(self, i: int) -> int:
        """
        Return the length of sequence for the provided uid.

        Args:
            uid (int): A valid user id.

        Returns:
            int: length of the sequence of transactions for this uid.
        """
        return self.uid_sizes[self.ids[i]]

    def _load_batch(self, inds: Sequence[int]) -> Sequence[Dict[str, np.ndarray]]:
        """
        Get sequences for a list of UIDS.

        Args:
            inds: List of UIDS to get sequences for.

        Returns:
            seqs (Sequence[Dict[str, np.ndarray]]): Sequence of dicts.
        """
        uids = [self.ids[i] for i in inds]
        # get the data for all the ids
        indices = []
        for id_ in uids:
            indices.extend([i for i in range(self.id_index[id_][0], self.id_index[id_][1] + 1)])
        raw_data = self.df.take(indices)
        # TODO Mark: Drop old time column?
        # rename time
        raw_data["t"] = raw_data[self.time_col]

        # TODO Mark: Do we need to do this? It will load everything into memory. Might
        #   be worth checking if we need to do it OOC with Vaex?
        pre_out = _dataframe_to_dict(raw_data)

        for key in pre_out:
            if not isinstance(pre_out[key], (np.ndarray, np.generic)):
                pre_out[key] = np.array(pre_out[key])

        # slice it up by ID and apply the transform
        seqs = []
        offsets = [0]
        # split the query result into sequences by uid
        for next_item in uids:
            len_ = self.uid_sizes[next_item]
            offsets.append(offsets[-1] + len_)
            this_seq = {k: v[offsets[-2] : offsets[-1]] for k, v in pre_out.items()}
            seqs.append(this_seq)

        return seqs


def _dataframe_sort(df: DataFrameT, by: List[str]) -> DataFrameT:
    """
    Sort a dataframe by the list of columns provided in priority order.

    This is a compatibility shim between Pandas and Vaex which have a different
    interface for dataframe sorting.

    Args:
        df (DataFrameT):
        by (List[str]):

    Returns:
        DataFrameT: The sorted dataframe (of the same type as the input).
    """
    if isinstance(df, pd.DataFrame):
        return df.sort_values(by)
    return df.sort(by)


def _dataframe_length(df: DataFrameT) -> int:
    """
    Get the length of the dataframe.

    For Vaex, this is the full length of the underlying dataframe, not just the active
    portion.

    Args:
        df (DataFrameT): A Pandas/Vaex dataframe.

    Returns:
        int: Number of rows in the dataframe
    """
    if isinstance(df, pd.DataFrame):
        return len(df)
    return df.length_original()


def _scalar_to_py(value: Union[pyarrow.Scalar, Any]) -> Union[pyarrow.Scalar, Any]:
    """
    Convert a pyarrow scalar to its corresponding python value.

    Leave any other input untouched.

    Args:
        value (Union[pyarrow.Scalar, Any]):

    Returns:
        value: The pyarrow scalar as a python primitive, or the origin input if not a
            scalar.
    """
    if isinstance(value, pyarrow.Scalar):
        return value.as_py()
    return value


def _dataframe_to_dict(df: DataFrameT) -> Dict[str, Sequence]:
    """
    Convert the dataframe to a dict of format {column -> [values]}.

    Args:
        df (DataFrameT): A Pandas/Vaex dataframe.

    Returns:
        Dict: Number of rows in the dataframe
    """
    if isinstance(df, pd.DataFrame):
        return df.to_dict(orient="list")
    return df.to_dict()
