"""A series of functions to convert dataframes to pytorch tensors."""
from typing import Dict, Union

import numpy as np
import pandas as pd
import torch

from pandas.api.types import is_float_dtype, is_integer_dtype, is_numeric_dtype


def torchify(x: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    out = {}
    for key, val in x.items():
        try:
            out[key] = torch.from_numpy(val)
        except TypeError:
            out[key] = val
    return out


# TODO: remove if no longer needed
def torchify_old(x: Dict[str, np.ndarray]) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
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
            out[k] = torch.tensor(v.astype(nice_type))
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


def remove_isolated_transactions(df: pd.DataFrame, uid_name: str):
    """
    Delete users which only have 1 transaction.

    Args:
        df (pd.DataFrame): The dataframe to clean.
        uid_name (str): The name of the user id column.

    Returns:
        pd.DataFrame: The cleaned dataframe.
    """
    counts = df[uid_name].value_counts()

    out = df[df[uid_name].isin(set(counts[counts > 1].index))]
    return out


def detorch(x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """
    Convert a torch tensor to a numpy array.

    Args:
        x (Union[torch.Tensor, np.ndarray]): The tensor to convert.

    Returns:
        np.ndarray: The numpy array.
    """
    if isinstance(x, torch.Tensor):
        return x.cpu().detach().numpy()
    else:
        return x


def batch_to_dataframe(x: Dict[str, Union[torch.Tensor, np.ndarray]]) -> pd.DataFrame:
    """
    Convert a dictionary of tensors or numpy arrays to a pandas dataframe. Drops the "offsets" column.

    Args:
        x (Dict[str, Union[torch.Tensor, np.ndarray]]): The dictionary to convert.

    Returns:
        pd.DataFrame: The dataframe.
    """
    return pd.DataFrame.from_dict({k: detorch(v) for k, v in x.items() if k != "offsets"})
