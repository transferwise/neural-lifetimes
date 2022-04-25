"""A series of functions to convert dataframes to pytorch tensors."""
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch

from .datasets.sequence_sampling_dataset import SequenceSamplingDataset

# TODO move this / rename this. Is it better to have this in data. or data.utils. or utils.data


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


def make_sequence_dataset(
    df: pd.DataFrame,
    cont_feat: List[str],
    discr_feat: List[str],
    transform: Callable,
    max_item_len: int,
    device: torch.device,
    uid_name: str,
    time_col: str,
) -> Tuple[SequenceSamplingDataset, Dict[str, set]]:
    """
    Convert a dataframe to a SequenceSamplingDataset.

    Args:
        df (pd.DataFrame): The dataframe to convert.
        cont_feat (List[str]): The continuous features.
        discr_feat (List[str]): The discrete features.
        transform (Callable): The transform to apply to the dataframe.
        max_item_len (int): The maximum length of an item.
        device (torch.device): The device to use.
        uid_name (str): The name of the user id column.
        time_col (str): The name of the time column.

    Returns:
        Tuple[SequenceSamplingDataset, Dict[str, set]]: The SequenceSamplingDataset and the set of unique user ids.
    """
    data_dict = {c: torch.tensor(df[c].values, dtype=torch.float32, device=device) for c in cont_feat}

    data_dict[uid_name] = torch.tensor(df[uid_name].values, dtype=torch.long, device=device)

    data_dict[time_col] = df[time_col].values

    # time since epoch in seconds
    data_dict["t"] = torch.tensor(
        df[time_col].values.astype(np.int64) // 10**9,
        dtype=torch.float32,
        device=device,
    )

    for f in discr_feat:
        data_dict[f] = df[f].values.astype(str)

    return SequenceSamplingDataset(data_dict, transform, uid_name, max_item_len)


def normalize_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the types of the dataframe: object -> str, float64 -> float32.

    Args:
        df (pd.DataFrame): The dataframe to normalize.

    Returns:
        pd.DataFrame: The dataframe with normalized types.
    """
    for c in df.columns:
        if df[c].dtype == np.dtype("O"):
            df[c] = df[c].astype(str)

        if df[c].dtype == np.dtype("float64"):
            df[c] = df[c].astype(np.float32)
    return df


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
