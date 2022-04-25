import glob
from typing import Callable

import pandas as pd

from .query import caching_query


def jsons_files_to_iterable(pattern: str):
    """
    Iterates over all files in a directory and yields a json / dictionary for each file.

    Args:
        pattern (str): The pattern to match files against.

    Yields:
        dict: A dictionary representing the json file.
    """
    return files_to_df_iterable(pattern, lambda x: pd.read_json(x, lines=True))


def csv_files_to_iterable(pattern: str):
    """
    Iterates over all files in a directory and yields a dataframe for each file.

    Args:
        pattern (str): The pattern to match files against.

    Yields:
        pd.DataFrame: A dataframe representing the csv file.
    """
    return files_to_df_iterable(pattern, lambda x: pd.read_csv(x))


def files_to_df_iterable(pattern: str, loader: Callable = lambda x: caching_query(x, None), sort_files=False):
    """
    Iterates over all files in a directory and yields the loaded data for each file.

    Args:
        pattern (str): The pattern to match files against.
        loader (Callable, optional): The function to load the data. Defaults to lambda x: caching_query(x, None).
        sort_files (bool, optional): Whether to sort the files. Defaults to False.

    Yields:
        The loaded data for each file.
    """
    files = glob.glob(pattern)
    if sort_files:
        files = sorted(files)
    for file in files:
        # print(file)
        yield (loader(file))
