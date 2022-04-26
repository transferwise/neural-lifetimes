import datetime
import gzip
import logging
import os
import pickle
from typing import Callable, Optional
from uuid import uuid4

import cloudpickle
import pandas as pd

from .s3_utils import file_exists_in_s3, get_file_from_s3, save_file_to_s3


def caching_query(
    save_loc: str,
    query_fun: Callable,
    force_query: bool = False,
    cache_format: Optional[str] = None,
    df_save_fun: Optional[Callable] = None,
    df_load_fun: Optional[Callable] = None,
):
    """
    Retrieve cached data if available, query and cache otherwise.

    Args:
        save_loc (str): Location of the saved data.
        query_fun (Callable): Function to query the data
        force_query (Optional[bool]): Whether to force a query. Defaults to False.
        cache_format (Optional[str]): Format of the cache. Defaults to None. If so, it will be inferred from save_loc.
        df_save_fun (Optional[Callable]): Function to save the data. Defaults to None.
        df_load_fun (Optional[Callable]): Function to load the data. Defaults to None.
    """
    if df_save_fun is None and df_load_fun is None:
        if cache_format is None:
            ending = save_loc.split(".")[-1]
            cache_format = {
                "h5": "h5",
                "hdf5": "h5",
                "hdf": "h5",
                "csv": "csv",
                "pklz": "pklz",
            }.get(ending, "zip")
            print("using cache format", cache_format)

        if cache_format == "zip":

            def df_save_fun(df, loc):
                return df.to_pickle(loc)

            def df_load_fun(loc):
                return pd.read_pickle(loc)

        elif cache_format == "csv":

            def df_save_fun(df, loc):
                return df.to_csv(loc)

            def df_load_fun(loc):
                return pd.read_csv(loc)

        elif cache_format == "h5":

            def df_save_fun(df, loc):
                return df.to_hdf(loc, "default", complevel=5, complib="blosc:zlib")

            def df_load_fun(loc):
                return pd.read_hdf(loc, "default")

        elif cache_format == "pklz":

            def df_save_fun(df, loc):
                with gzip.open(loc, "wb") as f:
                    cloudpickle.dump(df, f)

            def df_load_fun(loc):
                with gzip.open(loc, "rb") as f:
                    df = cloudpickle.load(f)
                return df

    if save_loc[:2] == "s3":
        return caching_query_s3(save_loc, query_fun, force_query, df_save_fun, df_load_fun)

    if os.path.isfile(save_loc) and not force_query:
        try:
            df = df_load_fun(save_loc)
        except Exception:
            # other objects that are not dataframes
            with gzip.open(save_loc, "rb") as f:
                df = pickle.load(f)
    else:
        print(f"Couldn't find {save_loc}, querying...")
        df = query_fun()
        save_dir = os.path.realpath(os.path.dirname(save_loc))
        if not os.path.isdir(save_dir) and save_dir != "":
            os.mkdir(save_dir)
        if isinstance(df, pd.DataFrame):
            df_save_fun(df, save_loc)
        elif df is not None:
            # other objects that are not dataframes
            with gzip.open(save_loc, "wb") as f:
                pickle.dump(df, f)
    return df


def caching_query_s3(
    s3_url: str,
    query_fun: Callable,
    force_query=False,
    df_save_fun: Callable = lambda df, loc: df.to_pickle(loc, compression="gzip"),
    df_load_fun: Callable = lambda loc: pd.read_pickle(loc, compression="gzip"),
):
    """
    Retrieve cached data if available, query and cache otherwise.

    Args:
        s3_url (str): Location of the saved data.
        query_fun (Callable): Function to query the data
        force_query (Optional[bool]): Whether to force a query. Defaults to False.
        df_save_fun (Optional[Callable]): Function to save the data. Defaults to pickling to a gzip file.
        df_load_fun (Optional[Callable]): Function to load the data. Defaults to unpickling from a gzip file.
    """
    # generate a unique one to avoid collisions when running in parallel
    tmp_file_name = str(uuid4())

    if file_exists_in_s3(s3_url) and not force_query:
        get_file_from_s3(s3_url, tmp_file_name)
        df = df_load_fun(tmp_file_name)
        os.remove(tmp_file_name)
        logging.info(f"File {s3_url} exists, loaded it")
    else:
        print(f"Didn't find {s3_url} so calling query_fun")
        df = query_fun()
        df_save_fun(df, tmp_file_name)
        save_file_to_s3(tmp_file_name, s3_url)
        os.remove(tmp_file_name)
        logging.info(f"File {s3_url} didn't exist, created it")
    return df


def load_date_range(
    start_date: datetime.date,
    end_date: datetime.date,
    df_from_date: Callable,
    verbose: bool = False,
    aggregate_fun: Callable = pd.concat,
) -> pd.DataFrame:
    """
    Take a range of dates and a callable that gives one dataset per day, extracts and concatenates them.

    Args:
        start_date (datetime.date): Date to query from
        end_date (datetime.date): Date to query to (not included)
        df_from_date (Callable): A function that takes a date and returns a dataframe
        verbose (Optional[bool]): Whether to print progress. Defaults to False.
        aggregate_fun (Optional[Callable]): The function to apply to the list of date-wise results.
            Defaults to pd.concat.

    Returns:
        pd.DataFrame: Combination of all the date-wise results
    """
    data = []
    assert start_date <= end_date, "Start date can not be after end date!"
    this_date = start_date
    while this_date < end_date:
        if verbose:
            print(this_date)
        data.append(df_from_date(this_date))
        this_date += datetime.timedelta(days=1)
    data = aggregate_fun(data)
    return data


if __name__ == "__main__":
    bucket = "staging-model-data"
    df = pd.DataFrame([{"test": 1}])
    # df.to_hdf("tmp.h5", "default")
    # save_file_to_s3(bucket, "tmp.h5", "tmp/tmp.h5")
    # get_file_from_s3(bucket, "tmp/tmp.h5", "tmp2.h5")
    # test = pd.read_hdf("tmp2.h5", "default")
    # print(test)
    test2 = caching_query("s3://staging-model-data/tmp/tmp2.gzip", lambda: df)
    print(test2)
    print("yay!")
