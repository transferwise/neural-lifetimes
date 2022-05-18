import datetime
from typing import Union
import numpy as np
import torch


def date2time(date: datetime.date) -> datetime.datetime:
    """
    Convert a date to a datetime by adding the minimum time.

    Args:
        date (datetime.date): The date to convert.

    Returns:
        datetime.datetime: The converted date.
    """
    return datetime.datetime.combine(date, datetime.datetime.min.time())


DatetimeLike = Union[torch.tensor, np.array, datetime.datetime]
TimestampLike = Union[torch.tensor, np.array, float, int]


def datetime2float(t: DatetimeLike, unit: str = "h") -> TimestampLike:
    if isinstance(t, datetime.datetime):
        t = t.timestamp() * 1e6
    elif isinstance(t, np.ndarray):
        t = t.astype("datetime64[us]").astype(np.int64).astype(np.float32)
    elif isinstance(t, torch.tensor):
        raise NotImplementedError
    else:
        raise TypeError(f"Datetime must be of type 'DatetimeLike'. Got '{type(t).__name__}'")
    # time is in microseconds
    t = t / (1e6 * 60 * 60)
    # time is in hours
    return t


def float2datetime(t: TimestampLike, unit: str = "h"):
    t = t * (1e6 * 60 * 60)
    if isinstance(t, (float, int)):
        t = datetime.datetime.fromtimestamp(t / 1e6)
    elif isinstance(t, np.ndarray):
        t = t.astype("datetime64[us]")
    elif isinstance(t, torch.tensor):
        raise NotImplementedError
    else:
        raise TypeError(f"Datetime must be of type 'DatetimeLike'. Got '{type(t).__name__}'")

    return t
