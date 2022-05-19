import datetime
from typing import Union
import numpy as np


def date2time(date: datetime.date) -> datetime.datetime:
    """
    Convert a date to a datetime by adding the minimum time.

    Args:
        date (datetime.date): The date to convert.

    Returns:
        datetime.datetime: The converted date.
    """
    return datetime.datetime.combine(date, datetime.datetime.min.time())


DatetimeLike = Union[np.array, datetime.datetime]
TimestampLike = Union[np.array, float, int]

_conversion_factors = {
    "d": 1e6 * 60 * 60 * 24,
    "h": 1e6 * 60 * 60,
    "min": 1e6 * 60,
    "s": 1e6,
    "ms": 1000,
}

# TODO: Implement lists, test different timescales, implement torch


def datetime2float(t: DatetimeLike, unit: str = "h") -> TimestampLike:
    if unit not in _conversion_factors:
        raise ValueError(f"Unit parameter '{unit}' not supported.")
    if isinstance(t, datetime.datetime):
        t = t.timestamp() * 1e6
    elif isinstance(t, np.ndarray):
        t = t.astype("datetime64[us]").astype(np.int64).astype(np.float32)
    else:
        raise TypeError(f"Datetime must be of type 'DatetimeLike'. Got '{type(t).__name__}'")
    # time is in microseconds
    t = t / _conversion_factors[unit]
    # time is in hours
    return t


def float2datetime(t: TimestampLike, unit: str = "h"):
    if unit not in _conversion_factors:
        raise ValueError(f"Unit parameter '{unit}' not supported.")
    t = t * _conversion_factors[unit]
    if isinstance(t, (float, int)):
        t = datetime.datetime.fromtimestamp(t / 1e6)
    elif isinstance(t, np.ndarray):
        t = t.astype("datetime64[us]")
    else:
        raise TypeError(f"Datetime must be of type 'DatetimeLike'. Got '{type(t).__name__}'")

    return t
