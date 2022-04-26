import datetime


def date2time(date: datetime.date) -> datetime.datetime:
    """
    Convert a date to a datetime by adding the minimum time.

    Args:
        date (datetime.date): The date to convert.

    Returns:
        datetime.datetime: The converted date.
    """
    return datetime.datetime.combine(date, datetime.datetime.min.time())
