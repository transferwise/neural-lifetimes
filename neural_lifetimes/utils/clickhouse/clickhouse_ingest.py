import datetime
from typing import Callable, Dict, Sequence

import numpy as np
from clickhouse_driver import Client
from sqlalchemy.engine import Engine

from neural_lifetimes.data.utils import normalize_types
from neural_lifetimes.utils.aws import caching_query

from .schema import make_clickhouse_schema


def clickhouse_ingest(
    db_io,
    client: Client,
    insert_fn: Callable,
    daily_fn: Callable,
    start_date: datetime.date,
    end_date: datetime.date,
    data_dir: str,
    table_name: str,
    uid_name: str,
    time_col: str,
    high_granularity: Sequence = (),
    flush_table: bool = False,
    verbose: bool = False,
) -> None:

    # iterates over a range of dates and dumps the results into clickhouse

    assert start_date <= end_date, "Start date can not be after end date!"
    initialized = False
    this_date = start_date
    while this_date < end_date:
        fn = data_dir + f"events_{daily_fn.__name__}_{this_date}.h5"
        if verbose:
            print(fn)
        this_df = caching_query(fn, lambda: daily_fn(db_io, this_date))
        if this_df is not None:

            if not initialized:
                client.execute("CREATE DATABASE IF NOT EXISTS events")
                if flush_table:
                    client.execute("DROP TABLE IF EXISTS events.ras_slice")
                dtypes = this_df.dtypes
                schema = make_clickhouse_schema(
                    dtypes,
                    table_name,
                    (uid_name, time_col),
                    high_granularity=high_granularity,
                )
                client.execute(schema)
                initialized = True

            this_df = normalize_types(this_df)
            this_df[time_col] = this_df[time_col].dt.strftime("%Y-%m-%d %H:%M:%S")
            insert_fn(this_df)

        this_date += datetime.timedelta(days=1)

    return dtypes

def clickhouse_ranges(
    engine: Engine,
    discr_feat: Sequence[str],
    table_name: str
) -> Dict[str, np.ndarray]:

    out = {f: np.array(engine.execute(f"SELECT DISTINCT {f} from {table_name}").fetchall()) for f in discr_feat}
    return out
