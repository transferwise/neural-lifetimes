from functools import lru_cache

import pandas as pd
from sqlalchemy import create_engine

@lru_cache(1024)  # caches results for speed
def dtypes_from_table(
    user: str,
    password: str,
    host: str,
    database: str,
    table: str = None,
    query: str = None,
    port=5432
):

    ch_url = f'postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}'
    ch_engine = create_engine(ch_url, echo=False)

    if query is None:
        query = f"select *  from {database}.{table} limit 1"
    else:
        query += " limit 1"

    df = pd.read_sql(query, ch_engine)
    return df.dtypes
