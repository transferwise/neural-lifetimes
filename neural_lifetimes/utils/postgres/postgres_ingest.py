from functools import lru_cache
from typing import Sequence, Union, Dict

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

def postgres_ranges(engine: Engine, discr_feat: Sequence[str], table_name: str) -> Dict[str, np.ndarray]:
    out = {f: np.array(engine.execute(f"""SELECT DISTINCT "{f}" from {table_name}""").fetchall()) for f in discr_feat}
    return out

def dtypes_from_table(
    user: str,
    password: str,
    host: str,
    database: str,
    table: str = None,
    query: str = None,
    port=5432
):

    ch_url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
    ch_engine = create_engine(ch_url, echo=False)

    if query is None:
        query = f"select *  from {table} limit 1"
    else:
        query += " limit 1"

    df = pd.read_sql(query, ch_engine)
    return df.dtypes
