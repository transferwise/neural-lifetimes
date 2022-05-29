from functools import lru_cache
from typing import Sequence, Union, Dict

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

def postgres_ranges(
    engine: Engine,
    discr_feat: Sequence[str],
    table_name: str
) -> Dict[str, np.ndarray]:

    out = {f: np.array(engine.execute(f"SELECT DISTINCT {f} from {table_name}").fetchall()) for f in discr_feat}
    return out

