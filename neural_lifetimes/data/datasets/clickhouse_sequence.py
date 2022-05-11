import datetime
from typing import Dict, Optional, Sequence

import numpy as np
from clickhouse_driver import Client

from ...utils.clickhouse.schema import dtypes_from_table
from .sequence_dataset import SequenceDataset


class ClickhouseSequenceDataset(SequenceDataset):
    def __init__(
        self,
        host: str,
        port: int,
        http_port: int,
        database: str,
        table_name: str,
        uid_name: str,
        time_col: str,
        asof_time: datetime.datetime,  # return no records after this
        min_items_per_uid: int = 1,
        limit: Optional[int] = None,
    ):
        self.host = host
        self.port = port
        self.http_port = http_port
        self.conn = Client(host=host, port=port)
        self.database = database
        self.table_name = table_name
        self.uid_name = uid_name
        self.time_col = time_col
        self.asof_time = asof_time
        self.limit = limit

        # get all the UIDs with at least min_items_per_uid events
        date_filt = self.asof_filter.replace("and", "where") if self.asof_filter else ""

        uid_query = f"""SELECT * from (
        SELECT {uid_name},
            count(*) as cnt,
            min({time_col}) as first_t,
            min({time_col}) as last_t
        from {database}.{table_name}
        {date_filt}
        group by {uid_name}
        order by {uid_name}
        ) as tmp
        where cnt >={min_items_per_uid}
        """
        if self.limit:
            uid_query += f" LIMIT {self.limit}"

        result = np.array(self.conn.execute(uid_query))
        # ordered list of all the ids we're considering

        self.all_uids = result[:, 0]
        # self.first_t = result[:, 2]
        # self.last_t = result[:, 3]

        # number of events for each ID
        self.all_uid_sizes = {x[0]: x[1] for x in result}

        self.uids = np.copy(self.all_uids)
        self.uid_sizes = self.all_uid_sizes.copy()

    def __getstate__(self):
        out = self.__dict__.copy()
        del out["conn"]
        return out

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.conn = Client(self.host)

    @property
    def asof_filter(self):
        return "" if self.asof_time is None else f" and {self.time_col} < toDateTime64('{self.asof_time}',3)"

    def uids_filter(self, new_uids):
        if isinstance(new_uids, list):
            new_uids = np.array(new_uids)

        new_uids = [id for id in new_uids if id in self.uids]
        self.uids = new_uids
        self.uid_sizes = {id: self.all_uid_sizes[id] for id in new_uids}

        return

    def __len__(self):
        return len(self.uids)

    def get_seq_len(self, i: int) -> int:
        return self.uid_sizes[self.uids[i]]

    def _load_batch(self, inds: Sequence[int]) -> Sequence[Dict[str, np.ndarray]]:
        # get sequences for a list of UIDS,
        # so we call the database only once
        uids = np.array(sorted([self.uids[i] for i in inds]))
        # get the data for all the ids
        query = f"""
            SELECT * from {self.database}.{self.table_name}
            where {self.uid_name} in ({','.join(uids.astype(str))})
            {self.asof_filter}
            order by {self.uid_name}, {self.time_col}
        """
        raw_data = self.conn.execute(query)
        data = np.array(raw_data).T
        pre_out = {}

        # get the data types and column names
        dtypes = dtypes_from_table(self.host, self.database, table=self.table_name, port=self.port)

        # match data to column names and cast the variables to correct types
        for x, (cname, ctype) in zip(data, dtypes.iteritems()):
            pre_out[cname] = x.astype(ctype)
            if cname == self.time_col:
                pre_out["t"] = x

        # slice it up by ID and apply the transform
        seqs = []
        offsets = [0]
        # split the query result into sequences by uid
        for next_item in uids:
            len_ = self.uid_sizes[next_item]
            offsets.append(offsets[-1] + len_)
            # split out the data for a particular ID
            this_seq = {k: v[offsets[-2] : offsets[-1]] for k, v in pre_out.items()}
            seqs.append(this_seq)

        return seqs
