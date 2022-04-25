from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
import pytest
import vaex

from neural_lifetimes.data.datasets import PandasSequenceDataset

TIME_COL = "transaction_time"
UID_COL = "uid"


# TODO xfail until PandasSequenceDataset fixes
pytestmark = pytest.mark.xfail


class DataframeType(Enum):
    pandas = "pandas"
    vaex = "vaex"


@pytest.fixture
def start_date():
    return datetime(2020, 1, 1)


@pytest.fixture
def uids():
    return list(range(1, 11))


@pytest.fixture
def df(start_date, uids):
    rows = []
    for uid in uids:
        # Each user will get the same number of transactions as their uid int
        for i in range(uid):
            rows.append(
                {
                    TIME_COL: start_date + timedelta(days=i),
                    UID_COL: uid,
                }
            )
    return pd.DataFrame(rows).sort_values(by=[TIME_COL])


def _construct_dataset(
    df: pd.DataFrame,
    asof_time: Optional[datetime] = None,
    min_items_per_uid: int = 1,
    df_type: DataframeType = DataframeType.pandas,
) -> PandasSequenceDataset:
    if df_type == DataframeType.vaex:
        df = vaex.from_pandas(df)
    return PandasSequenceDataset(
        df,
        UID_COL,
        TIME_COL,
        asof_time=asof_time,
        min_items_per_uid=min_items_per_uid,
    )


class TestConstruction:
    @pytest.mark.parametrize("df_type", DataframeType)
    def test_sets_static_data(self, df, df_type):
        dataset = _construct_dataset(df, df_type=df_type)

        assert dataset.id_column == UID_COL
        assert dataset.time_col == TIME_COL

    @pytest.mark.parametrize("df_type", DataframeType)
    def test_filters_by_min_items_per_uid(self, df, uids, df_type):
        # We should lose uids 1 and 2
        dataset = _construct_dataset(df, min_items_per_uid=3, df_type=df_type)

        unique_uids = dataset.df[UID_COL].unique()
        expected = set(uids) - {1, 2}

        assert set(unique_uids) == expected
        assert len(dataset.ids) == len(expected)
        assert set(dataset.ids) == expected

    @pytest.mark.parametrize("df_type", DataframeType)
    def test_filters_by_as_of_time(self, df, start_date, uids, df_type):
        dataset = _construct_dataset(df, asof_time=start_date, df_type=df_type)

        unique_dates = dataset.df[TIME_COL].unique()
        unique_uids = dataset.df[UID_COL].unique()

        assert len(dataset.df) == len(uids)

        assert len(unique_dates) == 1
        assert unique_dates[0] == np.datetime64(start_date)

        # Should now have one transaction per uid
        assert len(unique_uids) == len(uids)
        assert set(unique_uids) == set(uids)

    @pytest.mark.parametrize("df_type", DataframeType)
    def test_len(self, df, uids, df_type):
        dataset = _construct_dataset(df, df_type=df_type)
        result = len(dataset)

        assert result == len(set(uids))


class TestGetItem:
    @pytest.mark.parametrize("df_type", DataframeType)
    def test_returns_length_of_sequence(self, df, df_type):
        uid = 5
        sequence_length = uid  # As defined in the fixtures
        dataset = _construct_dataset(df, df_type=df_type)

        result = dataset.get_seq_len(uid - 1)

        assert result == sequence_length


class TestGetBulk:
    @pytest.mark.parametrize("df_type", DataframeType)
    def test_gets_transactions_for_uid_sequence(self, df, df_type):
        uids = np.array([4, 8])
        indices = uids - np.ones_like(uids)
        dataset = _construct_dataset(df, df_type=df_type)

        result = dataset[list(indices)]

        assert len(result) == len(uids)

        for uid, sequence in zip(uids, result):
            assert len(sequence["t"]) == uid  # Sequence length == uids from the fixtures

    @pytest.mark.parametrize("df_type", DataframeType)
    def test_works_for_a_single_transaction(self, df, df_type):
        uid = 1
        index = uid - 1
        dataset = _construct_dataset(df, df_type=df_type)

        result = dataset[index]

        assert len(result) == 1

        sequence = result[0]
        assert len(sequence["t"]) == 1  # Sequence length == uids from the fixtures
