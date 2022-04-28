from datetime import datetime
import pytest

from neural_lifetimes.data.datasets import ClickhouseSequenceDataset

TIME_COL = "transaction_time"
UID_COL = "uid"


@pytest.fixture
def start_date():
    return datetime(2020, 1, 1)


@pytest.fixture
def uids():
    return list(range(1, 11))


def _construct_dataset():
    return ClickhouseSequenceDataset()


class TestConstruction:
    @pytest.mark.xfail
    def test_sets_static_data(self):
        assert False

    @pytest.mark.xfail
    def test_filters_by_min_items_per_uid(self):
        assert False

    @pytest.mark.xfail
    def test_filters_by_as_of_time(self):
        assert False

    @pytest.mark.xfail
    def test_len(self):
        assert False


class TestGetItem:
    @pytest.mark.xfail
    def test_item_getter_int(self):
        assert False

    @pytest.mark.xfail
    def test_item_getter_list(self):
        assert False
