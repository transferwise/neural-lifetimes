import math
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pytest

from neural_lifetimes.data.datasets.btyd import (
    BTYD,
    GenMode,
    NotTrackedError,
    expected_num_transactions_from_parameters_and_history,
)
from neural_lifetimes.data.datasets.sequence_dataset import SequenceDataset, SliceableDataset

from .datamodels import DataModel, EventprofilesDataModel


@pytest.fixture
def data_model():
    return EventprofilesDataModel()


@pytest.fixture
def data_dir() -> str:
    return str(Path(__file__).parents[2] / "examples")


@pytest.fixture
def dataset(data_dir, data_model):
    dataset = BTYD(
        num_customers=10,
        a=[2],
        b=[20],
        r=[1],
        alpha=[50],
        data_dir=data_dir,
        start_date=date(2021, 1, 1),
        start_limit_date=date(2021, 1, 15),
        end_date=date(2021, 1, 31),
        continuous_features=data_model.cont_feat,
        discrete_features=data_model.discr_feat,
        seq_gen_dynamic=False,
    )
    dataset[:]
    return dataset


def _fetch_data(dataset: BTYD, indices: Iterable[int]):
    indices = list(indices)
    # We currently only sample in __getitem__ but it returns only the length
    _ = [dataset[i] for i in indices]
    return dataset._load_batch(indices)


def _build_from_modes(
    modes: List[GenMode],
    data_dir: str,
    data_model: DataModel,
    ratios: Optional[List[float]] = None,
) -> BTYD:
    return BTYD.from_modes(
        num_customers=1,
        modes=modes,
        data_dir=data_dir,
        start_date=date(2021, 1, 1),
        start_limit_date=date(2021, 1, 15),
        end_date=date(2021, 1, 31),
        continuous_features=data_model.cont_feat,
        discrete_features=data_model.discr_feat,
        seq_gen_dynamic=False,
        mode_ratios=ratios,
    )


class TestInitConstructor:
    @pytest.mark.parametrize(
        "prior_params",
        (
            {"a": [2], "b": [3], "r": [4], "alpha": [5]},
            {"a": [2, 4], "b": [3, 6], "r": [4, 8], "alpha": [5, 10]},
        ),
    )
    def test_btyd_params_from_list(self, data_dir, data_model, prior_params):
        dataset = BTYD(
            num_customers=10,
            **prior_params,
            data_dir=data_dir,
            start_date=date(2021, 1, 1),
            start_limit_date=date(2021, 1, 15),
            end_date=date(2021, 1, 31),
            continuous_features=data_model.cont_feat,
            discrete_features=data_model.discr_feat,
            seq_gen_dynamic=False,
        )
        dataset[:]

    def test_btyd_params_from_atomic(self, data_dir, data_model):
        dataset = BTYD(
            num_customers=10,
            a=1,
            b=2,
            r=3,
            alpha=4,
            data_dir=data_dir,
            start_date=date(2021, 1, 1),
            start_limit_date=date(2021, 1, 15),
            end_date=date(2021, 1, 31),
            continuous_features=data_model.cont_feat,
            discrete_features=data_model.discr_feat,
            seq_gen_dynamic=False,
        )
        dataset[:]


class TestDistParamsValidation:
    @staticmethod
    def _construct_dataset(a, b, r, alpha, data_dir, data_model, mode_ratios=None) -> BTYD:
        return BTYD(
            a=a,
            b=b,
            r=r,
            alpha=alpha,
            num_customers=10,
            data_dir=data_dir,
            start_date=date(2021, 1, 1),
            start_limit_date=date(2021, 1, 15),
            end_date=date(2021, 1, 31),
            continuous_features=data_model.cont_feat,
            discrete_features=data_model.discr_feat,
            seq_gen_dynamic=False,
            mode_ratios=mode_ratios,
        )

    def test_not_all_numeric(self, data_dir, data_model):
        with pytest.raises(AssertionError):
            self._construct_dataset(
                a=0.25,
                b=0.6,
                r=np.array([0.12]),
                alpha=1.5,
                data_dir=data_dir,
                data_model=data_model,
            )

    def test_not_all_array_like(self, data_dir, data_model):
        with pytest.raises(AssertionError):
            self._construct_dataset(
                a=[0.25],
                b=0.6,
                r=[0.12],
                alpha=[1.5],
                data_dir=data_dir,
                data_model=data_model,
            )

    def test_a_is_not_all_array_like(self, data_dir, data_model):
        with pytest.raises(AssertionError) as exc_info:
            self._construct_dataset(
                a="Not what you expected",
                b=[0.6],
                r=[0.12],
                alpha=[1.5],
                data_dir=data_dir,
                data_model=data_model,
            )

        assert "should be lists" in str(exc_info.value)

    def test_validate_length_of_ratios(self, data_dir, data_model):
        with pytest.raises(AssertionError) as excinfo:
            self._construct_dataset(
                a=[0.25, 0.5],
                b=[0.6, 0.5],
                r=[0.12, 0.5],
                alpha=[1.5, 0.5],
                data_dir=data_dir,
                data_model=data_model,
                mode_ratios=[0.1, 0.2, 9],
            )
        (msg,) = excinfo.value.args
        assert (
            msg
            == "`mode_ratios` has to match the number of modes in length. Received 2 modes and 3 `mode_ratio`s instead."
        )


class TestFromModes:
    def test_unimodal_from_modes(self, data_dir, data_model):
        mode = GenMode(a=1, b=3, r=5, alpha=15)

        result = _build_from_modes(modes=[mode], data_dir=data_dir, data_model=data_model)

        assert result.a == [mode.a]
        assert result.b == [mode.b]
        assert result.r == [mode.r]
        assert result.alpha == [mode.alpha]

    def test_multimodal_from_modes(self, data_dir, data_model):
        modes = [
            GenMode(a=1, b=3, r=5, alpha=15),
            GenMode(a=5, b=2, r=4, alpha=16),
        ]

        result = _build_from_modes(modes=modes, data_dir=data_dir, data_model=data_model)

        def mode_to_param_list(modes, param):
            return [getattr(mode, param) for mode in modes]

        assert result.a == mode_to_param_list(modes, "a")
        assert result.b == mode_to_param_list(modes, "b")
        assert result.r == mode_to_param_list(modes, "r")
        assert result.alpha == mode_to_param_list(modes, "alpha")

    def test_default_uniform_ratios(self, data_dir, data_model):
        num_modes = 7
        modes = [GenMode(a=1, b=3, r=5, alpha=15)] * num_modes

        result = _build_from_modes(modes=modes, data_dir=data_dir, data_model=data_model)

        assert result.mode_ratios == [1.0 / num_modes] * num_modes

    def test_validate_length_of_ratios(self, data_dir, data_model):
        num_modes = 7
        modes = [GenMode(a=1, b=3, r=5, alpha=15)] * num_modes
        ratios = [0.5, 0.5]

        # TODO Mark: replace this with a proper exception
        with pytest.raises(AssertionError) as excinfo:
            _build_from_modes(modes=modes, data_dir=data_dir, ratios=ratios, data_model=data_model)
        (msg,) = excinfo.value.args
        assert msg == (
            f"`mode_ratios` has to match the number of modes in length. Received {num_modes} modes and {len(ratios)}"
            " `mode_ratio`s instead."
        )


class TestGetBulk:
    def test_gets_the_correct_number_of_customers_data(self, dataset):
        indices = list(range(10))
        data = _fetch_data(dataset, indices)

        assert len(data) == len(indices)

    def test_gets_the_same_number_of_entries_in_each_field(self, dataset):
        data = _fetch_data(dataset, range(3))

        lengths_equal = [all(len(field) for field in d) for d in data]
        assert all(lengths_equal)

    def test_static_sequence(self, dataset):
        dataset.seq_gen_dynamic = False
        customer_id = 5432

        first = _fetch_data(dataset, [customer_id])[0]
        second = _fetch_data(dataset, [customer_id])[0]

        assert first.keys() == second.keys()
        for field in first:
            assert (first[field] == second[field]).all(), f"Field: {field}"

    def test_dynamic_sequence(self, dataset):
        dataset.seq_gen_dynamic = True
        customer_id = 5432

        first = _fetch_data(dataset, [customer_id])[0]
        second = _fetch_data(dataset, [customer_id])[0]

        # Seed is fixed so this should be sufficient
        assert first.keys() == second.keys()
        field_is_different = [(first[field] != second[field]).any() for field in first]
        assert any(field_is_different)

    def test_multimodal_sequence_generation(self, data_dir, data_model):
        modes = [
            GenMode(a=1, b=3, r=5, alpha=15),
            GenMode(a=5, b=2, r=4, alpha=16),
        ]
        dataset = _build_from_modes(modes, data_dir, data_model)

        indices = list(range(10))
        data = _fetch_data(dataset, indices)

        assert len(data) == len(indices)
        assert all("btyd_mode" in record for record in data)
        assert all(len(record["btyd_mode"]) == len(record["t"]) for record in data)

    @pytest.mark.xfail
    def test_customer_id_not_in_cache(self, dataset: SliceableDataset):
        # TODO Mark: replace this with a custom exception or change behavior
        # The BTYD now creates data for any key requested. Is that better?
        with pytest.raises(KeyError):
            dataset._load_batch([9876])


class TestGetItem:
    def test_gets_length_of_transaction_series_in_cache(self, dataset: SequenceDataset):
        customer_id = 1234
        length = 5
        dataset.cache[customer_id] = {
            "t": np.array([datetime.now()] * length),
            "lambda": np.array([0.123] * length),
            "p": np.array([0.5] * length),
            "btyd_mode": np.zeros(0),
        }

        result = dataset.get_seq_len(customer_id)

        assert result == length

    def test_creates_new_sample_if_not_in_cache(self, dataset):
        customer_id = 765
        result = dataset.get_seq_len(customer_id)

        cached = dataset.cache[customer_id]
        expected = len(cached["t"])

        assert result == expected


tracked_attributes = [
    "truncated_sequences",
    "num_generated_sequences",
    "start_dates_from_start_date",
    "end_dates_from_end_date",
    "all_dates_from_start_date",
    "range_transaction_days",
    "num_events_per_customer",
]


class TestTracking:
    @staticmethod
    def _construct_dataset(a, b, r, alpha, data_dir, data_model, track, mode_ratios=None) -> BTYD:
        return BTYD(
            a=a,
            b=b,
            r=r,
            alpha=alpha,
            num_customers=10,
            data_dir=data_dir,
            start_date=date(2021, 1, 1),
            start_limit_date=date(2021, 1, 15),
            end_date=date(2021, 1, 31),
            continuous_features=data_model.cont_feat,
            discrete_features=data_model.discr_feat,
            seq_gen_dynamic=False,
            mode_ratios=mode_ratios,
            track_statistics=track,
        )

    @pytest.mark.parametrize("attribute", tracked_attributes)
    def test_not_tracked_error(self, data_dir, data_model, attribute):
        dataset = self._construct_dataset(
            a=0.25,
            b=0.6,
            r=0.12,
            alpha=1.5,
            data_dir=data_dir,
            data_model=data_model,
            track=False,
        )
        dataset[:]
        with pytest.raises(NotTrackedError) as excinfo:
            getattr(dataset, attribute)
        (msg,) = excinfo.value.args
        assert msg == f"`{attribute}` not tracked. Set `track_statistics=True`"

    @pytest.mark.parametrize("attribute", tracked_attributes)
    def test_properties_exist(self, data_dir, data_model, attribute):
        dataset = self._construct_dataset(
            a=0.25,
            b=0.6,
            r=0.12,
            alpha=1.5,
            data_dir=data_dir,
            data_model=data_model,
            track=True,
        )
        dataset[:]
        getattr(dataset, attribute)


class TestPlotting:
    @staticmethod
    def _construct_dataset(a, b, r, alpha, data_dir, data_model, track, mode_ratios=None) -> BTYD:
        return BTYD(
            a=a,
            b=b,
            r=r,
            alpha=alpha,
            num_customers=10,
            data_dir=data_dir,
            start_date=date(2021, 1, 1),
            start_limit_date=date(2021, 1, 15),
            end_date=date(2021, 1, 31),
            continuous_features=data_model.cont_feat,
            discrete_features=data_model.discr_feat,
            seq_gen_dynamic=False,
            mode_ratios=mode_ratios,
            track_statistics=track,
        )

    def test_zero_samples_error(self, data_dir, data_model):
        dataset = self._construct_dataset(
            a=0.25,
            b=0.6,
            r=0.12,
            alpha=1.5,
            data_dir=data_dir,
            data_model=data_model,
            track=True,
        )
        with pytest.raises(IndexError) as excinfo:
            dataset.plot_tracked_statistics()
        (msg,) = excinfo.value.args
        assert msg == "No data generated yet. Consider running `BTYD()[:]` to create samples."

    def test_returns_plot(self, data_dir, data_model):
        import plotly.graph_objects as go

        dataset = self._construct_dataset(
            a=0.25,
            b=0.6,
            r=0.12,
            alpha=1.5,
            data_dir=data_dir,
            data_model=data_model,
            track=True,
        )
        dataset[:]
        fig = dataset.plot_tracked_statistics()
        assert isinstance(fig, go.Figure)


class TestExpectedValues:
    @staticmethod
    def _construct_dataset(
        data_dir,
        data_model,
        a=[2, 5],
        b=[20, 25],
        r=[1, 2],
        alpha=[50, 100],
        track=False,
        mode_ratios=None,
    ) -> BTYD:
        dataset = BTYD(
            a=a,
            b=b,
            r=r,
            alpha=alpha,
            num_customers=10,
            data_dir=data_dir,
            start_date=date(2021, 1, 1),
            start_limit_date=date(2021, 1, 15),
            end_date=date(2021, 6, 30),
            continuous_features=data_model.cont_feat,
            discrete_features=data_model.discr_feat,
            seq_gen_dynamic=False,
            mode_ratios=mode_ratios,
            track_statistics=track,
        )
        dataset[:]
        return dataset

    @pytest.mark.parametrize(
        "prior_params",
        (
            {},  # empty
            {
                "a": [2, 5],
                "b": [20, 25],
                "r": [1, 2],
                "alpha": [50, 100],
                "time_interval": [100, 100],
            },  # multi-modal
        ),
    )
    def test_expected_num_transactions_from_priors(self, data_dir, data_model, prior_params):
        dataset = self._construct_dataset(data_dir=data_dir, data_model=data_model, track=True)
        t = prior_params["time_interval"][0] if len(prior_params) > 0 else dataset.time_interval
        # Expected: a=2, b=20, r=1, alpha=50, t=180 is 21[1-5/23*F(1,20,21,18/23)]=2.7769567381582543
        # Expected: a=2, b=20, r=1, alpha=50, t=100 is 21[1-1/3*F(1,20,21,2/3)]=1.708363406776206
        expected_given_t = {180: 2.7769567381582543, 100: 1.708363406776206}
        return_values = dataset.expected_num_transactions_from_priors(**prior_params)
        assert math.isclose(expected_given_t[t], return_values[0], rel_tol=1e-3)

    def test_expected_num_transactions_from_parameters(self, data_dir, data_model):
        dataset = self._construct_dataset(data_dir=data_dir, data_model=data_model, track=True)
        return_value = dataset.expected_num_transactions_from_parameters(0.2, 0.01)
        # expected value for p=0.2, lambda=0.01, t=180 is 5(1-exp(0.01*0.2*180)=1.5116183696448449
        assert math.isclose(return_value, 1.5116183696448449, rel_tol=1e-4)

    @pytest.mark.parametrize("params", ({}, {"a": [1, 8], "b": [5, 2]}))
    def test_expected_p_churn_from_priors(self, data_dir, data_model, params):
        a_list, b_list = [2, 5], [20, 25]
        dataset = self._construct_dataset(
            a=a_list,
            b=b_list,
            data_dir=data_dir,
            data_model=data_model,
            track=True,
        )
        if len(params) > 0:
            a_list = params["a"]
            b_list = params["b"]

        expected = [a / (a + b) for a, b in zip(a_list, b_list)]
        res = dataset.expected_p_churn_from_priors(a_list, b_list)
        for r, e in zip(res, expected):
            assert math.isclose(r, e)

    @pytest.mark.parametrize(
        "prior_params",
        (
            {},  # empty
            {
                "r": [1, 2],
                "alpha": [3, 4],
            },  # multi-modal
        ),
    )
    def test_expected_time_interval_from_priors(self, data_dir, data_model, prior_params):
        dataset = self._construct_dataset(
            data_dir=data_dir,
            data_model=data_model,
            track=True,
        )
        # expected value is 1/E[lambda] = alpha/r
        expected = 3 if prior_params else 50
        return_value = dataset.expected_time_interval_from_priors(**prior_params)
        assert math.isclose(return_value[0], expected)

    @pytest.mark.parametrize(
        "prior_params",
        (
            {},  # empty
            {
                "a": [2, 2],
                "b": [20, 20],
                "r": [1, 1],
                "alpha": [50, 50],
                "time_interval": 100,
                "asof_time": 80,
                "num_transactions": [1, 0],  # test 0 events for denominator
                "last_transaction": [60, 70],
            },  # multi-modal
        ),
    )
    def test_expected_num_transactions_from_priors_and_history(self, data_dir, data_model, prior_params):
        dataset = self._construct_dataset(data_dir=data_dir, data_model=data_model, track=True)
        return_values = dataset.expected_num_transactions_from_priors_and_history(**prior_params)
        # expected value for a=2, b=20, r=1, alpha=50, t=180, T=180, tx=29.002, x=2 is 0.6215754775615621
        if not prior_params:
            # first_customer = dataset[0]
            # num_events = len(first_customer[0]["t"])  # == 2
            # last_event = dataset.timestamp2float(first_customer[0]["t"][-1])  # == 29.00226851851852
            # time_interval = dataset.time_interval  # == 180
            # asof_time = dataset.timestamp2float(dataset.asof_time)  # == 180
            expected_values = [0.6215754775615621]
        else:
            # expected value for a=2, b=20, r=1, alpha=50, t=180, T=180, tx=29.002, x=2
            expected_values = [1.2295652725514419, 0.7202343376625906]

        for r, e in zip(return_values, expected_values):
            assert math.isclose(r, e, rel_tol=1e-3)


class TestTimeConversions:
    @staticmethod
    def _construct_dataset(data_dir, data_model) -> BTYD:
        return BTYD(
            a=[2],
            b=[3],
            r=[4],
            alpha=[5],
            num_customers=1,
            data_dir=data_dir,
            start_date=date(2019, 1, 1),
            start_limit_date=date(2019, 6, 1),
            end_date=date(2021, 1, 1),
            continuous_features=data_model.cont_feat,
            discrete_features=data_model.discr_feat,
            seq_gen_dynamic=False,
        )

    def test_timestamp2float(self, data_dir, data_model):
        dataset = self._construct_dataset(data_dir, data_model)
        time = dataset.timestamp2float(datetime(2019, 2, 4, 13, 26, 24))
        assert isinstance(time, float)
        assert time == 34.56

    def test_float2timestamp(self, data_dir, data_model):
        dataset = self._construct_dataset(data_dir, data_model)
        timestamp = dataset.float2timestamp(34.56)
        assert isinstance(timestamp, datetime)
        assert timestamp == datetime(2019, 2, 4, 13, 26, 24)

    def test_timedelta2float(self, data_dir, data_model):
        dataset = self._construct_dataset(data_dir, data_model)
        timestamp = dataset.timedelta2float(timedelta(days=34, seconds=48384))
        assert isinstance(timestamp, float)
        assert timestamp == 34.56

    def test_float2timedelta(self, data_dir, data_model):
        dataset = self._construct_dataset(data_dir, data_model)
        timestamp = dataset.float2timedelta(34.56)
        assert isinstance(timestamp, timedelta)
        assert timestamp == timedelta(days=34, seconds=48384)

    def test_time_interval(self, data_dir, data_model):
        dataset = self._construct_dataset(data_dir, data_model)
        assert isinstance(dataset.time_interval, float)
        assert dataset.time_interval == float(365 + 366)  # does not count end date


class TestFormulas:
    @staticmethod
    def test_expected_num_transactions_from_parameters_and_history():
        # expected value for a=2, b=20, r=1, alpha=50, t=180, T=180, tx=29.002, x=2 is 0.6215754775615621

        p, lambda_ = [2 / 22], [1 / 50]
        time_interval, asof_time = 180, 200
        num_transactions, last_transaction = [2], [29.002]
        (res,) = expected_num_transactions_from_parameters_and_history(
            p, lambda_, time_interval, asof_time, num_transactions, last_transaction
        )
        assert math.isclose(res, 0.7568101, rel_tol=1e-4)
