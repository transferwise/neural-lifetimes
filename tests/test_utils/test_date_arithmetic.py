from datetime import datetime, timedelta

import numpy as np
import torch

import pytest
from neural_lifetimes.utils.date_arithmetic import datetime2float, float2datetime


@pytest.fixture
def datetimes():
    return [datetime(2000 + i, i, 2 * i) for i in range(1, 13)]


@pytest.fixture
def timestamps():
    return [
        271776.0,
        281328.0,
        290808.0,
        300383.0,
        309911.0,
        319463.0,
        328991.0,
        338567.0,
        348119.0,
        357647.0,
        367200.0,
        376752.0,
    ]


# tolerance in hours
@pytest.fixture
def error_tolerance():
    return 2


class Test_Base:
    @staticmethod
    def test_datetime2float(datetimes, timestamps, error_tolerance):
        res = [datetime2float(dt) for dt in datetimes]
        assert all([(r - t) < error_tolerance for r, t in zip(res, timestamps)])

    @staticmethod
    def test_float2datetime(datetimes, timestamps, error_tolerance):
        res = [float2datetime(ts) for ts in timestamps]
        assert all([(r - t) < timedelta(hours=error_tolerance) for r, t in zip(res, datetimes)])

    @staticmethod
    def test_both(datetimes, error_tolerance):
        res = [float2datetime(datetime2float(dt)) for dt in datetimes]
        assert all([(r - t) < timedelta(hours=error_tolerance) for r, t in zip(res, datetimes)])


class Test_Numpy:
    @staticmethod
    def test_datetime2float(datetimes, timestamps, error_tolerance):
        datetimes = np.array(datetimes, dtype="datetime64[us]")
        timestamps = np.array(timestamps)
        res = datetime2float(datetimes)
        assert np.all(np.isclose(res, timestamps, rtol=0, atol=error_tolerance))

    @staticmethod
    def test_float2datetime(datetimes, timestamps, error_tolerance):
        datetimes = np.array(datetimes, dtype="datetime64[us]")
        timestamps = np.array(timestamps)
        res = float2datetime(timestamps)
        assert np.all((res - datetimes) < timedelta(hours=error_tolerance))

    @staticmethod
    def test_both(datetimes, timestamps, error_tolerance):
        datetimes = np.array(datetimes, dtype="datetime64[us]")
        timestamps = np.array(timestamps)
        res = float2datetime(datetime2float(datetimes))
        assert np.all((res - datetimes) < timedelta(hours=error_tolerance))


class Test_AcrossTypes:
    @staticmethod
    def test_numpy_base(datetimes, error_tolerance):
        res_base = np.array([datetime2float(dt) for dt in datetimes])
        np_datetimes = np.array(datetimes, dtype="datetime64[us]")
        res_np = datetime2float(np_datetimes)
        assert np.all(np.isclose(res_base, res_np, rtol=0, atol=error_tolerance))

    @staticmethod
    @pytest.mark.xfail
    def test_numpy_torch():
        assert False
