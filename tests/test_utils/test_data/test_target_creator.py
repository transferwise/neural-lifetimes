import numpy as np

from neural_lifetimes.utils.data import TargetCreator


def test_constructor():
    TargetCreator(cols=["CF1", "DF1"])


def test_parameter_dict():
    target_creator = TargetCreator(cols=["CF1", "DF1"])
    assert target_creator.build_parameter_dict() == {"columns": "['CF1', 'DF1']"}


def test_call():
    target_creator = TargetCreator(cols=["CF1", "DF1"])
    data = {
        "t": np.array([1, 3, 4]),
        "dt": np.array([0, 2, 1]),
        "CF1": np.array([1, 2, 3]),
        "DF1": np.array([1, 2, 3]),
    }
    expected = {
        "next_dt": np.array([2, 1]),
        "next_CF1": np.array([2, 3]),
        "next_DF1": np.array([2, 3]),
        **data,
    }

    transformed_data = target_creator(data)

    for k, v in transformed_data.items():
        assert np.all(expected[k] == v)
