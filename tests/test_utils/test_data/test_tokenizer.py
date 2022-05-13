from datetime import datetime
from typing import Dict
import numpy as np

import pytest

from neural_lifetimes.utils.data import Tokenizer


@pytest.fixture
def data():
    return {
        "CF1": np.array([1, 2, 3, 4]),
        "CF2": np.array([4, 3, 2, 1]),
        "Other_CF": np.array([0, 1, 0, 1]),
        "DF1": np.array(["level_1", "level_1", "level_2", "level_3"]),
        "Other_DF": np.array(["level_1", "level_1", "level_2", "level_3"]),
        "dates": np.array(
            [datetime(2022, 1, 1), datetime(2022, 2, 1), datetime(2022, 3, 1), datetime(2022, 1, 1)],
            dtype=np.datetime64,
        ),
    }


@pytest.fixture
def tokenizer():
    return Tokenizer(
        continuous_features=["CF1", "CF2"],
        discrete_features=["DF1"],
        max_item_len=2,
        start_token_continuous=float("inf"),
        start_token_discrete="BigBang",
        start_token_other=-1,
        start_token_timestamp=datetime(1970, 1, 1, 0, 0, 0, 0),
    )


def test_constructor(tokenizer):
    assert tokenizer


def test_call(data: Dict[str, np.ndarray], tokenizer: Tokenizer):
    tokenized = tokenizer(data)

    expected = {
        "CF1": np.array([np.inf, 3, 4]),
        "CF2": np.array([np.inf, 2, 1]),
        "Other_CF": np.array([-1, 0, 1]),
        "DF1": np.array(["BigBang", "level_2", "level_3"]),
        "Other_DF": np.array([-1, "level_2", "level_3"]),
        "dates": np.array(
            [datetime(1970, 1, 1, 0, 0, 0, 0), datetime(2022, 3, 1), datetime(2022, 1, 1)],
            dtype=np.datetime64,
        ),
    }

    for k in tokenized.keys():
        assert np.all(tokenized[k] == expected[k])


def test_features(tokenizer: Tokenizer):
    feat = tokenizer.features
    assert feat == ["CF1", "CF2", "DF1"]
