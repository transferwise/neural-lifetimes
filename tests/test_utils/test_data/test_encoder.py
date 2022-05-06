import numpy as np
import torch

import pytest

from neural_lifetimes.utils.data import FeatureDictionaryEncoder, OrdinalEncoderWithUnknown


@pytest.fixture
def data():
    return {
        "CF1": np.array([1, 2, 3, 4]),
        "CF2": np.array([4, 3, 2, 1]),
        "DF1": np.array(["level_1", "level_1", "level_2", "level_3"]),
        "DF2": np.array(["l1", "l1", "l2", "l3"]),
    }


@pytest.fixture
def discrete_values():
    return {
        "DF1": np.array(["level_1", "level_2", "level_3"]),
        "DF2": np.array(["l1", "l2", "l3"]),
    }


class Test_OrdinalEncoderWithUnkown:
    def _construct_and_fit(self):
        encoder = OrdinalEncoderWithUnknown()
        encoder.fit(np.array(["level_1", "level_2", "level_3"]))
        return encoder

    def test_fit(self):
        self._construct_and_fit()

    def test_len(self):
        encoder = self._construct_and_fit()
        assert len(encoder) == 4

    def test_transform_inverse_transform(self):
        encoder = self._construct_and_fit()
        data = np.array(["level_1", "level_2", "level_3"])
        reconstructed_data = encoder.inverse_transform(encoder.transform(data))
        assert np.all(data == reconstructed_data.squeeze())

    def test_inverse_transform_form_torch_tensor(self):
        encoder = self._construct_and_fit()
        data = torch.tensor([1, 2, 3])
        decoded = encoder.inverse_transform(data)
        assert np.all(decoded == np.array(["level_1", "level_2", "level_3"]))

    def test_transform_with_unknown(self):
        encoder = self._construct_and_fit()
        data = np.array(["level_1", "level_2", "level_3", "new_1", "new_2"])
        encoded = encoder.transform(data)
        reconstructed_data = encoder.inverse_transform(encoded)
        assert np.all(encoded.squeeze() == np.array([1, 2, 3, 0, 0]))
        assert np.all(reconstructed_data.squeeze() == np.array(["level_1", "level_2", "level_3", "<Unknow", "<Unknow"]))


class Test_FeatureDictionaryEncoder:
    pass
