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

    def test_levels(self):
        encoder = self._construct_and_fit()
        assert np.all(encoder.levels == np.array(["<Unknow", "level_1", "level_2", "level_3"]))

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

    def test_dict(self):
        encoder = self._construct_and_fit()
        expected = {"<Unknow": 0, "level_1": 1, "level_2": 2, "level_3": 3}
        encoder_dict = encoder.to_dict()

        assert len(expected) == len(encoder_dict)
        for raw, encoded in encoder_dict.items():
            assert expected[raw] == encoded


class Test_FeatureDictionaryEncoder:
    def _construct(self, discrete_values, pre_encoded=False, start_token=None):
        return FeatureDictionaryEncoder(
            continuous_features=["CF1", "CF2"],
            discrete_features=discrete_values,
            pre_encoded=pre_encoded,
            start_token_discrete=start_token,
        )

    @pytest.mark.parametrize("pre_encoded", (True, False))
    @pytest.mark.parametrize("start_token", (None, "ST"))
    def test_constructor(self, discrete_values, pre_encoded, start_token):
        encoder = self._construct(discrete_values, pre_encoded, start_token)
        if start_token is None:
            assert "ST" not in encoder.enc["DF1"].levels
        else:
            assert "ST" in encoder.enc["DF1"].levels

    def test_pre_encoded(self, discrete_values):
        encoder = self._construct(discrete_values, True)
        data = np.array([0, 1, 2, 3])
        encoded = encoder.encode("DF1", data)
        decoded = encoder.decode("DF1", encoded)

        assert np.all(encoded == data)
        assert np.all(decoded == data)

    def test_features(self, discrete_values):
        encoder = self._construct(discrete_values)
        assert encoder.features == ["CF1", "CF2", "DF1", "DF2"]

    @pytest.mark.parametrize("pre_encoded", (True, False))
    @pytest.mark.parametrize("start_token", (None, "ST"))
    def test_feature_size(self, discrete_values, pre_encoded, start_token):
        encoder = self._construct(discrete_values, pre_encoded, start_token)
        assert encoder.feature_size("DF1") == 3 + int(start_token is not None) + 1  # 3 levels + start token + unknown
        assert encoder.feature_size("CF1") == 1
        with pytest.raises(KeyError) as excinfo:
            encoder.feature_size("HELLO")
        (msg,) = excinfo.value.args
        assert msg == "'HELLO' unknown."

    def test_dump_and_construct_from_dict(self):
        assert False

    def test_encode_decode(self):
        assert False

    def test_call_encode_decode(self):
        assert False
