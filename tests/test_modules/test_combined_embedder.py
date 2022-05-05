import pytest
import torch

from neural_lifetimes.models.nets import CombinedEmbedder
from neural_lifetimes.utils.data import FeatureDictionaryEncoder


@pytest.fixture
def embed_dim():
    return 64


@pytest.fixture
def drop_rate():
    return 0.1


class TestEmbedderConstruction:
    @staticmethod
    def _get_embedder(continuous_features, discrete_features, *args, **kwargs) -> CombinedEmbedder:
        encoder = FeatureDictionaryEncoder(continuous_features, discrete_features)
        return CombinedEmbedder(encoder, *args, **kwargs)

    # TODO The embedder should be able to handle only continuous or only discrete features.
    # One currently leads to a warning.
    @pytest.mark.parametrize("continuous_features", [[], ["FEAT_1", "FEAT_2"]])
    @pytest.mark.parametrize("category_dict", [{}, {"CAT_1": [1, 4], "CAT_2": ["A", "B"]}])
    def test_constructor(self, continuous_features, category_dict, embed_dim, drop_rate):
        self._get_embedder(continuous_features, category_dict, embed_dim, drop_rate)

    # TODO: empty category_dicts should not be allowed. e.g. {'CAT_1': []}
    @pytest.mark.xfail
    @pytest.mark.parametrize("category_dict", [{"CAT_1": []}])
    def test_constructor_empty_set_cat_dict(
        self, continuous_features, category_dict, embed_dim, drop_rate, pre_encoded=True
    ):
        with pytest.raises(Exception):
            self._get_embedder(continuous_features, category_dict, embed_dim, drop_rate, pre_encoded)

    def test_parameter_dict(self, embed_dim, drop_rate):
        emb = self._get_embedder([], {}, embed_dim, drop_rate)
        pars = emb.build_parameter_dict()
        expected = {
            "embed_dim": embed_dim,
            "embedder_drop_rate": drop_rate,
        }
        assert pars == expected

    # TODO this should work. Somehow it doesnt.
    @pytest.mark.xfail
    @pytest.mark.parametrize("continuous_features", [[], ["FEAT_1", "FEAT_2"]])
    @pytest.mark.parametrize("category_dict", [[], {"CAT_1": [1, 4], "CAT_2": ["A", "B"]}])
    @pytest.mark.parametrize("pre_encoded", [True, False])
    def test_forward(self, continuous_features, category_dict, embed_dim, drop_rate, pre_encoded):
        emb = self._get_embedder(continuous_features, category_dict, embed_dim, drop_rate, pre_encoded)
        x = {
            "FEAT_1": torch.tensor([0.5, 0.1]),
            "FEAT_2": torch.tensor([3.5, 1]),
            "CAT_1": [1, 4],
            "CAT_2": ["a", "whatever"],
        }
        emb(x)
