from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from neural_lifetimes.utils.data import FeatureDictionaryEncoder


class CombinedEmbedder(nn.Module):
    """
    An embedder for continous and discrete features. Is a nn.Module.

    Args:
        continuous_features (List[str]): list of continuous features
        category_dict (Dict[str, List]): dictionary of discrete features
        embed_dim (int): embedding dimension
        drop_rate (float): dropout rate. Defaults to ``0.0``.
        pre_encoded (bool): whether to use the input data as is. Defaults to ``False``.
    """

    def __init__(
        self,
        feature_encoder: FeatureDictionaryEncoder,
        embed_dim: int,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.drop_rate = drop_rate
        self.encoder = feature_encoder

        # create the continuous feature encoding, with one hidden layer for good measure
        num_cf = len(self.continuous_features)
        self.c1 = nn.Linear(num_cf, 2 * num_cf)
        self.c2 = nn.Linear(2 * num_cf, embed_dim)
        self.combine = nn.Linear(len(self.discrete_features) + 1, 1)
        self.layer_norm = nn.LayerNorm(normalized_shape=embed_dim)

        # create the discrete feature encoding
        self.enc = {}
        self.emb = nn.ModuleDict()
        for name in self.discrete_features:
            self.emb[name] = nn.Embedding(self.encoder.feature_size(name), embed_dim)

        self.output_shape = [None, embed_dim]

    @property
    def continuous_features(self):
        return self.encoder.continuous_features

    @property
    def discrete_features(self):
        return self.encoder.discrete_features

    def build_parameter_dict(self) -> Dict[str, Any]:
        """Return a dict of parameters.

        Returns:
            Dict[str, Any]: Parameters of the embedder
        """
        return {
            "embed_dim": self.embed_dim,
            "embedder_drop_rate": self.drop_rate,
        }

    def forward(self, x: Dict[str, torch.Tensor]):
        combined_emb = []

        # batch x num_cont_features
        cf = torch.stack([x[f] for f in self.continuous_features], dim=1)
        cf[cf.isnan()] = 0  # TODO do not do if nan is start token

        cf_emb = F.dropout(F.relu(self.c1(cf)), self.drop_rate, self.training)
        cf_emb = torch.clip(cf_emb, -65000, 65000)
        assert not cf_emb.isnan().any(), "First Linear for continuous features contains `NaN` values."

        # batch x embed_dim
        cf_emb = F.dropout(F.relu(self.c2(cf_emb)), self.drop_rate, self.training)
        assert not cf_emb.isnan().any(), "Second Linear for continuous features contains `NaN` values."
        combined_emb.append(cf_emb)

        # out = torch.clip(out, -65000, 65000)
        for name in self.discrete_features:
            disc_emb = F.dropout(self.emb[name](x[name]))
            assert not disc_emb.isnan().any(), f"Embedding for discrete feature '{name}' contains `NaN` values."
            combined_emb.append(disc_emb)

        combined_emb = torch.stack(combined_emb, dim=-1)
        out = self.combine(combined_emb).squeeze()
        assert not out.isnan().any(), "Combined Feature Embeddings contain `NaN` values."

        out = self.layer_norm(out)  # TODO try removing this again once all features are properly normalized
        assert not out.isnan().any(), "Normalized Embeddings contain `NaN` values."
        return out
