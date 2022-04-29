from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch import nn


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
        continuous_features: List[str],
        discrete_features: Dict[str, List],
        embed_dim: int,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.drop_rate = drop_rate
        self.continuous_features = continuous_features
        self.discrete_features = discrete_features

        # create the continuous feature encoding, with one hidden layer for good measure
        num_cf = len(self.continuous_features)
        self.c1 = nn.Linear(num_cf, 2 * num_cf)
        self.c2 = nn.Linear(2 * num_cf, embed_dim)
        self.combine = nn.Linear(len(self.discrete_features) + 1, 1)  # + 1 for the continuous features

        # create the discrete feature encoding
        self.emb = nn.ModuleDict()
        for name in discrete_features:
            # need to remember the Unknown value
            self.emb[name] = nn.Embedding(len(discrete_features[name]) + 1, embed_dim)  # TODO use encoder dimensions

        self.output_shape = [None, embed_dim]

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
        # batch x num_cont_features
        cf = torch.stack([x[f] for f in self.continuous_features], dim=1)
        cf[cf.isnan()] = 0

        out = []
        hidden = F.dropout(F.relu(self.c1(cf)), self.drop_rate, self.training)
        # batch x embed_dim
        out.append(F.dropout(F.relu(self.c2(hidden)), self.drop_rate, self.training))

        for name in self.discrete_features:
            out.append(F.dropout(self.emb[name](x[name])))

        out = torch.stack(out, dim=-1)
        out = self.combine(out).squeeze()

        assert not torch.isnan(out.sum())

        return out
