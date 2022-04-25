from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from neural_lifetimes.utils.encoder_with_unknown import OrdinalEncoderWithUnknown


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
        category_dict: Dict[str, List],
        embed_dim: int,
        drop_rate: float = 0.0,
        pre_encoded: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.drop_rate = drop_rate
        self.continuous_features = continuous_features

        # create the continuous feature encoding, with one hidden layer for good measure
        num_cf = len(self.continuous_features)
        self.c1 = nn.Linear(num_cf, 2 * num_cf)
        self.c2 = nn.Linear(2 * num_cf, embed_dim)

        # create the discrete feature encoding
        self.enc = {}
        self.emb = nn.ModuleDict()
        for name, values in category_dict.items():
            self.enc[name] = OrdinalEncoderWithUnknown()
            self.enc[name].fit(values)
            # need to remember the Unknown value
            self.emb[name] = nn.Embedding(len(self.enc[name].categories_[0]) + 1, embed_dim)

        self.output_shape = [None, embed_dim]
        self.pre_encoded = pre_encoded

    def build_parameter_dict(self) -> Dict[str, Any]:
        """Return a dict of parameters.

        Returns:
            Dict[str, Any]: Parameters of the embedder
        """
        return {
            "embed_dim": self.embed_dim,
            "embedder_drop_rate": self.drop_rate,
            "pre_encoded_features": self.pre_encoded,
        }

    def encode(self, name: str, x: np.ndarray) -> torch.Tensor:
        device = self.emb[name].weight.device
        if self.pre_encoded:
            encoded = x
        else:
            encoded = self.enc[name].transform(x).reshape(-1)

        if not isinstance(encoded, torch.Tensor):
            encoded = torch.tensor(encoded, dtype=torch.long, device=device)
        return encoded

    def forward(self, x: Dict[str, torch.Tensor]):
        # batch x num_cont_features
        cf = torch.stack([x[f] for f in self.continuous_features], dim=1)
        cf[cf.isnan()] = 0

        out = F.dropout(F.relu(self.c1(cf)), self.drop_rate, self.training)
        # batch x embed_dim
        out = F.dropout(F.relu(self.c2(out)), self.drop_rate, self.training)
        assert not torch.isnan(out.sum())

        for name, enc in self.enc.items():
            encoded = self.encode(name, x[name])
            assert not torch.isnan(encoded.sum())
            out += F.dropout(self.emb[name](encoded))
            assert not torch.isnan(out.sum())

        return out
