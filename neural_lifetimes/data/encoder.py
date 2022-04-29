from typing import Dict, List, Any

import numpy as np
import torch

from neural_lifetimes.utils import OrdinalEncoderWithUnknown


# TODO if we do not use continuous_features, dont store it in here
class CombinedFeatureEncoder:
    def __init__(
        self, continuous_features: List[str], discrete_features: Dict[str, List[Any]], pre_encoded: bool = False
    ) -> None:
        self.continuous_features = continuous_features
        self.discrete_features = discrete_features
        self.pre_encoded = pre_encoded
        self.enc = {}
        for name, values in discrete_features.items():
            self.enc[name] = OrdinalEncoderWithUnknown()
            self.enc[name].fit(values)

    def feature_size(self, name: str) -> torch.Tensor:
        return len(self.enc[name].categories_[0])

    # encode single item
    def encode(self, name: str, x: np.ndarray) -> torch.Tensor:
        if self.pre_encoded or name not in self.discrete_features:
            encoded = x
        else:
            encoded = self.enc[name].transform(x).reshape(-1)
        return encoded

    # decode single item
    def decode(self, name: str, x: np.ndarray) -> torch.Tensor:
        if self.pre_encoded or name not in self.discrete_features:
            decoded = x
        else:
            decoded = self.enc[name].inverse_transform(x).reshape(-1)
        return decoded

    def __call__(self, x: Dict[str, Any], mode: str = "encode") -> Dict[str, Any]:
        assert mode in ["encode", "decode"]
        if mode == "encode":
            return {k: self.encode(k, v) for k, v in x.items()}
        else:
            return {k: self.decode(k, v) for k, v in x.items()}

    @property
    def features(self):
        return list(self.discrete_features.keys()) + self.continuous_features
