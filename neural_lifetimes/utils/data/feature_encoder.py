from typing import Dict, List, Any

import numpy as np
import torch

from .encoder_with_unknown import OrdinalEncoderWithUnknown


class FeatureDictionaryEncoder:
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

    def feature_size(self, name: str) -> int:
        return len(self.enc[name])

    # encode single item
    def encode(self, name: str, x: np.ndarray) -> np.ndarray:

        # if discrete features are not pre-encoded, encode them
        if not self.pre_encoded and name in self.discrete_features:
            x = self.enc[name].transform(x).reshape(-1)

        # change types for continuous and discrete features for smooth torch conversion
        if name in self.continuous_features:
            encoded = x.astype(np.float32)
        elif name in self.discrete_features:
            encoded = x.astype(np.int64)
        else:
            encoded = x  # for non-numeric data. e.g. user profile ID

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

    def config_dict(self):
        return {
            "continuous_features": self.continuous_features,
            "discrete_features": self.discrete_features,
            "pre_encoded": self.pre_encoded,
        }

    @classmethod
    def from_dict(cls, dictionary):
        assert "continuous_features" in dictionary
        assert "discrete_features" in dictionary
        assert "pre_encoded" in dictionary

        return cls(
            continuous_features=dictionary["continuous_features"],
            discrete_features=dictionary["discrete_features"],
            pre_encoded=dictionary["pre_encoded"],
        )
