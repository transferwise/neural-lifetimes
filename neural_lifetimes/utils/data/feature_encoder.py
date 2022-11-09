from typing import Dict, List, Any, Optional, Union

import numpy as np
import torch

from .encoder_with_unknown import OrdinalEncoderWithUnknown


class FeatureDictionaryEncoder:
    def __init__(
        self,
        continuous_features: List[str],
        discrete_features: Dict[str, np.ndarray],
        pre_encoded: bool = False,
        start_token_discrete: Optional[str] = None,
    ) -> None:
        """A Combined encoder for dictionary-based batches.

        This Encoder applies an ``utils.data.OrdinalEncoderWithUnknown`` discrete features and
        the identity function to continuous features in a dictionary. It further, converts values to normalised types
        for Pytorch training, i.e. `int64` for discrete and `float32` for continuous features.

        Args:
            continuous_features (List[str]): The names of items to be treated as continuous features.
            discrete_features (Dict[str, np.ndarray]): A dictionary of discrete features with their names as keys and
                levels as value.
            pre_encoded (bool, optional): If features are loaded pre-encoded, set this to ``True``.
                This will skip encoding while still enabling decoding. Defaults to False.
            start_token_discrete (Optional[str], optional): This encoder assumes that the start token of discrete
                features is part of the levels as passed into ``discrete_values``. If it isn't, specify the token here
                to append it manually. Defaults to None.
        """
        self.continuous_features = continuous_features
        self.discrete_features = discrete_features
        self.pre_encoded = pre_encoded
        self.start_token_discrete = start_token_discrete

        self.enc = {}
        for name, values in discrete_features.items():
            if start_token_discrete is not None:
                values = np.concatenate(([self.start_token_discrete], values)).astype(values.dtype)
            self.enc[name] = OrdinalEncoderWithUnknown()
            self.enc[name].fit(values)

    def feature_size(self, name: str) -> int:
        if name in self.discrete_features:
            return len(self.enc[name])
        elif name in self.continuous_features:
            return 1
        else:
            raise KeyError(f"'{name}' unknown.")

    # encode single item
    def encode(self, name: str, x: np.ndarray) -> np.ndarray:
        """Encode a single feature.

        Args:
            name (str): feature name.
            x (np.ndarray): raw feature values.

        Returns:
            np.ndarray: encoded feature values.
        """
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
    def decode(self, name: str, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Decode a single feature.

        Args:
            name (str): feature name.
            x (Union[np.ndarray, torch.Tensor]): encoded feature values.

        Returns:
            np.ndarray: decoded feature values.
        """
        if self.pre_encoded or name not in self.discrete_features:
            decoded = x
        else:
            decoded = self.enc[name].inverse_transform(x).reshape(-1)
        return decoded

    def __call__(self, x: Dict[str, Any], mode: str = "encode") -> Dict[str, Any]:
        """Transform a dictionary of features. This call can encode and decode.

        Args:
            x (Dict[str, Any]): The input features.
            mode (str, optional): The transform mode can be ``enode`` or ``decode``. Defaults to "encode".

        Returns:
            Dict[str, Any]: The transformed features.
        """
        assert mode in ["encode", "decode"]
        if mode == "encode":
            return {k: self.encode(k, v) for k, v in x.items()}
        else:
            return {k: self.decode(k, v) for k, v in x.items()}

    @property
    def features(self) -> List[str]:
        return self.continuous_features + list(self.discrete_features.keys())

    def config_dict(self) -> Dict[str, Any]:
        """Dump the encoder into a dictionary. It can be used to re-initialise the object using ``.from_dict()``.

        Returns:
            Dict[str, Any]: The Encoder config dictionary. A dictionary containing all arguments required to initialise
                the object.
        """
        return {
            "continuous_features": self.continuous_features,
            "discrete_features": self.discrete_features,
            "pre_encoded": self.pre_encoded,
            "start_token_discrete": self.start_token_discrete,
        }

    @classmethod
    def from_dict(cls, dictionary: Dict[str, Any]):
        """Initialise encoder from dictionary.

        Args:
            dictionary (Dict[str, Any]): A configuration dict as dumped by ``.config_dict()``.
        """
        assert "continuous_features" in dictionary
        assert "discrete_features" in dictionary
        assert "pre_encoded" in dictionary

        return cls(
            continuous_features=dictionary["continuous_features"],
            discrete_features=dictionary["discrete_features"],
            pre_encoded=dictionary["pre_encoded"],
            start_token_discrete=dictionary["start_token_discrete"],
        )
