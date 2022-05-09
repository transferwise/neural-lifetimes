from typing import Dict, Union
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

import torch


# TODO What does this function actually do? it doesn't normalize the data
def normalize(x):
    """
    Normalize the data. Only 1 encoding is handled at a time.

    Args:
        x: The data to be normalized.

    Returns:
        np.array: The normalized data.

    Note:
        Since we are using np.array, it may lead to errors with GPUs.
    """
    try:
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    except Exception:
        pass

    x = np.array(x)  # TODO Why copy the data?
    if len(x.shape) == 1:
        x = x[:, None]  # TODO is this the same as np.expand_dims() ?

    assert x.shape[1] == 1  # only handle one encoding at a time
    return x


# TODO The encoder truncates the "<Unkown>" token when the original dtype is shorted. This could be better.
class OrdinalEncoderWithUnknown(OrdinalEncoder):
    """An ordinal encoder that encodes unknown values as 0.

    The OrdinalEncoderWithUnknown works with unknown values. If an unknown value is passed into ``transform()``,
    it will be encoded as ``0``. The ``inverse_transform`` maps ``0`` to ``<Unknown>``.
    The encoder acts as a surjective mapping.

    Attributes:
        levels (np.ndarray): The raw levels that can be decoded to. Includes the ``<Unknown>`` token.

    Basis:
        ``sklearn.preprocessing.OrdinalEncoder``
    """

    # uses 0 to encode unknown values
    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transforms the data into encoded format.

        Args:
            x (np.ndarray): The raw data.

        Returns:
            np.ndarray: The encoded data with dtype ``int64``.
        """
        x = normalize(x)
        out = np.zeros(x.shape).astype(int)
        # The below was the old implementation
        # known = np.array([xx[0] in self.categories_[0] for xx in x])
        # this should give identical results but faster
        known = np.isin(x, self.categories_[0]).reshape(-1)
        if any(known):
            out[known] = super(OrdinalEncoderWithUnknown, self).transform(np.array(x)[known]) + 1
        return out

    def fit(self, x: np.ndarray) -> None:
        """Fits the encoder.

        Args:
            x (np.ndarray): The raw data array.

        Returns:
            _type_: The encoded data array.
        """
        x = normalize(x)
        return super().fit(x)

    def __len__(self) -> int:
        return len(self.categories_[0]) + 1

    def inverse_transform(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Transforms the data into the decoded format.

        Unknown values will be decoded as "<Unknown>".

        Args:
            x (Union[np.ndarray, torch.Tensor]): The encoded data.

        Returns:
            np.ndarray: The decoded data. The dtype will match the dtype of the array past into the ``fit`` method.

        Note:
            If the string dtype passed into ``fit`` too short for "<Unkown>", this token will be truncated.
        """
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        out = np.full_like(x, "<Unknown>", dtype=self.categories_[0].dtype)
        known = x > 0
        if any(known):
            out[known] = (
                super(OrdinalEncoderWithUnknown, self)
                .inverse_transform(np.expand_dims(x[known], axis=-1) - 1)
                .reshape(-1)
            )
        return out

    @property
    def levels(self):
        return np.concatenate((np.array(["<Unknown>"]).astype(self.categories_[0].dtype), self.categories_[0]))

    def to_dict(self) -> Dict[str, int]:
        """Converts the encoder into a dictionary structure mapping raw to encoded values. Includes unknown token.

        Returns:
            Dict[str, int]: Dictionary of form ``raw: encoded``.
        """
        return {level: self.transform(np.array([level])).item() for level in self.levels}
