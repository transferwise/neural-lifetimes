from typing import Union
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


class OrdinalEncoderWithUnknown(OrdinalEncoder):
    """An ordinal encoder that encodes unknown values as 0."""

    # uses 0 to encode unknown values
    def transform(self, x: np.ndarray) -> np.ndarray:
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
        x = normalize(x)
        return super().fit(x)

    def __len__(self) -> int:
        return len(self.categories_[0]) + 1

    def inverse_transform(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
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
