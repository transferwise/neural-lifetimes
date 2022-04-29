import numpy as np
from sklearn.preprocessing import OrdinalEncoder


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
        # wrap the import into try-except so the code works even when no Pytorch installed
        from torch import Tensor

        if isinstance(x, Tensor):
            x = x.detach().cpu().numpy()
    except Exception:
        pass

    x = np.array(x)  # leads to error with GPU
    if len(x.shape) == 1:
        x = x[:, None]

    assert x.shape[1] == 1  # only handle one encoding at a time
    return x


class OrdinalEncoderWithUnknown(OrdinalEncoder):
    """An ordinal encoder that encodes unknown values as 0."""

    # uses 0 to encode unknown values
    def transform(self, x):
        x = normalize(x)
        out = np.zeros(x.shape).astype(int)
        # The below was the old implementation
        # known = np.array([xx[0] in self.categories_[0] for xx in x])
        # this should give identical results but faster
        known = np.isin(x, self.categories_[0]).reshape(-1)
        if any(known):
            out[known] = super(OrdinalEncoderWithUnknown, self).transform(np.array(x)[known]) + 1
        return out

    def fit(self, x):
        x = normalize(x)
        return super().fit(x)

    def __len__(self):
        return len(self.categories_[0]) + 1
