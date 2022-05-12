from datetime import datetime
from typing import List, Dict

from dataclasses import dataclass

import numpy as np


@dataclass
class Tokenizer:
    """A callable class to tokenize your dictionary-based batches.

    The Tokenizer left-truncates sequences and appends start tokens.

    Args:
        continuous_features (List[str]): A list containing the names of the continuous features.
        discrete_features (List[str]): A list containing the names of the discrete features.
        max_item_len (int): The maximum length to which the sequence should be truncated.
            The tokenizer performs left-truncation. The length of returned sequences will be ``max_item_len + 1``.
        start_token_continuous (np.float32): The start token for variables specified in ``continuous_features``.
        start_token_discrete (str): The start token for variables specified in ``discrete_features``.
        start_token_timestamp (datetime.datetime): The start token for variables with data type ``np.datetime64``.
        start_token_other (np.float32): The start token for variables not specified in ``continuous_features``,
            `discrete_features`` or of type ``np.datetime64``.

    Attributes:
        continuous_features (List[str]): A list containing the names of the continuous features.
        discrete_features (List[str]): A list containing the names of the discrete features.
        features (List[str]): A list containing the names of both continuous and discrete features.
        max_item_len (int): The maximum length to which the sequence should be truncated.
            The tokenizer performs left-truncation. The length of returned sequences will be ``max_item_len + 1``.
        start_token_continuous (np.float32): The start token for variables specified in ``continuous_features``.
        start_token_discrete (str): The start token for variables specified in ``discrete_features``.
        start_token_timestamp (datetime.datetime): The start token for variables with data type ``np.datetime64``.
        start_token_other (np.float32): The start token for variables not specified in ``continuous_features``,
            `discrete_features`` or of type ``np.datetime64``.
    """

    continuous_features: List[str]
    discrete_features: List[str]
    max_item_len: int
    start_token_continuous: np.float32
    start_token_discrete: str
    start_token_timestamp: datetime
    start_token_other: np.float32

    def __call__(self, x: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Tokenizes and truncates the data ``x``.

        Args:
            x (Dict[str, np.ndarray]): The raw data dictionary.

        Returns:
            Dict[str, np.ndarray]: The transformed data.
        """
        # truncate sequence
        x = {k: v[-(self.max_item_len) :] for k, v in x.items()}

        # add start tokens
        for k, v in x.items():
            if k in self.features:
                if k in self.discrete_features:
                    x[k] = np.append([self.start_token_discrete], v)
                else:
                    x[k] = np.append([self.start_token_continuous], v)
            else:
                if v.dtype == np.datetime64:
                    x[k] = np.append(np.array([self.start_token_timestamp], dtype=np.datetime64), v)
                else:
                    x[k] = np.append([self.start_token_other], v)
        return x

    @property
    def features(self) -> List[str]:
        return self.continuous_features + list(self.discrete_features)
