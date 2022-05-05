from typing import List, Any, Dict

from dataclasses import dataclass

import numpy as np


@dataclass
class Tokenizer:
    continuous_features: List[str]
    discrete_features: List[str]
    max_item_len: int
    start_token_continuous: np.float32
    start_token_discrete: str
    start_token_other: np.float32

    def __call__(self, x: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:

        # truncate sequence
        x = {k: v[-(self.max_item_len) :] for k, v in x.items()}

        # add start tokens
        for k, v in x.items():
            if k in self.features:
                if k in self.discrete_features.keys():
                    x[k] = np.append([self.start_token_discrete], v)
                else:
                    x[k] = np.append([self.start_token_continuous], v)
            else:
                x[k] = np.append([self.start_token_other], v)
        return x

    @property
    def features(self):
        return self.continuous_features + list(self.discrete_features.keys())
