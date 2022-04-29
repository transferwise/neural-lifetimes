from typing import List, Any, Dict

from dataclasses import dataclass

import numpy as np


@dataclass
class Tokenizer:
    continuous_features: List[str]
    discrete_features: List[str]
    start_token_continuous: Any = 0
    start_token_discrete: Any = None

    def __call__(self, x: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        for k, v in x.items():
            if k == "dt":
                continue
            if k in self.features:
                if k in self.discrete_features.keys():
                    x[k] = np.append([self.start_token_discr], v, dtype=np.int64)
                else:
                    x[k] = np.append([self.start_token_cont], v, dtype=np.float32)
            else:
                x[k] = np.append([None], v)
        return x

    def features(self):
        return self.continuous_features + self.discrete_features
