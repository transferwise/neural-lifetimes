from typing import List, Dict, Any

import numpy as np


class DataModel:
    def __init__(
        self,
        continuous_features: List[str],
        discrete_features: Dict[str, List[np.ndarray]],
        start_tokens: Dict[str, Any],
        targets: List[str],
    ) -> None:
        self.continuous_features = continuous_features
        self.discrete_features = discrete_features
        self.start_tokens = start_tokens
        self.targets = targets

    @property
    def columns(self) -> List[str]:
        return self.continuous_features + list(self.discrete_features.keys()) + self.targets

    def load_from_json(cls, path):
        pass

    def load_from_clickhouse(cls, path):
        pass

    def dump_to_file(self, path):
        pass

    def to_dict(self):
        return self.__dict__
