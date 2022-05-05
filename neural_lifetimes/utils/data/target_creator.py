from typing import Any, Dict, List, Sequence, Union

import numpy as np
import torch
from torch import nn


class DummyTransform:
    def __call__(self, x, *args, **kwargs):
        return x

    def output_len(self, input_len: int):
        return input_len


class TargetCreator(nn.Module):
    """
    A class to create targets for a sequence of events. Is an instance of nn.Module.

    Args:
        cols (List[str]): The list of columns to use as features.

    Attributes:
        cols (List[str]): The list of columns to use as features.
    """

    def __init__(
        self,
        cols: List[str],
    ):
        super().__init__()
        self.cols = cols

    def build_parameter_dict(self) -> Dict[str, Any]:
        """Return a dictionary of parameters.

        Returns:
            Dict[str, Any]: A dictionary of the target transform parameters
        """
        return {
            "columns": str(self.cols),
        }

    def __call__(
        self,
        x: Dict[str, np.ndarray],
    ) -> Dict[str, Union[np.ndarray, Sequence[str]]]:

        for c in self.cols + ["dt"]:
            x[f"next_{c}"] = x[c][1:]
            assert len(x[f"next_{c}"]) == len(x["t"]) - 1

        return x
