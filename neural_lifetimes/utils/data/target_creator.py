from typing import Any, Dict, List, Sequence, Union

import numpy as np


class DummyTransform:
    def __call__(self, x, *args, **kwargs):
        return x

    def output_len(self, input_len: int):
        return input_len


class TargetCreator:
    """
    A class to create targets for a sequence of events.

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

    def build_parameter_dict(self) -> Dict[str, str]:
        """Return a dictionary of parameters.

        Returns:
            Dict[str, Any]: A dictionary of the target transform parameters
        """
        return {
            "columns": str(self.cols),
        }

    def __call__(self, x: Dict[str, np.ndarray]) -> Dict[str, Union[np.ndarray, Sequence[str]]]:
        """Appends the data dict ``x`` with right-shifted copies for all keys specified in ``cols`` and ``dt``.

        Args:
            x (Dict[str, np.ndarray]): data dictionary.

        Returns:
            Dict[str, Union[np.ndarray, Sequence[str]]]: The appended data dictionary.
        """
        for c in self.cols + ["dt"]:
            x[f"next_{c}"] = x[c][1:]
            assert len(x[f"next_{c}"]) == len(x["t"]) - 1

        return x
