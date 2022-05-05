import datetime
from typing import Any, Dict, List, Sequence, Union

import numpy as np
import torch
from torch import nn

from .encoder_with_unknown import OrdinalEncoderWithUnknown

from neural_lifetimes.models.nets import CombinedEmbedder


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
        enc (Dict[str, OrdinalEncoderWithUnknown]): The encoders to use for each feature.
        max_item_len (int): The maximum length of the sequence. Defaults to ``100``.
        pre_encode_features (bool): Whether to pre-encode the features. Defaults to ``False``.
        start_token_discr (str): The token to use for the start of a discrete feature. Defaults to ``"StartToken"``.
        start_token_cont (int): The token to use for the start of a continuous feature. Defaults to ``-1e6``.

    Attributes:
        enc (Dict[str, OrdinalEncoderWithUnknown]): The encoders to use for each feature.
        cols (List[str]): The list of columns to use as features.
        t_name (str): The name of the time feature. Is always ``"t"``.
        max_item_len (int): The maximum length of the sequence.
        start_token_discr (str): The token to use for the start of a discrete feature.
        start_token_cont (int): The token to use for the start of a continuous feature.
    """

    def __init__(
        self,
        cols: List[str],
        emb: CombinedEmbedder,
        max_item_len: int = 100,
        start_token_discr: str = "StartToken",
        start_token_cont: int = -1e6,
    ):
        super().__init__()
        self.enc = emb.enc
        self.cols = cols
        self.t_name = "t"
        self.max_item_len = max_item_len
        self.pre_encode_features = emb.pre_encoded
        self.start_token_discr = start_token_discr
        self.start_token_cont = start_token_cont

    def build_parameter_dict(self) -> Dict[str, Any]:
        """Return a dictionary of parameters.

        Returns:
            Dict[str, Any]: A dictionary of the target transform parameters
        """
        return {
            "columns": str(self.cols),
            "time_column": self.t_name,
            "max_item_len": self.max_item_len,
            "pre_encode_features": self.pre_encode_features,
            "start_token_discrete": self.start_token_discr,
            "start_token_continuous": self.start_token_cont,
        }

    def output_len(self, input_dim: int) -> int:
        """
        Returns the length of the output of the model.

        Args:
            input_dim (int): The length of the input.

        Returns:
            int: The length of the output.
        """
        return min(self.max_item_len, input_dim)

    def datetime2tensor(self, x: np.ndarray):
        """
        Create a timestamp(seconds) in hours since epoch.

        Args:
            x (np.ndarray): The array of datetimes to convert.

        Returns:
            np.ndarray[np.float64]: int64 / float64 to maintain precision.

        Note:
            This should be converted to float32 before training using astype(np.float32)
        """
        return x.astype("datetime64[us]").astype(np.int64).astype(np.float64) / (1e6 * 60 * 60)

    def __call__(
        self,
        x: Dict[str, np.ndarray],
        asof_time: datetime.date,
    ) -> Dict[str, Union[torch.Tensor, Sequence[str]]]:

        x["t"] = self.datetime2tensor(x["t"])
        asof_time = self.datetime2tensor(np.array([asof_time]))

        # trim the too long sequences
        x = {k: v[-(self.max_item_len) :] for k, v in x.items()}

        # Add starting tokens
        # TODO The copy on 102 and 125 look useless
        x_token = x.copy()
        x_token["dt"] = np.append([self.start_token_cont, 0], x_token["t"][1:] - x_token["t"][:-1]).astype(np.float32)
        x_token["t"] = x_token["t"].astype(np.float32)

        for k, v in x_token.items():
            if k == "dt":
                continue
            if k in self.cols:
                if k in self.enc.keys():
                    x_token[k] = np.append([self.start_token_discr], v)
                else:
                    x_token[k] = np.append([self.start_token_cont], v)
            else:
                x_token[k] = np.append([None], v)

        assert len(x_token["dt"]) == len(x_token["t"])
        if len(x_token["dt"]) > 2:
            assert (
                sum(x_token["dt"][2:] < 0) == 0
            ), "We're getting negative time intervals, are you slicing by profile correctly? "

        x_out = x_token.copy()
        x_out["t_to_now"] = asof_time - x_token["t"][-1]
        for c in self.cols + ["dt"]:
            x_out[f"next_{c}"] = x_out[c][1:]

            if c in self.enc.keys():
                # if the variable is categorical and needs encoding, pre-encode the targets
                x_out[f"next_{c}"] = self.enc[c].transform(x_out[f"next_{c}"]).reshape(-1)
                if self.pre_encode_features:
                    # and encode all categorical features
                    x_out[c] = self.enc[c].transform(x_out[c]).reshape(-1)

            assert len(x_out[f"next_{c}"]) == len(x_out["t"]) - 1

        return x_out
