from ast import Call
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from ..datasets.sequence_dataset import SequenceSubset

# from ..datasets.sequence_sampling_dataset import torchify


class SequenceLoader:
    """
    Randomly samples a dataset of variable-length sequences into a batch with a maximum number of points.

    Args:
        data (Subset): Subset of SequenceSamplingDataset
        max_points (int): maximum number of points in a batch
        min_points (int): minimum number of points in a batch. Defaults to 2.
        transform (Optional[Callable]): function to transform the data. Defaults to ``None``.

    Attributes:
        data (Subset): Subset of SequenceSamplingDataset.
        max_points (int): maximum number of points in a batch.
        min_points (int): minimum number of points in a batch.
        next_ind (int): index of the next item to be returned. Initialised to 0.
        transform (Optional[Callable]): function to transform the data.
    """

    def __init__(
        self,
        data: SequenceSubset,  # Subset of SequenceSamplingDataset,
        max_points: int,
        # tokenizer: Callable,
        min_points: int = 2,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        self.data = data
        self.max_points = max_points
        self.min_points = min_points
        self.next_ind = 0
        self.transform = transform
        self.target_transform = target_transform
        # self.tokenizer = Tokenizer(
        #     continuous_features=self.transform.continuous_features, discrete_features=self.transform.discrete_features
        # )
        self.start_token_discr = 0
        self.start_token_cont = np.nan
        self.start_token_other = np.nan
        self.max_item_len = 120

    def __iter__(self):
        """Get self as iterator of sequence batches."""
        return self

    def __len__(self):
        """
        Return 1e10 as a bodge.

        See https://github.com/PyTorchLightning/pytorch-lightning/issues/323#issuecomment-540840761 for details
        """
        return 1e10  # TODO there is probably a better solution

    def __next__(self) -> torch.Tensor:
        """Return next batch of sequences.

        Returns:
            torch.Tensor: The next batch of sequences.
        """
        inds = []
        total_size = 0

        if self.next_ind >= len(self.data):
            self.next_ind = 0
            raise StopIteration

        # collect as many ids as we need up to total length of max_points
        while True:
            if self.next_ind >= len(self.data):
                break

            next_len = self.data.get_seq_len(self.next_ind)
            next_len = self.target_transform.output_len(next_len)  # cut the length if necessary

            if total_size + next_len <= self.max_points:
                if next_len >= self.min_points:
                    total_size += next_len
                    inds.append(self.next_ind)
                self.next_ind += 1
            else:
                break

        # get data for all those IDs, in one call so it's faster
        pre_out = self.data[inds]
        out = self.bulk_transform(pre_out)
        out = torchify(out)
        for val in out.values():
            assert isinstance(val, torch.Tensor)
        return out

    def bulk_transform(self, seqs: Sequence[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """
        Process bulk-extracted data by translating each sequence separately.

        Args:
            seqs (Sequence[Dict[str, np.ndarray]]): Sequence of sequences to transform.

        Returns:
            Dict[str, np.ndarray]: Transformed sequences.
        """
        # now do some processing on the bulk-extracted data
        # translate each sequence separately

        offsets = [0]  # the 0 is included for convenience in later processing
        pre_out = []
        for s in seqs:
            s = self.transform(s)

            # convet timestamps to float
            s["t"] = self.datetime2tensor(s["t"])
            asof_time = self.datetime2tensor(np.array([self.data.asof_time]))

            # trim the too long sequences
            s = {k: v[-(self.max_item_len) :] for k, v in s.items()}

            # add dt
            s["dt"] = np.append([self.start_token_cont, 0], s["t"][1:] - s["t"][:-1]).astype(np.float32)
            s["t"] = s["t"].astype(np.float32)

            # add start tokens
            # tokenize(s, self.start_token_cont, self.start_token_discr, self.transform.features,)
            for k, v in s.items():
                if k == "dt":
                    continue
                if k in self.transform.features:
                    if k in self.transform.discrete_features.keys():
                        s[k] = np.append([self.start_token_discr], v).astype(np.int64)
                    else:
                        s[k] = np.append([self.start_token_cont], v).astype(np.float32)
                else:
                    s[k] = np.append([self.start_token_other], v)

            # assert that all dts are postitive
            assert len(s["dt"]) == len(s["t"])
            if len(s["dt"]) > 2:
                assert (
                    sum(s["dt"][2:] < 0) == 0
                ), "We're getting negative time intervals, are you slicing by profile correctly? "

            s["t_to_now"] = asof_time - s["t"][-1]

            # create next_ shifted variables
            next_out = self.target_transform(s, self.data.asof_time)

            # append offset
            if next_out is not None:
                offsets.append(offsets[-1] + next_out["t"].shape[0])  # the offsets refer to the features
                pre_out.append(next_out)

        # check that the offsets match the array lengths
        for d, slice in zip(np.diff(np.array(offsets)), pre_out):
            for k, v in slice.items():
                if k == "t_to_now":
                    assert len(v) == 1
                # we don't have targets, or time to next event, for the last event in the sequence
                elif k[:5] == "next_":
                    assert len(v) == d - 1
                # but we do keep that last sequence element's features, to model churn
                else:
                    assert len(v) == d

        # collate all the sequences
        out = {}
        for key, val in pre_out[0].items():
            out[key] = np.concatenate([x[key] for x in pre_out], axis=0)

        out["offsets"] = np.array(offsets)

        return out

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


# # doesn't seem like it's used anywhere
# def last_slices(
#     batch: Dict[str, torch.Tensor], model_out: Dict[str, torch.Tensor]
# ) -> Dict[str, torch.Tensor]:
#     """
#     The raw model output contains concatenated sequences for several users
#     This just takes the last value from each sequence

#     :param x: Output of the event model
#     :return: Same, but subsampled to only contain the last element in each sequence
#     """
#     # TODO: append userid
#     final = [i - 1 for i in batch["offsets"][1:]]
#     out = {k: v[final] for k, v in model_out.items()}
#     out["seq_len"] = np.diff(np.array(batch["offsets"]))

#     return out


def trim_last(batch: Dict[str, torch.Tensor], model_out: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    final = [i.item() - 1 for i in batch["offsets"][1:]]  # final elements
    inds = [k for k in range(batch["offsets"][-1]) if k not in final]
    out = {k: v[inds] for k, v in model_out.items()}
    return batch, out


def get_last(batch: Dict[str, torch.Tensor], model_out: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Obtain the last element of each sequence in the batch.

    Args:
        batch (Dict[str, torch.Tensor]): batch of sequences
        model_out (Dict[str, torch.Tensor]): model output

    Returns:
        Dict[str, torch.Tensor]: batch of sequences with only the last element
    """
    final = [i.item() - 1 for i in batch["offsets"][1:]]
    pre_final = [i.item() - 2 for i in batch["offsets"][1:]]  # final elements

    batch["dt"] = batch["dt"][final]
    out = {k: v[pre_final] for k, v in model_out.items()}

    return batch, out


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


def torchify(x: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    out = {}
    for key, val in x.items():
        try:
            out[key] = torch.from_numpy(val)
        except TypeError:  # TODO pass
            pass
    return out
