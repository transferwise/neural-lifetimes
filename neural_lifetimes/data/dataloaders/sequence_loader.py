from typing import Callable, Dict, Sequence

import numpy as np
import torch

from ..datasets.sequence_dataset import SequenceSubset
from ...data.utils import torchify


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
        transform: Callable,
        target_transform: Callable,
        tokenizer: Callable,
        max_points: int,
        min_points: int = 2,
    ):
        self.data = data
        self.max_points = max_points
        self.min_points = min_points
        self.next_ind = 0
        self.transform = transform
        self.target_transform = target_transform
        self.tokenizer = tokenizer

    def __iter__(self):
        """Get self as iterator of sequence batches."""
        return self

    def __len__(self):
        """
        Return 1e10 as a bodge.

        See https://github.com/PyTorchLightning/pytorch-lightning/issues/323#issuecomment-540840761 for details
        """
        return 1e10  # TODO there is probably a better solution

    def __next__(self) -> Dict[str, torch.Tensor]:
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
            next_len = min(self.tokenizer.max_item_len, next_len)

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

        # assert that all relevant elements are tensors (if the original are tensors, so are the 'next_' shifted)
        required_tensors = self.transform.features + ["dt", "t"]
        for name, val in out.items():
            if name in required_tensors:
                assert isinstance(val, torch.Tensor)

        # assert that discrete tensors are int64 and continuous float32
        for key, val in out.items():
            if key in self.transform.discrete_features.keys():
                assert val.dtype == torch.int64
            elif key in self.transform.continuous_features:  # TODO will fail on BTYD mode
                assert val.dtype == torch.float32

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

            # convert timestamps to float
            s["t"] = self.datetime2tensor(s["t"])
            asof_time = self.datetime2tensor(np.array([self.data.asof_time]))

            # add dt
            s["dt"] = np.append([0], s["t"][1:] - s["t"][:-1]).astype(np.float32)
            s["t"] = s["t"].astype(np.float32)

            # tokenize: add tokens and truncate sequences
            s = self.tokenizer(s)

            # encoder features
            s = self.transform(s)

            # assert that all dts are postitive
            assert len(s["dt"]) == len(s["t"])
            if len(s["dt"]) > 2:
                assert (
                    sum(s["dt"][2:] < 0) == 0
                ), "We're getting negative time intervals, are you slicing by profile correctly? "

            # add to to now
            s["t_to_now"] = asof_time - s["t"][-1]

            # create next_ shifted variables
            next_out = self.target_transform(s)

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
        for name in pre_out[0].keys():
            out[name] = np.concatenate([x[name] for x in pre_out], axis=0)

        out["offsets"] = np.array(offsets)

        return out

    # TODO: check duplicate with date_arithmetic
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
