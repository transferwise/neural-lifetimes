from typing import Callable, Dict, Optional, Sequence

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
        max_points: int,
        min_points: int = 2,
        transform: Optional[Callable] = None,
    ):
        self.data = data
        self.max_points = max_points
        self.min_points = min_points
        self.next_ind = 0
        self.transform = transform

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
            next_len = self.transform.output_len(next_len)  # cut the length if necessary

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
            next_out = self.transform(s, self.data.asof_time)
            if next_out is not None:
                offsets.append(offsets[-1] + next_out["t"].shape[0])  # the offsets refer to the features
                pre_out.append(next_out)

        # check that the offsets match the array lengths
        for d, slice in zip(np.diff(np.array(offsets)), pre_out):
            for k, v in slice.items():
                if k == "t_to_now":
                    assert len(v) == 1
                # we don't have targets, or time to next event, for the last event in the sequence
                elif k == "next_dt" or k[:5] == "next_":
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
