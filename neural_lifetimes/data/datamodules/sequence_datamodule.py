from typing import Any, Dict, List

import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split

from neural_lifetimes.utils.data import TargetCreator

from ..dataloaders.sequence_loader import SequenceLoader
from ..datasets.sequence_dataset import SequenceDataset, SequenceSubset


# TODO Rename to be more indicative of what it does.
class SequenceDataModule(pl.LightningDataModule):
    """
    A Pytorch Lightning (pl) module is a wrapper around different dataloaders and datasets.

    The pl automatically selects the correct split of the dataset given the current step.

    Args:
        dataset (SequenceDataset): A pytorch Dataset instance
        test_size (float): The proportion of data points that will be part of the validation set. 0 < test_size < 1
        batch_points (int): Batch size

    Attributes:
        dataset (SequenceDataset): A pytorch Dataset instance
        test_size (float): The proportion of data points that will be part of the validation set. 0 < test_size < 1
        batch_points (int): Batch size
        target_transform (TargetCreator): A class that transforms the targets.
    """

    def __init__(
        self,
        dataset: SequenceDataset,
        test_size: float,
        batch_points: int,
        target_transform: TargetCreator,
        min_points: int,
    ):
        super().__init__()
        self.dataset = dataset
        self.test_size = test_size
        self.batch_points = batch_points
        self.target_transform = target_transform
        self.min_points = min_points

        self.train_inds: List[int] = []
        self.predict_inds: List[int] = []
        self.test_inds: List[int] = []

    def setup(self, stage: str):
        """
        Create splits of the dataset.

        This is also the place to add data transformation and augmentations.
        Gets automatically called by pytorch. In case of distributed learning setup
        will be exectuted for each GPU.
        Docs: see https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html#setup

        Args:
            stage (str): [description]
        """
        if stage == "fit":
            self.train_inds, self.valid_inds = train_test_split(range(len(self.dataset)), test_size=self.test_size)
        if stage == "test":
            self.test_inds = list(range(len(self.dataset)))
        if stage == "predict":
            self.predict_inds = list(range(len(self.dataset)))

    def build_parameter_dict(self) -> Dict[str, Any]:
        """
        Return a dictionary of datamodule parameters.

        Returns:
            Dict[str, Any]: Parameters of the datamodule
        """
        return {
            "test_size": self.test_size,
            "batch_points": self.batch_points,
            "min_points": self.min_points,
            **self.target_transform.build_parameter_dict(),
        }

    def _build_dataloader(self, indices: List[int]) -> SequenceLoader:
        """Build a dataloader over the provided index list.

        Args:
            indices (List[int]): indices of the rows which this dataloader will provide.

        Returns:
            SequenceLoader: A SequenceLoader allowing access to data with the provided indices.
        """
        return SequenceLoader(
            SequenceSubset(self.dataset, indices),
            self.batch_points,
            self.min_points,
            self.target_transform,
        )

    def train_dataloader(self):
        """Build a dataloader for training steps.

        Returns:
            SequenceLoader: the dataloader for training.
        """
        return self._build_dataloader(self.train_inds)

    def val_dataloader(self):
        """Build a dataloader for validation steps.

        Returns:
            SequenceLoader: the dataloader for validation.
        """
        return self._build_dataloader(self.valid_inds)

    def test_dataloader(self):
        """Build a dataloader for testing.

        Returns:
            SequenceLoader: the dataloader for testing.
        """
        return self._build_dataloader(self.test_inds)

    def predict_dataloader(self):
        """Build a dataloader for prediction.

        Returns:
            SequenceLoader: the dataloader for prediction.
        """
        return self._build_dataloader(self.predict_inds)
