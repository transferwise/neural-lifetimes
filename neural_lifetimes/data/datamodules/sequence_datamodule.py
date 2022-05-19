import random
from typing import Any, Dict, List

import pytorch_lightning as pl
from sklearn.model_selection import train_test_split

from ..dataloaders.sequence_loader import SequenceLoader
from ..datasets.sequence_dataset import SequenceDataset, SequenceSubset
from neural_lifetimes.utils.data import FeatureDictionaryEncoder, Tokenizer, TargetCreator


class SequenceDataModule(pl.LightningDataModule):
    """
    A Pytorch Lightning (pl) module is a wrapper around different dataloaders and datasets.

    The pl automatically selects the correct split of the dataset given the current step.

    Args:
        dataset (SequenceDataset): The dataset for training.
        test_size (float): The proportion of data points that will be part of the validation set. 0 < test_size < 1
        batch_points (int): Batch size (count of events).
        transform (FeatureDictionaryEncoder): The encoder to be used on the batch. Will be passed on to the
            ``SequenceLoader``.
        target_transform (TargetCreator): The transform used to generate targets. Will be passed on to the
            ``SequenceLoader``.
        tokenizer (Tokenizer): The tokenizer appending start tokens for packing the sequences. Will be passed
            on to the ``SequenceLoader``.
        min_points (int): ??
        forecast_dataset (SequenceDataset, Optional): The dataset used for forecasting. Defaults to None.
        forecast_limit (int, Optional): The maximum number of data points on which to perform forecasting.
            This will generate a simple random sample of size ``forecast_limit`` from the indexes of the
            respective split of indices. if ``None``, no sample is taken. Defaults to None.

    Attributes:
        dataset (SequenceDataset): A pytorch Dataset instance
        test_size (float): The proportion of data points that will be part of the validation set. 0 < test_size < 1
        batch_points (int): Batch size
        target_transform (TargetCreator): A class that transforms the targets.
        TODO: all
    """

    def __init__(
        self,
        dataset: SequenceDataset,
        test_size: float,
        batch_points: int,
        transform: FeatureDictionaryEncoder,
        target_transform: TargetCreator,
        tokenizer: Tokenizer,
        min_points: int,
        forecast_dataset: SequenceDataset = None,
        forecast_limit: int = None,
    ):
        super().__init__()
        self.dataset = dataset
        self.forecast_dataset = forecast_dataset
        self.test_size = test_size
        self.batch_points = batch_points
        self.transform = transform
        self.target_transform = target_transform
        self.tokenizer = tokenizer
        self.min_points = min_points
        self.forecast_limit = forecast_limit

        self.save_hyperparameters(self.build_parameter_dict())

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
        hparams = {
            "test_size": self.test_size,
            "batch_points": self.batch_points,
            "min_points": self.min_points,
        }

        hparams = {f"datamodule/{k}": v for k, v in hparams.items()}
        # return {**hparams, **self.dataset.build_parameter_dict(), **self.target_transform.build_parameter_dict()}

        # get target creator parameters
        # TODO improve documentation on this
        try:
            hparams.update(
                {f"target_transform/{k}": v for k, v in self.target_transform.build_parameter_dict().items()}
            )
        except AttributeError:
            pass

        # get dataset parameters
        try:
            hparams.update({f"dataset/{k}": v for k, v in self.dataset.build_parameter_dict().items()})
        except AttributeError:
            pass

        return hparams

    def _build_dataloader(self, dataset: SequenceDataset, indices: List[int]) -> SequenceLoader:
        """Build a dataloader over the provided index list.

        Args:
            indices (List[int]): indices of the rows which this dataloader will provide.

        Returns:
            SequenceLoader: A SequenceLoader allowing access to data with the provided indices.
        """
        return SequenceLoader(
            SequenceSubset(dataset, indices),
            self.transform,
            self.target_transform,
            self.tokenizer,
            self.batch_points,
            self.min_points,
        )

    def train_dataloader(self):
        """Build a dataloader for training steps.

        Returns:
            SequenceLoader: the dataloader for training.
        """
        return self._build_dataloader(self.dataset, self.train_inds)

    def val_dataloader(self):
        """Build a dataloader for validation steps.

        Returns:
            SequenceLoader: the dataloader for validation.
        """
        return self._build_dataloader(self.dataset, self.valid_inds)

    def test_dataloader(self):
        """Build a dataloader for testing.

        Returns:
            SequenceLoader: the dataloader for testing.
        """
        return self._build_dataloader(self.dataset, self.test_inds)

    def predict_dataloader(self):
        """Build a dataloader for prediction.

        Returns:
            SequenceLoader: the dataloader for prediction.
        """
        # TODO: decide on forecast vs train dataset
        return self._build_dataloader(self.forecast_dataset, self.predict_inds)

    def _forecast_indices(self, indices):
        if indices is None or len(indices) <= self.forecast_limit:
            return indices
        else:
            return random.sample(indices, self.forecast_limit)

    def train_forecast_dataloader(self):
        return self._build_dataloader(self.forecast_dataset, self._forecast_indices(self.train_inds))

    def val_forecast_dataloader(self):
        return self._build_dataloader(self.forecast_dataset, self._forecast_indices(self.valid_inds))

    def test_forecast_dataloader(self):
        return self._build_dataloader(self.forecast_dataset, self._forecast_indices(self.test_inds))
