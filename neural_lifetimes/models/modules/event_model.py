# flake8: noqa
from typing import Any, Dict, List, Union

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics import Accuracy, MeanAbsoluteError, MetricCollection
from abc import ABC, abstractmethod

from neural_lifetimes.losses import ChurnLoss, SumLoss, TauLoss, VariationalEncoderDecoderLoss

from ...data.dataloaders.sequence_loader import trim_last
from ..nets.embedder import CombinedEmbedder
from ..nets.encoder_decoder import VariationalEncoderDecoder
from ..nets.event_model import EventEncoder
from ..nets.heads import CategoricalHead, ChurnProbabilityHead, CompositeHead, ExponentialHeadNoLoss, NormalHead

from neural_lifetimes.utils.data import FeatureDictionaryEncoder

from neural_lifetimes.utils.callbacks import GitInformationLogger
from .configure_optimizers import configure_optimizers


# TODO implement base _EventModel
class _EventModel(pl.LightningModule, ABC):  # TODO Add better docstring
    @abstractmethod
    def configure_criterion(self):
        pass

    @abstractmethod
    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pass


__all__ = [_EventModel]
