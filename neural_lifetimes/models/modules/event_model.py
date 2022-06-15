# flake8: noqa
from typing import Any, Dict, Tuple

import pytorch_lightning as pl
import torch
from abc import ABC, abstractmethod


# TODO implement base _EventModel
class EventModel(pl.LightningModule, ABC):  # TODO Add better docstring
    @abstractmethod
    def configure_criterion(self):
        pass

    @abstractmethod
    def build_parameter_dict(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def encode(self, x: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor]:
        pass


__all__ = [EventModel]
