from typing import Tuple
import torch
import torch.nn as nn
import pytorch_lightning as pl

from neural_lifetimes.inference import InferenceEngine


class ForecastingCallback(pl.Callback):
    def __init__(self, splits: Tuple[str], interval, frequency) -> None:
        super().__init__()
        self.splits = splits
        self.interval = interval
        self.generate_loaders = {k: None for k in splits}
        self.engines = None

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.trainer = trainer
        self.pl_module = pl_module
        self.engines = InferenceEngine(pl_module)
        self.generate_loaders = {s: getattr(trainer.datamodule, f"{s}_forecast_dataloader") for s in self.splits}
        return super().on_fit_start(trainer, pl_module)
