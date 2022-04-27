from typing import Optional, Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence

import torchmetrics


class KullbackLeiblerDivergence(torchmetrics.Metric):
    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ) -> None:
        super().__init__(compute_on_step, dist_sync_on_step, process_group, dist_sync_fn)
        self.add_state("kl", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # expects both preds ant target to be log scale
        self.kl.extend(F.kl_div(preds, target, reduction="none", log_target=True))

    def compute(self) -> torch.Tensor:
        return torch.stack(self.kl).mean()


class ParametricKullbackLeiblerDivergence(torchmetrics.Metric):
    def __init__(
        self,
        distribution: torch.distributions.Distribution,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ) -> None:
        super().__init__(compute_on_step, dist_sync_on_step, process_group, dist_sync_fn)
        self.add_state("kl", default=[], dist_reduce_fx="cat")
        self.distribution = distribution

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = self.distribution(target)
        q = self.distribution(preds)
        self.kl.extend(kl_divergence(p, q))

    def compute(self) -> torch.Tensor:
        return torch.stack(self.kl).mean()
