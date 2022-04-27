from typing import Dict, Optional, Any

import torch
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence

import torchmetrics


class KullbackLeiblerDivergence(torchmetrics.Metric):
    def __init__(self, compute_on_step: Optional[bool] = None, **kwargs: Dict[str, Any]) -> None:
        super().__init__(compute_on_step, **kwargs)
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
        compute_on_step: Optional[bool] = None,
        **kwargs: Dict[str, Any]
    ) -> None:
        super().__init__(compute_on_step, **kwargs)
        self.add_state("kl", default=[], dist_reduce_fx="cat")
        self.distribution = distribution

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = self.distribution(target)
        q = self.distribution(preds)
        self.kl.extend(kl_divergence(p, q))

    def compute(self) -> torch.Tensor:
        return torch.stack(self.kl).mean()
