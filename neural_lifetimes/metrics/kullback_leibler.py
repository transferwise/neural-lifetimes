from typing import Dict, Optional, Any, Tuple

import torch
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence

import torchmetrics


class KullbackLeiblerDivergence(torchmetrics.Metric):
    def __init__(self, compute_on_step: Optional[bool] = None, **kwargs: Dict[str, Any]) -> None:
        super().__init__()
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # expects both preds ant target to be log scale
        self.preds.append(preds)
        self.target.append(target)

    def compute(self) -> torch.Tensor:
        preds = torch.cat(self.preds)
        target = torch.cat(self.target)

        preds_logs_prob, target_log_prob = _histogram(preds, target)
        return F.kl_div(preds_logs_prob, target_log_prob, log_target=True, reduction="none").mean()


class ParametricKullbackLeiblerDivergence(torchmetrics.Metric):
    def __init__(
        self,
        distribution: torch.distributions.Distribution,
        compute_on_step: Optional[bool] = None,
        **kwargs: Dict[str, Any]
    ) -> None:
        super().__init__()
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")
        self.distribution = distribution

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        preds, target = preds.flatten(), target.flatten()

        assert preds.shape == target.shape
        assert preds.dim() == 1

        self.preds.append(preds)
        self.target.append(target)

    def compute(self) -> torch.Tensor:
        preds = torch.cat(self.preds)
        target = torch.cat(self.target)

        p = self.distribution(target)
        q = self.distribution(preds)
        kl_div = kl_divergence(p, q)

        return kl_div.mean()


def _histogram(preds: torch.Tensor, target: torch.Tensor, nbins: int = 50, eps: float = 1.0e-6) -> Tuple[torch.Tensor]:
    assert len(preds) > 0 and len(target) > 0

    min_ = min(preds.min(), target.min())
    max_ = max(preds.max(), target.max())
    pred_log_prob = torch.log((torch.histc(preds, min=min_, max=max_) / len(preds)) + eps)
    target_log_prob = torch.log((torch.histc(target, min=min_, max=max_) / len(target)) + eps)
    return pred_log_prob, target_log_prob
