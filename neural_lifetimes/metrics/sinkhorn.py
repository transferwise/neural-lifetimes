from typing import Dict, Optional, Any

import torch

from scipy.stats import wasserstein_distance
import torchmetrics

# This implementation is based on scipy.stats.wasserstein_metric.
# There are solvers for differentiable approximations with GPU acceleration
# the geomloss package: https://www.kernel-operations.io/geomloss/api/pytorch-api.html#geomloss.SamplesLoss


class WassersteinMetric(torchmetrics.Metric):
    def __init__(self, compute_on_step: Optional[bool] = None, **kwargs: Dict[str, Any]) -> None:
        super().__init__()
        # intiialise states
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape, (
            "``preds`` and ``target`` need to have the same shape. "
            + f"Got ``{preds.shape}`` and ``{target.shape}`` instead."
        )
        preds, target = preds.flatten(), target.flatten()
        assert preds.dim() == 1, f"The input tensors need to be one-dimensional. Got {preds.dim()} dimensions."
        self.preds.append(preds)
        self.target.append(target)

    def compute(self):
        preds = torch.cat(self.preds)
        target = torch.cat(self.target)
        metric = wasserstein_distance(preds.cpu().numpy(), target.cpu().numpy())
        return metric
