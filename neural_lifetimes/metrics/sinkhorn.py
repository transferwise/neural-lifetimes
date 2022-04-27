from typing import Optional, Any, Callable

import torch

from scipy.stats import wasserstein_distance
import torchmetrics

# This implementation is based on scipy.stats.wasserstein_metric.
# There are solvers for differentiable approximations with GPU acceleration
# the geomloss package: https://www.kernel-operations.io/geomloss/api/pytorch-api.html#geomloss.SamplesLoss

class WassersteinMetric(torchmetrics.Metric):
    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
        **kwargs,
    ) -> None:
        super().__init__(compute_on_step, dist_sync_on_step, process_group, dist_sync_fn, **kwargs)
        # intiialise states
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert (
            preds.shape == target.shape
        ), f"``preds`` and ``target`` need to have the same shape. Got ``{preds.shape}`` and ``{target.shape}`` instead."
        assert preds.dim() == 1, f"The input tensors need to be one-dimensional. Got {preds.dim()} dimensions."
        self.preds.append(preds)
        self.target.append(target)

    def compute(self):
        preds = torch.cat(self.preds)
        target = torch.cat(self.target)
        metric = wasserstein_distance(preds, target)
        return metric


# Attempt at implementing GPU accelerator Wasserstein

# class SinkhornMetric(torchmetrics.Metric):
#     def __init__(
#         self,
#         norm: int = 2,
#         penalty: float = 0.05,
#         compute_on_step: bool = True,
#         dist_sync_on_step: bool = False,
#         process_group: Optional[Any] = None,
#         dist_sync_fn: Callable = None,
#         **kwargs
#     ) -> None:
#         super().__init__(compute_on_step, dist_sync_on_step, process_group, dist_sync_fn, **kwargs)
#         self.solver = geomloss.SamplesLoss(loss="sinkhorn", p=norm, blur=penalty)
#         # intiialise states
#         self.add_state("preds", default=[], dist_reduce_fx="cat")
#         self.add_state("target", default=[], dist_reduce_fx="cat")

#     def update(self, preds: torch.Tensor, target: torch.Tensor):
#         assert preds.shape == target.shape
#         self.preds.append(preds)
#         self.target.append(target)

#     def compute(self):
#         preds = torch.cat(self.preds)
#         target = torch.cat(self.target)
#         # TODO standardize inputs
#         return self.solver(preds, target)


# class WassersteinMetric(SinkhornMetric):
#     def __init__(
#         self,
#         norm: int = 2,
#         compute_on_step: bool = True,
#         dist_sync_on_step: bool = False,
#         process_group: Optional[Any] = None,
#         dist_sync_fn: Callable = None,
#         **kwargs
#     ) -> None:
#         super().__init__(norm, 0, compute_on_step, dist_sync_on_step, process_group, dist_sync_fn, **kwargs)
