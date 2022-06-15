import torch
import torch.nn as nn

from neural_lifetimes.utils.score_estimators import SpectralScoreEstimator
from neural_lifetimes.utils.scheduler import WeightScheduler

# code heavily relies on https://github.com/zhouyiji/MIGE


class MutualInformationGradientEstimator(nn.Module):
    def __init__(self, n_eigen=None, n_eigen_threshold=None) -> None:
        super().__init__()
        self.spectral_j = SpectralScoreEstimator(n_eigen=n_eigen, n_eigen_threshold=n_eigen_threshold)
        self.spectral_m = SpectralScoreEstimator(n_eigen=n_eigen, n_eigen_threshold=n_eigen_threshold)

    def forward(self, x, y):
        loss = self.entropy_surrogate(self.spectral_j, torch.cat([x, y], dim=-1)) - self.entropy_surrogate(
            self.spectral_m, y
        )
        return loss

    @staticmethod
    def entropy_surrogate(estimator, samples):
        dlog_q = estimator.compute_gradients(samples.detach(), None)
        surrogate_cost = torch.mean(torch.sum(dlog_q.detach() * samples, -1))
        return surrogate_cost


class InformationBottleneckLoss(nn.Module):
    def __init__(
        self, fit_loss, weight_scheduler: WeightScheduler, n_eigen: int = None, n_eigen_threshold: float = None
    ) -> None:
        super().__init__()
        self.fit_loss = fit_loss
        self.weight_scheduler = weight_scheduler
        self.mi = MutualInformationGradientEstimator(n_eigen=n_eigen, n_eigen_threshold=n_eigen_threshold)

    def forward(self, pred, target):
        fit_loss, losses_dict = self.fit_loss(pred, target)
        latent_loss = self.mi(pred["event_encoding"], pred["bottleneck"])
        loss = fit_loss + self.reg_weight * latent_loss

        losses_dict["model_fit"] = fit_loss
        losses_dict["total"] = loss
        losses_dict["mutual_information"] = latent_loss

        losses_dict = {f"loss/{name}": loss for name, loss in losses_dict.items()}

        assert loss not in [-torch.inf, torch.inf], "Loss not finite!"
        assert not torch.isnan(loss), "Got a NaN loss"

        # if the sum is finite, this should be redundant?!
        assert sum(losses_dict.values()) not in [-torch.inf, torch.inf], "Loss not finite!"
        assert not torch.isnan(sum(losses_dict.values())), "Got a NaN loss"

        return loss, {k: v.detach() for k, v in losses_dict.items()}

    @property
    def reg_weight(self):
        return self.weight_scheduler.weight
