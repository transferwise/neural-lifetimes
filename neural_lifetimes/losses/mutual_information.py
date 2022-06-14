from abc import ABC, abstractmethod
import torch
import torch.nn as nn

from utils.score_estimators import SpectralScoreEstimator
from utils.scheduler import WeightScheduler

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
    def __init__(self, fit_loss, weight_scheduler: WeightScheduler) -> None:
        super().__init__()
        self.fit_loss = fit_loss
        self.weight_scheduler = weight_scheduler
        self.mi = MutualInformationGradientEstimator()

    def forward(self, pred, target, data, latent):
        model_fit = self.fit_loss(pred, target)
        latent_loss = self.mi(data, latent)

        loss = model_fit + self.reg_weight * latent_loss

        return (
            loss,
            latent_loss.detach(),
            model_fit.detach(),
        )  # detach these so they arent accidentally used for training

    @property
    def reg_weight(self):
        return self.weight_scheduler.weight
