from typing import Dict, Tuple
import torch
import torch.nn as nn

from neural_lifetimes.utils.score_estimators import SpectralScoreEstimator
from neural_lifetimes.utils.scheduler import WeightScheduler

# code heavily relies on https://github.com/zhouyiji/MIGE


class MutualInformationGradientEstimator(nn.Module):
    def __init__(self, n_eigen: int = None, n_eigen_threshold: float = None) -> None:
        """The `MutualInformationGradientEstimator` provides a Monte Carlo estimator of Mutual Information.

        This class allows to train a model using mutual information based on empirical data distributions as an
        objective function. The Mutual Information Gradients are estimated using the MIGE method, Wen et al, 2020. The
        method is fully compatable with stochastic optimization techniques. This class can be used like any loss in
        PyTorch. Beware, that this is the mutual information, so gradient descent would have to be used with the
        negative mutual information.

        See: https://openreview.net/forum?id=ByxaUgrFvH&noteId=rkePFjKYor

        Args:
            n_eigen (int, optional): Sets the number of eigenvalues to be used for the Nyström approximation.
                If ``None``, all values will be used. Defaults to None.
            n_eigen_threshold (float, optional): Sets the threshold for eigenvalues to be used for the Nystöm
                approximation. If ``None``, all values will be used. Defaults to None.
        """
        super().__init__()
        self.spectral_j = SpectralScoreEstimator(n_eigen=n_eigen, n_eigen_threshold=n_eigen_threshold)
        self.spectral_m = SpectralScoreEstimator(n_eigen=n_eigen, n_eigen_threshold=n_eigen_threshold)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Stores the mutual information gradient in the autograd system.

        Args:
            x (torch.Tensor): The input data.
            y (torch.Tensor): The output data.

        Returns:
            torch.Tensor: A scalar leading associated with the correct gradient of the MI.
        """
        loss = self._entropy_surrogate(self.spectral_j, torch.cat([x, y], dim=-1)) - self._entropy_surrogate(
            self.spectral_m, y
        )
        return loss

    @staticmethod
    def _entropy_surrogate(estimator, samples):
        dlog_q = estimator.compute_gradients(samples.detach(), None)
        surrogate_cost = torch.mean(torch.sum(dlog_q.detach() * samples, -1))
        return surrogate_cost


class InformationBottleneckLoss(nn.Module):
    def __init__(
        self,
        fit_loss: nn.Module,
        weight_scheduler: WeightScheduler,
        n_eigen: int = None,
        n_eigen_threshold: float = None,
    ) -> None:
        """Implements an information bottleneck loss.

        The information bottleneck is a penalty on the any other loss. The information bottlneck between A and B
        minimises the information about A contained in B, i.e. the mutual information I(A,B). The objective function is:
        ``L = fit_loss * reg_weight * I(A,B)``

        The NN predicts: X -> A -> B -> Y. Where, X is the data, A is the RNN output, B is the bottleneck, and Y is the
        prediction. The information bottleneck is enforced between A and B. If ``reg_weight`` is small enough, the fit
        loss forces information to flow from A -> B that is relevant to Y, however, the I(A,B) penalty will penalise all
        information flowing through, meaning that information in X irrelevant to Y will be supressed.

        Args:
            fit_loss (nn.Module): The loss function for the model fit.
            weight_scheduler (WeightScheduler): A weight scheduler for the weight of the mutual information in the total
                loss.
            n_eigen (int, optional): Sets the number of eigenvalues to be used for the Nyström approximation.
                If ``None``, all values will be used. Defaults to None.
            n_eigen_threshold (float, optional): Sets the threshold for eigenvalues to be used for the Nystöm
                approximation. If ``None``, all values will be used. Defaults to None.
        """
        super().__init__()
        self.fit_loss = fit_loss
        self.weight_scheduler = weight_scheduler
        self.mi = MutualInformationGradientEstimator(n_eigen=n_eigen, n_eigen_threshold=n_eigen_threshold)

    def forward(
        self, pred: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Calculated the information bottleneck loss.

        Args:
            pred (Dict[str, torch.Tensor]): the NN prediction dictionary. Beyond keys required for ``fit_loss``,
                this should contain "event_encoding" and "bottleneck" between which the penalty is imposed.
            target (Dict[str, torch.Tensor]): The target data.

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]: The loss.
        """
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
    def reg_weight(self) -> float:
        """The weight currently used for the mutual information. This is an alias and determined through the
        ``WeightScheduler``.

        Returns:
            float: The current weight
        """
        return self.weight_scheduler.weight
