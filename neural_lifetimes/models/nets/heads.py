import math
from typing import Callable, Dict

import torch
import torch.distributions as d
from torch import nn
from torch.nn import functional as F

from neural_lifetimes.losses import CategoricalLoss, CompositeLoss, ExponentialLoss, NormalLoss


class BasicHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        drop_rate: float,
        output_dim: int,
        transform: Callable,
        init_norm=False,
    ):
        super().__init__()
        self.drop_rate = drop_rate
        hidden_dim = int(math.sqrt(input_dim * output_dim))
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.transform = transform
        self.input_shape = [None, input_dim]
        self.output_shape = [None, output_dim]

        self.init_norm = init_norm
        self.init_bias = None
        self.init_std = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eps = 0.00001
        out = F.dropout(F.relu(self.fc1(x)), self.drop_rate, self.training)
        tmp = self.fc2(out)

        if self.init_norm:
            if self.init_bias is None:
                self.init_bias = -tmp.mean(dim=0, keepdims=True).detach()
                self.init_std = tmp.std(dim=0, keepdims=True).detach() + eps
            tmp = (tmp + self.init_bias) / self.init_std
        out = self.transform(tmp)
        return out


class ExponentialHead(BasicHead):
    """
    Output is the scale for a normalized distribution, typically a time delta.
    """

    def __init__(self, input_dim: int, drop_rate: float):
        super().__init__(input_dim, drop_rate, 1, self.normalize)

    def distribution(self, v: torch.Tensor) -> d.Distribution:
        return d.exponential.Exponential(1 / v)

    def loss_function(self):
        return ExponentialLoss()

    @staticmethod
    def normalize(x: torch.Tensor):
        eps = 0.00001
        x = torch.exp(torch.clamp(x, min=eps, max=60))
        return x


class ExponentialHeadNoLoss(ExponentialHead):
    def loss_function(self):
        return None


class ProbabilityHead(BasicHead):
    """
    Output is the scale for a normalized distribution, typically a time delta.
    """

    def __init__(self, input_dim: int, drop_rate: float):
        super().__init__(
            input_dim,
            drop_rate,
            1,
            self.sigmoid,  # was 0.5*x
            init_norm=False,
        )

    def distribution(self, v: torch.Tensor) -> d.Distribution:
        return d.bernoulli.Bernoulli(v)

    def loss_function(self):
        # TODO: insert the right thing here
        return NotImplementedError

    @staticmethod
    def sigmoid(x: torch.Tensor):
        return torch.sigmoid(x)


class ChurnProbabilityHead(ProbabilityHead):
    def distribution(self, v: torch.Tensor) -> d.Distribution:
        return d.bernoulli.Bernoulli(v)

    def loss_function(
        self,
    ):  # this one needs a special treatment because we don't need to clip last value in each seq
        return None


class NormalHead(BasicHead):
    """
    Output is the mean and std for a normal distribution.

    Could also be used for a LogNormal distribution.
    """

    def __init__(self, input_dim: int, drop_rate: float):
        super().__init__(input_dim, drop_rate, 2, self.normalize)

    def distribution(self, v: torch.Tensor) -> d.Distribution:
        return d.normal.Normal(v[:, 0], torch.exp(torch.clamp(v[:, 1], min=-30, max=50)))

    def loss_function(self):
        return NormalLoss()

    @staticmethod
    def normalize(x: torch.Tensor):
        # x[:, 1] = torch.exp(x[:, 1]/10)
        # x[:, 1] = x[:, 1]**2
        return x


class CategoricalHead(BasicHead):
    """
    Output is the scale for a normalized distribution, typically a time delta.
    """

    def __init__(self, input_dim: int, num_categories: int, drop_rate: float):
        super().__init__(input_dim, drop_rate, num_categories, self.softmax)

    def distribution(self, v: torch.Tensor) -> d.Distribution:
        return d.categorical.Categorical(v)

    def loss_function(self):
        return CategoricalLoss()

    @staticmethod
    def softmax(x: torch.Tensor):
        return F.log_softmax(x, dim=-1)


class CompositeDistribution:
    def __init__(self, distrs: Dict[str, d.Distribution]):
        self.distrs = distrs

    def sample(self, shape) -> Dict[str, torch.Tensor]:
        return {k: d.sample(shape) for k, d in self.distrs}


class CompositeHead(nn.Module):
    def __init__(self, d: Dict[str, nn.Module]):
        super().__init__()
        self.heads = nn.ModuleDict(d)
        self.input_shape = next(iter(d.values())).input_shape

    def forward(self, x):
        return {key: h(x) for key, h in self.heads.items()}

    def loss_function(self, preprocess: Callable) -> CompositeLoss:
        return CompositeLoss(
            {k: h.loss_function() for k, h in self.heads.items() if h.loss_function() is not None},
            preprocess,
        )

    def distribution(self, params: Dict[str, torch.Tensor]) -> CompositeDistribution:
        gens = {k: h.distribution(params[k]) for k, h in self.heads.items()}
        return CompositeDistribution(gens)
