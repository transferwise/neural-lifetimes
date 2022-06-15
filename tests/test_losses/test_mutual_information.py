import torch
from neural_lifetimes.losses.mutual_information import MutualInformationGradientEstimator
from neural_lifetimes.losses import InformationBottleneckLoss

import pytest

from neural_lifetimes.utils.scheduler import WeightScheduler


class TestMutualInformation:
    @staticmethod
    def get_mi_obj(n_eigen=None, n_eigen_threshold=None):
        return MutualInformationGradientEstimator(n_eigen, n_eigen_threshold)

    @pytest.mark.parametrize(("n_eigen", "n_eigen_threshold"), ((None, None), (10, None), (None, 1e-4)))
    def test_constructor(self, n_eigen, n_eigen_threshold):
        self.get_mi_obj(n_eigen, n_eigen_threshold)

    @pytest.mark.parametrize(("n_eigen", "n_eigen_threshold"), ((None, None), (10, None), (None, 1e-4)))
    def test_forward_isnotnan(self, n_eigen, n_eigen_threshold):
        fn = self.get_mi_obj(n_eigen, n_eigen_threshold)
        assert torch.isfinite(fn(torch.randn((10, 5)), torch.randn((10, 2))))


class TestInformationBottleneckLoss:
    @staticmethod
    def get_loss_fn(n_eigen: int = None, n_eigen_threshold: float = None):
        fit_loss = lambda x: x
        weight_scheduler = WeightScheduler()
        return InformationBottleneckLoss(fit_loss, weight_scheduler, n_eigen, n_eigen_threshold)

    @pytest.mark.xfail
    def test_constructor(self):
        pass

    @pytest.mark.xfail
    def test_reg_weight(self):
        assert self.get_loss_fn().reg_weight == 1

    @pytest.mark.xfail
    def test_forward(self):
        pass
