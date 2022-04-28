from typing import Any

import pytest
import torch.nn as nn

from pytorch_lightning import LightningModule

from neural_lifetimes.models.modules.configure_optimizers import configure_optimizers


@pytest.fixture
def model():
    class TestModel(LightningModule):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.net = nn.Linear(10, 10)

        def forward(self, x):
            return self.net(x)

    return TestModel()


class Test_Configure_Optimizer:
    def test_default_args(self, model: LightningModule):
        config_dict = configure_optimizers(model.parameters(), lr=0.1)
        assert set(config_dict.keys()) == {"optimizer", "lr_scheduler"}

    def test_custom_args(self, model: LightningModule):
        config_dict = configure_optimizers(
            model.parameters(),
            lr=0.1,
            optimizer="SGD",
            optimizer_kwargs={"momentum": 0.5},
            scheduler="MultiStepLR",
            scheduler_kwargs={"milestones": [1, 2, 3]},
            lightning_scheduler_config={"frequency": 5},
        )
        assert config_dict["optimizer"].param_groups[0]["momentum"] == 0.5
        assert config_dict["lr_scheduler"]["frequency"] == 5
        assert list(config_dict["lr_scheduler"]["scheduler"].milestones) == [1, 2, 3]
        assert set(config_dict.keys()) == {"optimizer", "lr_scheduler"}

    def test_raises_optimizer_error(self, model: LightningModule):
        with pytest.raises(NotImplementedError) as excinfo:
            configure_optimizers(model.parameters(), lr=0.1, optimizer="Newton-Raphson")
        (msg,) = excinfo.value.args
        assert msg == 'Optimizer "Newton-Raphson" not implemented.'

    def test_raises_multistep_error(self, model: LightningModule):
        with pytest.raises(AssertionError) as excinfo:
            configure_optimizers(model.parameters(), lr=0.1, scheduler="MultiStepLR")
        (msg,) = excinfo.value.args
        assert msg == "MultiStepLR requires you to set `milestones` manually."
