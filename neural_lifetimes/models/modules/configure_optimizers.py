from typing import Any, Dict, Optional

from torch import nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR


def configure_optimizers(
    parameters: nn.Module,
    lr: float,
    optimizer: str = "Adam",
    optimizer_kwargs: Optional[Dict[str, Any]] = None,
    scheduler: str = "ReduceLROnPlateau",
    scheduler_kwargs: Optional[Dict[str, Any]] = None,
    lightning_scheduler_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Configures optimizers and LR schedulers for any ``LightningModule``.

    Args:
        parameters (nn.Module): The models hyperparameters, e.g. ``nn.Module.parameters()``
        lr (float): The intial learning rate at which to start training.
        optimizer (str, optional): The optimizer to use. Defaults to "Adam".
        optimizer_kwargs (Optional[Dict[str, Any]], optional): Additional arguments to initialise optimizer.
            Defaults to None.
        scheduler (str, optional): The scheduler to use. Defaults to "ReduceLROnPlateau".
        scheduler_kwargs (Optional[Dict[str, Any]], optional): Additional arguments to initialise scheduler.
            Defaults to None.
        lightning_scheduler_config (Optional[Dict[str, Any]], optional): Arguments to overwrite the default
            ``lightning_scheduler_config``. Defaults to None.

    Raises:
        NotImplementedError: The specified ``optimizer`` is not implemented.
        AssertionError: When using ``MultiStepLR``, a list of milestones is required.

    Returns:
        Dict[str, Any]: The PyTorch-Lightning dictionary specifying the optimizer configuration. It has two elements
        ``optimizer`` sets the optimizer and ``lr_scheduler`` contains a lightning scheduler configuration dictionary.
    """

    # check inputs
    optimizer_kwargs = {} if optimizer_kwargs is None else optimizer_kwargs
    scheduler_kwargs = {} if scheduler_kwargs is None else scheduler_kwargs
    lightning_scheduler_config = {} if lightning_scheduler_config is None else lightning_scheduler_config

    # configure optimizer
    if optimizer == "Adam":
        opt = Adam(parameters, lr=lr, **optimizer_kwargs)
    elif optimizer == "SGD":
        opt = SGD(parameters, lr=lr, **optimizer_kwargs)
    else:
        raise NotImplementedError(f'Optimizer "{optimizer}" not implemented.')

    # configure scheduler
    lightning_scheduler_config = dict(
        {
            "interval": "epoch",
            "frequency": 200,
            "monitor": "val_loss/total",
            "strict": True,
            "name": f"lr-{optimizer}-{scheduler}",
        },
        **lightning_scheduler_config,
    )

    if scheduler == "ReduceLROnPlateau":
        scheduler_kwargs = dict(
            {
                "optimizer": opt,
                "mode": "min",
                "patience": 2,
                "verbose": True,
            },
            **scheduler_kwargs,
        )
        scheduler = ReduceLROnPlateau(**scheduler_kwargs)
    elif scheduler == "MultiStepLR":
        assert "milestones" in scheduler_kwargs, "MultiStepLR requires you to set `milestones` manually."
        scheduler_kwargs = dict(
            {
                "optimizer": opt,
                "verbose": True,
            },
            **scheduler_kwargs,
        )
        scheduler = MultiStepLR(**scheduler_kwargs)
    elif scheduler == "None":
        scheduler = None
    else:
        raise NotImplementedError(f'Scheduler "{scheduler}" not implemented.')

    lightning_scheduler_config["scheduler"] = scheduler
    return {"optimizer": opt, "lr_scheduler": lightning_scheduler_config}
