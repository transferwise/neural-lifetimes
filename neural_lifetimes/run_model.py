import os
from typing import Any, Dict, List, Optional, Union

import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import LightningLoggerBase, TensorBoardLogger

from .data.datamodules.sequence_datamodule import SequenceDataModule
from .models.modules.classic_model import ClassicModel
from .utils.callbacks import MonitorChurn, DistributionMonitor, MonitorProjection


def run_model(
    datamodule: SequenceDataModule,
    model: ClassicModel,
    log_dir: str,
    num_epochs: int = 100,
    checkpoint_path: Optional[str] = None,
    run_mode: str = "train",
    val_check_interval: Union[int, float] = 1.0,
    limit_val_batches: Union[int, float] = 1.0,
    gradient_clipping: float = 0.1,
    trainer_kwargs: Optional[Dict[str, Any]] = None,
    loggers: Optional[List[LightningLoggerBase]] = None,
    logger_kwargs: Optional[Dict[str, Any]] = None,
) -> pl.Trainer:
    """
    Run the model for training or testing.

    Note:
        This function function will run a model using the
        `Trainer <pytorch_lightning.Trainer>`. Some key arguments for you to set are
        given explicitly. You can overwrite any other argument of the Trainer or Logger manually.

    Args:
        datamodule (PLDataModule): The data on which the model is run.
        model (pytorch_lightning.LightningModule): The LightningModule containing the Embedder, TargetCreator, network
            and instructions for forward passes.
        log_dir (str): Path into which model checkpoints and tensorboard logs will be written.
        num_epochs (int): Number of epochs to train for. Default is 100.
        checkpoint_path (Optional[str]): Path to a model to load. If is ``None`` (default), we will train a new model.
        run_mode (str): 'train': Train the model 'test': run an inference pass. 'none': Do neither.
            Default is ``train``.
        val_check_interval (int | float): sets the frequency of running the model on the validation set.
            See: https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#val-check-interval.
            Default is ``1.0``.
        limit_val_batches (int | float): Limits the number of batches of the validation set that is run by the Trainer.
            See: https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#limit-val-batches.
            Default is ``1.0``.
        gradient_clipping (float): sets the threshold L2 norm for clipping the gradients.
            See: https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#gradient-clip-val.
            Default is ``0.1``.
        trainer_kwargs (Dict(str, Any)): Forward any keyword argument to the `Trainer <pytorch_lightning.Trainer>`.
            Any argument passed here, will be set in the Trainer constructor. Default is ``None``.
        loggers (Optional[List[LightningLoggerBase]]): This function uses the TensorboardLogger by default. If you wish
            to use another logger, you can pass a list of any other `Logger <pytorch_lightning.LightningLoggerBase>` in
            here. Default is ``None``.
        logger_kwargs (Dict(str, Any)): Forward any keword argument to the
            `Logger <pytorch_lightning.LightningLoggerBase>`. Any argument passed here, will be set in the Logger
            constructor. Default is ``None``.

    Returns:
        pl.Trainer: The trainer object.
    """
    if checkpoint_path:
        model.load_state_from_checkpoint(checkpoint_path)

    # configure loggers to be used by pytorch.
    # https://pytorch-lightning.readthedocs.io/en/latest/common/loggers.html
    if loggers is None:
        # process user arguments
        if logger_kwargs is None:
            logger_kwargs = {}
        # ensure that the user can overwrite the default Tb arguments
        logger_kwargs = dict(
            {
                "save_dir": log_dir,
                "default_hp_metric": True,
                "log_graph": False,
                "name": "logs",
            },
            **logger_kwargs,
        )
        loggers = [TensorBoardLogger(**logger_kwargs)]
    else:
        assert isinstance(loggers, List)
        assert [isinstance(logger, LightningLoggerBase) for logger in loggers]
        assert logger_kwargs is None, "If custom logger is supplied, 'logger_kwargs' argument must be 'None'"

    # configure callbacks. That is actions taken on certain events, e.g. epoch end.
    # TODO: renable Montioring Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(log_dir, f"version_{loggers[0].version}"),
            filename="{epoch}-{step}-{val_loss/total:.2f}",
            monitor="val_loss/total",
            mode="min",
            save_last=True,
            save_top_k=3,
            verbose=True,
        ),
        # DistributionMonitor(),
        # MonitorProjection(),
        # MonitorChurn(),
    ]

    # process user arguments for Trainer
    if trainer_kwargs is None:
        trainer_kwargs = {}
    # ensure the user can overwrite anything they want
    trainer_kwargs = dict(
        {
            "callbacks": callbacks,
            "logger": loggers,
            "max_epochs": num_epochs,
            "val_check_interval": val_check_interval,
            "limit_val_batches": limit_val_batches,
            "amp_backend": "native",
            "precision": 16,
            "track_grad_norm": 2,
            "gradient_clip_val": gradient_clipping,
        },
        **trainer_kwargs,
    )

    trainer = pl.Trainer(
        **trainer_kwargs,
    )

    # if model weights are supplied we do inference. Otherwise we train.
    if run_mode == "test":
        trainer.test(model, datamodule=datamodule)
    elif run_mode == "train":
        # start training
        trainer.fit(model, datamodule=datamodule)
    elif run_mode == "none":
        pass
    else:
        raise ValueError(f"`run_mode` must be`train`, `test` or `none`. Not {run_mode}.")
    return trainer


# TODO add train, test and inference functions for users to find in the docs and being redirected to run_model
