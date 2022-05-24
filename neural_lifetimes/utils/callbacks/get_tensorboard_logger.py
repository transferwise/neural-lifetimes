import pytorch_lightning as pl
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger


def _get_tensorboard_logger(trainer: pl.Trainer) -> TensorBoardLogger:
    for logger in trainer.loggers:
        if isinstance(logger, TensorBoardLogger):
            break
    else:
        logger_types = [type(logger).__name__ for logger in trainer.loggers]
        raise ValueError(f"Loggers needs to contain a Tensorboard logger to work. Got {logger_types}.")

    return logger
