import pytorch_lightning as pl
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger


def _get_tensorboard_logger(trainer: pl.Trainer) -> TensorBoardLogger:
    for logger in trainer.loggers:
        if isinstance(logger, TensorBoardLogger):
            break
    else:
        logger_types = [type(logger).__name__ for logger in trainer.loggers]
        raise ValueError(
            f"No Tensorboard logger found in the lightning Trainers loggers. Got {logger_types} instead."
            + "his callback requires Tensorboard loggers to be present."
        )

    return logger
