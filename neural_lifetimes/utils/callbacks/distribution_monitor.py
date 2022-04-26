import numpy as np
import pytorch_lightning as pl
import torch


class MonitorDistr(pl.Callback):
    """
    Class to monitor the distribution of the model. Records the various parameters and/or metrics of the model.
    """

    def __init__(self, max_batches: int = 10, mod: int = 5):
        super().__init__()
        self.max_batches = max_batches
        self.mod = mod

    def on_train_start(self, trainer, net):
        """
        Training start callback.

        The following parameters are recorded whenever it exists:

        - the actual lambda
        - the actual p_churn
        - the time to next event
        """
        loader = trainer.datamodule.val_dataloader()
        logger = trainer.logger

        # max_batches = 10
        lambdas = []
        p = []
        t_to_next = []

        # add_graph = False
        for i, batch in enumerate(loader):
            if i >= self.max_batches:
                break

            if "lambda" in batch.keys():
                x = batch["lambda"][batch["offsets"][1:].detach().cpu().numpy() - 1].astype("float")
                x = x[~np.isnan(x)]
                lambdas.append(x)

            if "p" in batch.keys():
                x = batch["p"][batch["offsets"][1:].detach().cpu().numpy() - 1].astype("float")
                x = x[~np.isnan(x)]
                p.append(x)

            t_to_next.append(batch["next_dt"])

        if len(lambdas) > 1:
            lambdas = np.concatenate(lambdas)
            logger.experiment.add_histogram("time_intervals/lambda_True", lambdas, global_step=trainer.global_step)

        if len(p) > 1:
            p = np.concatenate(p)
            logger.experiment.add_histogram("p_churn/True", p, global_step=trainer.global_step)

        t_to_next = torch.cat(t_to_next)
        logger.experiment.add_histogram("time_intervals/t_to_next_True", t_to_next, global_step=trainer.global_step)
        logger.experiment.add_histogram(
            "time_intervals/log_t_to_next_True",
            torch.log(t_to_next + 0.001),
            global_step=trainer.global_step,  # add 0.001 for make log(0) finite.
        )

    def on_validation_batch_end(self, trainer, net, output, batch, batch_idx, dataloader_idx):
        if batch_idx % self.mod == 0:  # trainer.log_every_n_steps
            out = net(batch)
            y_pred = out["next_dt"][batch["offsets"][1:].detach().cpu().numpy() - 2]
            lambda_ = 1 / y_pred

            logger = trainer.logger
            logger.experiment.add_histogram(
                "time_intervals/t_to_next_lambda",
                lambda_.cpu().detach().numpy(),
                global_step=trainer.global_step,
            )
            logger.experiment.add_histogram(
                "time_intervals/t_to_next_Pred",
                y_pred.cpu().detach().numpy(),
                global_step=trainer.global_step,
            )
            logger.experiment.add_histogram(
                "time_intervals/log_t_to_next_Pred",
                torch.log(y_pred).cpu().detach().numpy(),
                global_step=trainer.global_step,
            )
