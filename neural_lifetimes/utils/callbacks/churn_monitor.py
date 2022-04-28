from typing import Dict

import numpy as np
import pytorch_lightning as pl
import torch


class MonitorChurn(pl.Callback):
    def __init__(self, max_batches: int = 100, hist_mod: int = 5):
        super().__init__()
        self.max_batches = max_batches
        self.hist_mod = hist_mod

    def on_train_batch_end(self, trainer, net, outputs, batch, batch_idx):
        out = net(batch)
        _, x_remove_initial = remove_initial_event(batch, out)  # TODO Cornelius: why how what?
        _, x_last = get_last(batch, out)  # TODO Cornelius: why how what?

        logger = trainer.logger

        if "p" in batch.keys():
            actual_mean_p_churn = batch["p"][batch["p"] != np.array(None)].mean()
            actual_mean_last_p_churn = batch["p"][batch["offsets"][1:].cpu().detach().numpy() - 1].mean()

            logger.log_metrics({"p_churn/mean_True": actual_mean_p_churn}, step=trainer.global_step)
            logger.log_metrics(
                {"p_churn/mean_last_element_True": actual_mean_last_p_churn},
                step=trainer.global_step,
            )

        logger.log_metrics(
            {"p_churn/mean_train": x_remove_initial["p_churn"].mean()},
            step=trainer.global_step,
        )
        logger.log_metrics(
            {"p_churn/mean_last_element_train": x_last["p_churn"].mean()},
            step=trainer.global_step,
        )

        if batch_idx % self.hist_mod == 0:
            logger.experiment.add_histogram(
                "p_churn/predicted_train",
                x_remove_initial["p_churn"],
                global_step=trainer.global_step,
            )
            logger.experiment.add_histogram(
                "p_churn/predicted_last_element_train",
                x_last["p_churn"],
                global_step=trainer.global_step,
            )

    # TODO update call signature: remove `unused` as soon as plt patches
    def on_validation_batch_end(self, trainer: pl.Trainer, net: pl.LightningModule, outputs, batch, batch_idx, unused):
        if batch_idx == 0:
            return

        out = net(batch)
        _, x_remove_initial = remove_initial_event(batch.copy(), out)
        _, x_last = get_last(batch.copy(), out)

        logger = trainer.logger
        logger.log_metrics(
            {"p_churn/mean_val": x_remove_initial["p_churn"].mean()},
            step=trainer.global_step,
        )
        logger.log_metrics(
            {"p_churn/mean_last_element_val": x_last["p_churn"].mean()},
            step=trainer.global_step,
        )
        logger.experiment.add_histogram(
            "p_churn/predicted_val",
            x_remove_initial["p_churn"],
            global_step=trainer.global_step,
        )
        logger.experiment.add_histogram(
            "p_churn/last_element_predicted_val",
            x_last["p_churn"],
            global_step=trainer.global_step,
        )


def get_last(batch: Dict[str, torch.Tensor], model_out: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

    final = [i.item() - 1 for i in batch["offsets"][1:]]
    out = {k: v[final] for k, v in model_out.items()}

    return batch, out


def remove_initial_event(batch: Dict[str, torch.Tensor], model_out: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

    first = [i.item() for i in batch["offsets"][:-1]]  # first elements #batch["offsets"][:-1].tolist()
    inds = [k for k in range(batch["offsets"][-1]) if k not in first]
    out = {k: v[inds] for k, v in model_out.items()}

    return batch, out
