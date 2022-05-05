import collections
import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributions as d

from torchmetrics import MetricCollection

from neural_lifetimes.metrics import KullbackLeiblerDivergence, ParametricKullbackLeiblerDivergence, WassersteinMetric


class DistributionMonitor(pl.Callback):
    """
    Class to monitor the distribution of the model. Records the various parameters and/or metrics of the model.
    """

    def __init__(
        self,
        max_batches: int = 10,
        clip_t: bool = True,
        clip_lambda: bool = True,  # TODO: Only works when lambdas is passed in as a variable
        clip_t_quantile: float = 0.98,
        clip_lambda_quantile: float = 0.98,
    ):

        super().__init__()  # TODO: remove default clipping and change to auto or none
        self.max_batches = max_batches
        self.clip_t = clip_t
        self.clip_lambda = clip_lambda
        self.clip_t_val = float("inf")
        self.clip_lambda_val = float("inf")
        self.clip_t_quantile = clip_t_quantile
        self.clip_lambda_quantile = clip_lambda_quantile
        # This metric collection should not be called together, but eases computation and resetting the metrics
        self.metrics = MetricCollection(
            {
                "log_t_to_next_KLDiv": KullbackLeiblerDivergence(),
                "t_to_next_KLDiv": KullbackLeiblerDivergence(),
                "t_to_next_ParKLDiv": ParametricKullbackLeiblerDivergence(torch.distributions.Exponential),
                "t_to_next_Wasserstein": WassersteinMetric(),
            }
        )

        self.data = collections.defaultdict(list)

    def on_train_start(self, trainer, net):
        """
        Training start callback.

        The following parameters are recorded whenever it exists:

        - the actual lambda
        - the actual p_churn
        - the time to next event
        """
        loader = trainer.datamodule.val_dataloader()
        logger = trainer.loggers[0]

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

        if len(lambdas) > 0:
            lambdas = np.concatenate(lambdas)
            self.clip_lambda_val = np.quantile(lambdas, self.clip_lambda_quantile)
            logger.experiment.add_histogram(
                "time_intervals/lambda_True", np.clip(lambdas, 0, self.clip_lambda), global_step=trainer.global_step
            )
        else:
            if self.clip_lambda:
                raise ValueError(
                    "Auto-clipping plots is not supported when `lambda` is not in batch. "
                    + "Please try again using `clip_lambda=False`."
                )

        if len(p) > 0:
            p = np.concatenate(p)
            logger.experiment.add_histogram("p_churn/True", p, global_step=trainer.global_step)

        t_to_next = torch.cat(t_to_next)
        if self.clip_t:
            self.clip_t_val = torch.quantile(t_to_next, self.clip_t_quantile).item()

        logger.experiment.add_histogram(
            "time_intervals/t_to_next_True", torch.clamp(t_to_next, 0, self.clip_t_val), global_step=trainer.global_step
        )
        logger.experiment.add_histogram(
            "time_intervals/log_t_to_next_True",
            torch.log(t_to_next + 0.001),
            global_step=trainer.global_step,  # add 0.001 for make log(0) finite.
        )

    # TODO update call signature: remove `unused` as soon as plt patches
    def on_validation_batch_end(self, trainer: pl.Trainer, net: pl.LightningModule, output, batch, batch_idx, unused):
        if batch_idx > self.max_batches:
            return

        out = net(batch)
        batch_offsets = torch.tensor([o - i for i, o in enumerate(batch["offsets"])])[1:]

        y_true = batch["next_dt"][batch_offsets - 1]  # TODO check padding
        y_true = y_true.unsqueeze(dim=-1)
        y_pred = out["next_dt"][batch["offsets"][1:].detach().cpu().numpy() - 2]  # TODO why only last?

        self.data["y_pred"].append(y_pred)

        self.metrics["t_to_next_KLDiv"].update(y_pred, y_true)
        self.metrics["log_t_to_next_KLDiv"].update(y_pred, y_true)
        self.metrics["t_to_next_Wasserstein"].update(y_pred, y_true)

        lambda_pred = 1 / (y_pred + 1e-6)
        lambda_true = 1 / (y_true + 1e-6)

        self.metrics["t_to_next_ParKLDiv"].update(lambda_pred, lambda_true)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        y_pred = torch.cat(self.data["y_pred"])

        lambda_pred = 1 / (y_pred + 1e-6)

        dist = d.exponential.Exponential(lambda_pred)
        sampled_t_from_lambda_pred: torch.Tensor = dist.rsample(torch.Size([1000])).flatten()

        lambda_pred = lambda_pred.detach().cpu().numpy()
        y_pred = y_pred.cpu().detach().numpy()
        sampled_t_from_lambda_pred = sampled_t_from_lambda_pred.cpu().detach().numpy()

        logger = trainer.loggers[0]
        logger.experiment.add_histogram(
            "time_intervals/lambda_Pred",
            np.clip(lambda_pred, 0, self.clip_lambda_val),
            global_step=trainer.global_step,
        )
        logger.experiment.add_histogram(
            "time_intervals/t_to_next_Pred",
            np.clip(y_pred, 0, self.clip_t_val),
            global_step=trainer.global_step,
        )
        logger.experiment.add_histogram(
            "time_intervals/log_t_to_next_Pred",
            np.log(y_pred),
            global_step=trainer.global_step,
        )
        logger.experiment.add_histogram(
            "time_intervals/t_to_next_sampled_Pred",
            np.clip(sampled_t_from_lambda_pred, 0, self.clip_t_val),
            global_step=trainer.global_step,
        )
        logger.experiment.add_histogram(
            "time_intervals/log_t_to_next_sampled_Pred",
            np.log(sampled_t_from_lambda_pred),
            global_step=trainer.global_step,
        )

        # TODO better ways to approximate KL empirically:
        # https://www.semanticscholar.org/paper/Kullback-Leibler-divergence-estimation-of-P%C3%A9rez-Cruz/310974d3c141589a7800d737e5859b76676dcb5d?p2df
        metrics = self.metrics.compute()
        pl_module.log_dict({f"time_intervals/{k}": v for k, v in metrics.items()}, batch_size=len(y_pred))

        self.metrics.reset()
