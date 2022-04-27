from typing import Optional
import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributions as d

from neural_lifetimes.metrics import KullbackLeiblerDivergence, ParametricKullbackLeiblerDivergence, WassersteinMetric


class MonitorDistr(pl.Callback):
    """
    Class to monitor the distribution of the model. Records the various parameters and/or metrics of the model.
    """

    def __init__(
        self,
        max_batches: int = 10,
        log_every_n_steps: Optional[int] = None,
        clip_t: bool = True,
        clip_lambda: bool = True,  # TODO: Only works when lambdas is passed in as a variable
        clip_t_quantile: float = 0.98,
        clip_lambda_quantile: float = 0.98,
    ):

        super().__init__()  # TODO: remove default clipping and change to auto or none
        self.max_batches = max_batches
        self.log_every_n_steps = log_every_n_steps
        self.clip_t = clip_t
        self.clip_lambda = clip_lambda
        self.clip_t_val = float("inf")
        self.clip_lambda_val = float("inf")
        self.clip_t_quantile = clip_t_quantile
        self.clip_lambda_quantile = clip_lambda_quantile
        self.kl_log_time_interval = KullbackLeiblerDivergence()
        self.kl_time_interval = KullbackLeiblerDivergence()
        self.par_kl_time_interval = ParametricKullbackLeiblerDivergence(torch.distributions.Exponential)
        self.sinkhorn_time_interval = WassersteinMetric()

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

        if len(lambdas) > 1:
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

        if len(p) > 1:
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
        log_every_n = self.log_every_n_steps if self.log_every_n_steps is not None else trainer.log_every_n_steps
        if not batch_idx % log_every_n == 0:  # trainer.log_every_n_steps
            return

        out = net(batch)
        batch_offsets = torch.tensor([o - i for i, o in enumerate(batch["offsets"])])[1:]
        y_true = batch["next_dt"][batch_offsets - 1]  # TODO check padding
        lambda_true = 1 / (y_true + 1e-6)
        y_pred = out["next_dt"][batch["offsets"][1:].detach().cpu().numpy() - 2]  # TODO why only last?
        lambda_pred = 1 / y_pred

        dist = d.exponential.Exponential(lambda_pred)
        sampled_t_from_lambda_Pred: torch.Tensor = dist.rsample(torch.Size([1000])).flatten()

        logger = trainer.loggers[0]
        logger.experiment.add_histogram(
            "time_intervals/lambda_Pred",
            np.clip(lambda_pred.cpu().detach().numpy(), 0, self.clip_lambda_val),
            global_step=trainer.global_step,
        )
        logger.experiment.add_histogram(
            "time_intervals/t_to_next_Pred",
            np.clip(y_pred.cpu().detach().numpy(), 0, self.clip_t_val),
            global_step=trainer.global_step,
        )
        logger.experiment.add_histogram(
            "time_intervals/log_t_to_next_Pred",
            torch.log(y_pred).cpu().detach().numpy(),
            global_step=trainer.global_step,
        )
        logger.experiment.add_histogram(
            "time_intervals/t_to_next_sampled_Pred",
            np.clip(sampled_t_from_lambda_Pred.cpu().detach().numpy(), 0, self.clip_t_val),
            global_step=trainer.global_step,
        )
        logger.experiment.add_histogram(
            "time_intervals/log_t_to_next_sampled_Pred",
            torch.log(sampled_t_from_lambda_Pred).cpu().detach().numpy(),
            global_step=trainer.global_step,
        )

        # TODO better ways to approximate KL empirically:
        # https://www.semanticscholar.org/paper/Kullback-Leibler-divergence-estimation-of-P%C3%A9rez-Cruz/310974d3c141589a7800d737e5859b76676dcb5d?p2df
        divergences = {}
        log_p_pred, log_p_true = self.histogram(torch.log(y_pred + 1e-6), torch.log(y_true + 1e-6))
        divergences["log_t_to_next_KLDiv"] = self.kl_log_time_interval(log_p_pred, log_p_true)
        log_p_pred, log_p_true = self.histogram(y_pred, y_true)
        divergences["t_to_next_KLDiv"] = self.kl_time_interval(log_p_pred, log_p_true)
        divergences["t_to_next_Wasserstein"] = self.sinkhorn_time_interval(y_pred.flatten(), y_true)
        divergences["t_to_next_ParKLDiv"] = self.par_kl_time_interval(lambda_pred.flatten(), lambda_true)
        net.log_dict({f"time_intervals/{k}": v for k, v in divergences.items()})

    def histogram(self, preds, target, nbins=50):
        # flatten if needed
        preds = preds.flatten()
        target = target.flatten()

        MIN = min(preds.min(), target.min())
        MAX = max(preds.max(), target.max())
        edges = torch.linspace(MIN, MAX, steps=nbins)
        pred_log_prob = torch.log((torch.histogram(preds, edges)[0] + 1.0e-6) / len(preds))
        target_log_prob = torch.log((torch.histogram(target, edges)[0] + 1.0e-6) / len(target))

        return pred_log_prob, target_log_prob
