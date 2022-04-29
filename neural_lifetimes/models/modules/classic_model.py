from typing import Any, Dict, List, Union

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics import Accuracy, MeanAbsoluteError, MetricCollection

from neural_lifetimes.losses import ChurnLoss, SumLoss, TauLoss, VariationalEncoderDecoderLoss

from ...data.dataloaders.sequence_loader import trim_last
from ..nets.embedder import CombinedEmbedder
from ..nets.encoder_decoder import VariationalEncoderDecoder
from ..nets.event_model import EventEncoder
from ..nets.heads import CategoricalHead, ChurnProbabilityHead, CompositeHead, ExponentialHeadNoLoss, NormalHead

from .configure_optimizers import configure_optimizers


class ClassicModel(pl.LightningModule):  # TODO rename to VariationalGRUEncoder, Add better docstring
    """Initialises a ClassicModel instance.

    This is the model class. Each different model / method gets their own class.
    Docs: https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html

    Args:
        emb (CombinedEmbedder): a module to embed each data point.
        rnn_dim (int): dimensions of the recursive neural net feature vector.
        drop_rate (float): drop out rate in the model. 0 <= drop_rate < 1
        bottleneck_dim (int): dimensions of information bottleneck.
        lr (float): learning rate.
        **kwargs: Additional arguments for the pl.LighteningModule constructor
    """

    def __init__(
        self,
        # emb: CombinedEmbedder,
        data_config: Dict[str, Any],
        rnn_dim: int,
        drop_rate: float,
        bottleneck_dim: int,
        lr: float,
        # target_cols: List[str],
        vae_sample_z: bool = True,
        vae_sampling_scaler: float = 1.0,
        vae_KL_weight: float = 1.0,
        optimizer_kwargs: Dict[str, Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.data_config = data_config
        self.target_cols = self.data_config["targets"]

        self.rnn_dim = rnn_dim
        self.drop_rate = drop_rate
        self.bottleneck_dim = bottleneck_dim
        self.lr = lr

        self.vae_sample_z = vae_sample_z
        self.vae_KL_weight = vae_KL_weight
        self.vae_sampling_scaler = vae_sampling_scaler
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs is not None else {}

        # embedder
        self.emb = CombinedEmbedder(
            continuous_features=data_config["continuous_features"],
            discrete_features=data_config["discrete_features"],
            embed_dim=256,
            drop_rate=0.5,
        )

        # time series encoder
        self.rnn_encoder = EventEncoder(self.emb, rnn_dim, drop_rate)

        # decoder heads
        self.head = self.configure_heads(self.emb)

        # Encoder decoder
        self.net = VariationalEncoderDecoder(
            self.rnn_encoder,
            self.head,
            sample_z=vae_sample_z,
            epsilon_std=vae_sampling_scaler,
        )

        self.criterion = self.configure_criterion()
        self.save_hyperparameters(self.build_parameter_dict())
        self.configure_metrics()

    def on_fit_start(self) -> None:
        # encoder = self.trainer.datamodule.transform
        print("A")
        return super().on_fit_start()

    def build_parameter_dict(self) -> Dict[str, Any]:
        """Return a dictionary of parameters.

        Returns:
            Dict[str, Any]: Parameters of the ClassicModel instance
        """
        return {
            "rnn_dim": self.rnn_dim,
            "drop_rate": self.drop_rate,
            "bottleneck_dim": self.bottleneck_dim,
            "lr": self.lr,
            "target_cols": self.target_cols,
            "vae_sample_z": self.vae_sample_z,
            "vae_sampling_scaler": self.vae_sampling_scaler,
            "vae_KL_weight": self.vae_KL_weight,
            **self.emb.build_parameter_dict(),
        }

    def configure_optimizers(self):
        return configure_optimizers(
            self.parameters(), self.lr, optimizer="Adam", scheduler="ReduceLROnPlateau", **self.optimizer_kwargs
        )

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Conducts a forward pass of this model.

        Args:
            x (Dict[str, torch.Tensor]): batch input

        Returns:
            Dict[str, torch.Tensor]: model output
        """
        pred = self.net.forward(x)
        return pred

    def configure_criterion(self) -> nn.Module:
        """Configures a loss function. This might be a composite function.

        Returns:
            nn.Module: Module that returns loss function on __call__
        """
        pre_loss_ = self.head.loss_function(preprocess=trim_last)

        pre_loss = SumLoss(
            {
                "composite": pre_loss_,
                "tau": TauLoss(),
                "churn": ChurnLoss(self.head.heads["next_dt"], scale_by_seq=False),
            }
        )
        loss_fn = VariationalEncoderDecoderLoss(pre_loss, reg_weight=self.vae_KL_weight if self.vae_sample_z else None)
        return loss_fn

    def configure_heads(self, emb: nn.Module) -> nn.Module:
        def categorical_head_template(feature: str):
            return CategoricalHead(
                self.bottleneck_dim,
                1 if feature not in emb.discrete_features else len(emb.discrete_features[feature]) + 1,
                self.drop_rate,
            )

        head_continuous = {
            f"next_{f}": NormalHead(self.bottleneck_dim, self.drop_rate) for f in emb.continuous_features
        }
        head_discrete = {f"next_{f}": categorical_head_template(f) for f in emb.discrete_features.keys()}
        head_extra = {
            # "next_btyd_mode": categorical_head_template("btyd_mode"),
            "next_dt": ExponentialHeadNoLoss(self.bottleneck_dim, self.drop_rate),
            "p_churn": ChurnProbabilityHead(self.bottleneck_dim, self.drop_rate),
        }

        self.head_map = {**head_continuous, **head_discrete, **head_extra}

        return CompositeHead(self.head_map)

    def configure_metrics(self) -> nn.ModuleDict:
        """Configures metrics for this model.

        Torchmetrics implements metric classes that will be called each step and then automatically compute
        compute metrics on certain events triggered by pytorch lightning. Here we configure these metrics as
        classes, to be called later.

        Returns:
            nn.ModuleDict[str, torchmetrics.Metric]: dictionary with metric instances
        """
        # add MAE for continuous features
        metrics_cont_feat = {n: MeanAbsoluteError() for n in self.emb.continuous_features}
        metrics_discr_feat = {n: Accuracy() for n in self.emb.discrete_features}

        metrics = MetricCollection({**metrics_cont_feat, **metrics_discr_feat})

        self.metrics = nn.ModuleDict(
            {
                "train_metrics": metrics.clone(prefix="train_metrics/"),
                "val_metrics": metrics.clone(prefix="val_metrics/"),
                "test_metrics": metrics.clone(prefix="test_metrics/"),
            }
        )

    def get_and_log_loss(self, y_pred: Dict[str, torch.Tensor], y_true: Dict[str, Any], split: str) -> torch.Tensor:
        """Calculates and logs loss.

        Args:
            y_pred (Dict[str, torch.Tensor]): model output
            y_true (Dict[str, Any]): batch
            split (str): the split of the step: (i.e. train, val, test)

        Returns:
            torch.Tensor: loss value
        """
        loss, loss_components = self.criterion(y_pred, y_true)
        loss_components = {f"{split}_{name}": loss for name, loss in loss_components.items()}
        self.log_dict(loss_components)
        return loss

    def get_and_log_clv(self, output, batch, split):
        """Estimate CLV and compare to dataset.

        Args:
            output (Dict[str, torch.Tensor]): model output
            batch (Dict[str, Any]): batch
            split (str): the split of the step: (i.e. train, val, test)

        Returns:
            metrics (Dict[str, torch.Tensor]): dictionary with metrics

        Notes:
            Currently only returns an empty dictionary -- is this intended?
        """
        return dict()
        # parametric estimate of CLV
        par_clv = self.trainer.datamodule.val_dataset._get_E_num_transactions(
            output["p_churn"].mean(), 1 / output["next_dt"].mean()
        )
        # par_clv = self.trainer.datamodule.val_dataset._get_E_num_transactions(
        #   output['p_churn'], 1/output['t_to_next']
        # )

        # non-parametric
        users, user_idx, user_transactions = np.unique(batch["USER_PROFILE_ID"], return_index=True, return_counts=True)
        users_true_clv = batch["clv"][
            user_idx
        ]  # TODO is this correct? Are we guaranteeing all transactions of the same user in the same batch?

        # calculate metrics
        metrics = {}
        metrics["par_clv_mean"] = par_clv.mean()
        metrics["par_clv_l1_error"] = torch.abs(par_clv.squeeze() - batch["clv"]).mean()
        metrics["npar_clv_mean"] = user_transactions.mean()
        metrics["npar_clv_var"] = user_transactions.var()
        metrics["npar_clv_l1_error"] = np.abs(user_transactions - users_true_clv.detach().cpu().numpy()).mean()

        # add split to metric names
        metrics = {f"{split}_{k}": v for k, v in metrics.items()}

        return metrics

    def get_and_log_metrics(
        self, y_pred: Dict[str, torch.Tensor], y_true: Dict[str, Any], split: str
    ) -> Dict[str, torch.Tensor]:
        """Update metric instances with each batch predictions and ground truths.

        Further, log them.

        Args:
            y_pred (Dict[str, torch.Tensor]): model output
            y_true (Dict[str, Any]): the batch
            split (str): the split of the step. (i.e. train, val or test)

        Returns:
            Dict[str, torch.Tensor]: dictionary of metrics for this batch.
        """
        metric_values = {}

        # for name, encoder in self.emb.discrete_features.items():
        #     discr_pred = y_pred[f"next_{name}"].argmax(dim=-1, keepdim=True).detach().cpu()
        #     discr_true = torch.tensor(encoder.transform(y_true[name]))
        #     metric_values[name] = self.metrics[f"{split}_metrics"][name](discr_pred, discr_true)

        for name in self.emb.continuous_features:
            cont_pred = y_pred[f"next_{name}"][:, 0]
            metric_values[name] = self.metrics[f"{split}_metrics"][name](cont_pred, y_true[name])

        # clv_metrics = self.get_and_log_clv(y_pred, y_true, split)
        # metric_values = {**metric_values, **clv_metrics}

        # TODO: once implemented, actually calculate metrics
        # for name, metric in self.metrics.items():
        #     if name.startswith(split + "_"):
        #         # calculate metrics
        #         metric_values[name] = metric(y_pred, y_true)

        # TODO: add logging for estimates of p and lambda

        metric_values = {f"{split}_metrics/{k}": v for k, v in metric_values.items()}

        self.log_dict(metric_values)
        return metric_values

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self.all_split_step(split="train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self.all_split_step(split="val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self.all_split_step(split="test", *args, **kwargs)

    def all_split_step(
        self, batch: Dict[str, Any], batch_idx: int, split: str
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """As steps in all splits have more in common than divides them, this is a universal step function.

        Args:
            batch (Dict[str, Any]): batch from dataloader
            batch_idx (int): index of the batch per epoch
            split (str): split of the step. ie. train, val or test

        Returns:
            Union[torch.Tensor, Dict[str, torch.Tensor]]: returns loss for training steps and metrics else.
                These values will be fed into callbacks and backward propagation.
        """
        output = self.forward(batch)

        # loss = self.get_and_log_loss(output, target, split)
        loss = self.get_and_log_loss(output, batch, split)
        self.get_and_log_metrics(output, batch, split)

        if split == "train":
            return loss
        else:
            return output

    def load_state_from_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load the model state from a checkpoint.

        Args:
            checkpoint_path (str): Path to the model checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path)
        model_dict = self.state_dict()
        checkpoint_dict = checkpoint["state_dict"]
        pretrained_dict = {k: v for k, v in checkpoint_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
