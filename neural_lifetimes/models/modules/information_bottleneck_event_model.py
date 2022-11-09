from typing import Any, Dict, List, Tuple, Union

import torch
from torch import nn
from torchmetrics import Accuracy, MeanAbsoluteError, MetricCollection

from neural_lifetimes.losses import ChurnLoss, SumLoss, TauLoss
from neural_lifetimes.models.modules import EventModel

from ...data.dataloaders.sequence_loader import trim_last
from ..nets.embedder import CombinedEmbedder
from ..nets.event_model import EventEncoder
from ..nets.heads import CategoricalHead, ChurnProbabilityHead, CompositeHead, ExponentialHeadNoLoss, NormalHead
from neural_lifetimes.losses import InformationBottleneckLoss
from neural_lifetimes.utils.scheduler import LinearWarmupScheduler
from neural_lifetimes.utils.data import FeatureDictionaryEncoder

from neural_lifetimes.utils.callbacks import GitInformationLogger
from .configure_optimizers import configure_optimizers


class InformationBottleneckEventModel(EventModel):  # TODO Add better docstring
    """Initialises an EventModel instance.

    This is the model class. Each different model / method gets their own class.
    Docs: https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html

    Args:
        emb (CombinedEmbedder): a module to embed each data point.
        rnn_dim (int): dimensions of the recursive neural net feature vector.
        drop_rate (float): drop out rate in the model. 0 <= drop_rate < 1
        bottleneck_dim (int): dimensions of information bottleneck.
        lr (float): learning rate.
        **kwargs: Additional arguments for the pl.LighteningModule constructor
        encoder_noise: the noise for the encoder, scalar of N(0,1)
    """

    def __init__(
        self,
        feature_encoder_config: Dict[str, Any],
        emb_dim: int,
        rnn_dim: int,
        drop_rate: float,
        bottleneck_dim: int,
        lr: float,
        target_cols: List[str],
        loss_cfg: Dict[str, Any],
        optimizer_kwargs: Dict[str, Any] = None,
        log_git_information: bool = False,
        config_dict_to_log: Dict[str, Any] = None,
        encoder_noise: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.rnn_dim = rnn_dim
        self.drop_rate = drop_rate
        self.bottleneck_dim = bottleneck_dim
        self.lr = lr
        self.target_cols = target_cols
        # the reconstruction of the feature encoder requires it to be deterministic
        self.feature_encoder = FeatureDictionaryEncoder.from_dict(feature_encoder_config)

        # TODO add arguments for setting NN parameters
        self.emb = CombinedEmbedder(
            feature_encoder=self.feature_encoder,
            embed_dim=emb_dim,
            drop_rate=drop_rate,
        )

        self.loss_cfg = loss_cfg

        self.event_encoder = EventEncoder(self.emb, rnn_dim, drop_rate)
        self.encoder_noise = encoder_noise
        self.project_encoding = nn.Linear(in_features=rnn_dim, out_features=bottleneck_dim)
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs is not None else {}

        self.head = self.configure_heads(self.feature_encoder)

        self.criterion = self.configure_criterion()
        # The awkward inclusion of git information is necessary for Tensorboard
        # If cfg_dict is passed in log that otherwise collect parameter dict.
        if config_dict_to_log is None:
            cfg_dict = {**self.build_parameter_dict()}
        else:
            cfg_dict = config_dict_to_log
        if log_git_information:
            cfg_dict.update({**GitInformationLogger().data_dict()})
        self.save_hyperparameters(cfg_dict)
        self.configure_metrics()

    def build_parameter_dict(self) -> Dict[str, Any]:
        """Return a dictionary of parameters.

        Returns:
            Dict[str, Any]: Parameters of the EventModel instance
        """
        hparams = {  # TODO add IB parameters
            "rnn_dim": self.rnn_dim,
            "drop_rate": self.drop_rate,
            "bottleneck_dim": self.bottleneck_dim,
            "lr": self.lr,
            "target_cols": self.target_cols,
            **self.emb.build_parameter_dict(),
        }

        return {f"model/{k}": v for k, v in hparams.items()}

    def configure_optimizers(self):
        kwargs = {
            "optimizer": "Adam",
            "scheduler": "ReduceLROnPlateau",
        }
        kwargs.update(self.optimizer_kwargs)

        return configure_optimizers(
            parameters=self.parameters(),
            lr=self.lr,
            **kwargs,
        )

    def forward(self, x: Dict[str, torch.Tensor], ib_only_last_event: bool = True) -> Dict[str, torch.Tensor]:
        """Conducts a forward pass of this model.

        Args:
            x (Dict[str, torch.Tensor]): batch input
            ib_only_last_event (bool): If `True`, the information bottleneck penalty is only applied to the last item
                in a sequence. Else it is applied to all items. Default is True.

        Returns:
            Dict[str, torch.Tensor]: model output
        """
        out = self.event_encoder(x)
        event_encoding = out + torch.randn_like(out) * self.encoder_noise
        bottleneck = self.project_encoding(event_encoding)

        out = self.head(bottleneck)

        if ib_only_last_event:
            last_elements = x["offsets"][1:] - 1
            bottleneck = bottleneck.index_select(0, last_elements)
            event_encoding = event_encoding.index_select(0, last_elements)

        out["bottleneck"] = bottleneck
        out["event_encoding"] = event_encoding
        return out

    def encode(self, x: Dict[str, torch.Tensor], stochastic: bool = False) -> Tuple[torch.Tensor]:
        out = self.event_encoder(x)
        event_encoding = out + torch.randn_like(out) * self.encoder_noise * int(stochastic)
        return (self.project_encoding(event_encoding), None)

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
        weight_scheduler = LinearWarmupScheduler(
            n_cold_steps=self.loss_cfg["n_cold_steps"],
            n_warmup_steps=self.loss_cfg["n_warmup_steps"],
            target_weight=self.loss_cfg["target_weight"],
        )
        loss_fn = InformationBottleneckLoss(
            fit_loss=pre_loss,
            weight_scheduler=weight_scheduler,
            n_eigen=self.loss_cfg["n_eigen"],
            n_eigen_threshold=self.loss_cfg["n_eigen_threshold"],
        )
        return loss_fn

    def configure_heads(self, feature_encoder: FeatureDictionaryEncoder) -> nn.Module:
        def categorical_head_template(feature: str):
            return CategoricalHead(
                self.bottleneck_dim,
                1 if feature not in feature_encoder.discrete_features else feature_encoder.feature_size(feature),
                self.drop_rate,
            )

        head_continuous = {
            f"next_{f}": NormalHead(self.bottleneck_dim, self.drop_rate) for f in feature_encoder.continuous_features
        }
        head_discrete = {f"next_{f}": categorical_head_template(f) for f in feature_encoder.discrete_features.keys()}
        head_extra = {
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
        metrics_cont_feat = {n: MeanAbsoluteError() for n in self.feature_encoder.continuous_features}
        metrics_discr_feat = {n: Accuracy() for n in self.feature_encoder.discrete_features}

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
        self.log_dict(loss_components, batch_size=y_true["next_dt"].shape[0])
        self.log("ib_weight", self.criterion.weight_scheduler.step())
        return loss

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

        for name in self.feature_encoder.discrete_features:
            discr_pred = y_pred[f"next_{name}"].argmax(dim=-1, keepdim=True)
            metric_values[name] = self.metrics[f"{split}_metrics"][name](discr_pred.squeeze(), y_true[name])

        for name in self.feature_encoder.continuous_features:
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

        self.log_dict(metric_values, batch_size=y_true["next_dt"].shape[0])
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
