import collections
from typing import Any, Dict, List, Union, Optional

import numpy as np
import collections
import pytorch_lightning as pl
import torch
from torch import nn
import datetime
from torchmetrics import Accuracy, MeanAbsoluteError, MetricCollection

from neural_lifetimes.losses import ChurnLoss, SumLoss, TauLoss, VariationalEncoderDecoderLoss

from ...data.dataloaders.sequence_loader import trim_last
from ..nets.embedder import CombinedEmbedder
from ..nets.encoder_decoder import VariationalEncoderDecoder
from ..nets.event_model import EventEncoder
from ..nets.heads import CategoricalHead, ChurnProbabilityHead, CompositeHead, ExponentialHeadNoLoss, NormalHead

from neural_lifetimes.utils.data import FeatureDictionaryEncoder
from neural_lifetimes.utils import datetime2float, float2datetime

from neural_lifetimes.utils.callbacks import GitInformationLogger
import neural_lifetimes as nl
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
        feature_encoder_config: Dict[str, Any],
        rnn_dim: int,
        drop_rate: float,
        bottleneck_dim: int,
        lr: float,
        target_cols: List[str],
        vae_sample_z: bool = True,
        vae_sampling_scaler: float = 1.0,
        vae_KL_weight: float = 1.0,
        optimizer_kwargs: Dict[str, Any] = None,
    ) -> None:
        super().__init__()
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
            embed_dim=256,
            drop_rate=0.5,
        )

        self.event_encoder = EventEncoder(self.emb, rnn_dim, drop_rate)
        self.vae_sample_z = vae_sample_z
        self.vae_KL_weight = vae_KL_weight
        self.vae_sampling_scaler = vae_sampling_scaler
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs is not None else {}

        self.head = self.configure_heads(self.feature_encoder)
        self.net = VariationalEncoderDecoder(
            self.event_encoder,
            self.head,
            sample_z=vae_sample_z,
            epsilon_std=vae_sampling_scaler,
        )

        self.criterion = self.configure_criterion()
        # The awkward inclusion of git information is necessary for Tensorboard
        self.save_hyperparameters({**self.build_parameter_dict(), **GitInformationLogger().data_dict()})
        self.configure_metrics()
        self.inference_engine = nl.ModelInference(self)

    def build_parameter_dict(self) -> Dict[str, Any]:
        """Return a dictionary of parameters.

        Returns:
            Dict[str, Any]: Parameters of the ClassicModel instance
        """
        hparams = {
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
        return hparams
        # return {f"model/{k}": v for k, v in hparams.items()}

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

        if batch_idx % 100 == 0:
            self.forecasting_metrics(split)

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

    def forecasting_metrics(self, split):
        start = datetime.datetime.now()
        with torch.no_grad():
            forecast_seq = self.forecast(split)
            true_seq = self.get_true_data_forecast(split)

            # get predicted sequence lengths
            pred_seq_lengths = {k: [len(x['t']) for x in v] for k, v in forecast_seq.items()}
            mean_pred_seq_lengths = {k: np.array(v).mean() for k, v in pred_seq_lengths.items()}
            # get the true sequence lengths
            true_seq_lengths = {k: len(v['t']) for k, v in true_seq.items()}
            # get difference between sequence lengths
            diffs = {k: abs(mean_pred_seq_lengths[k] - true_seq_lengths[k]) for k in true_seq_lengths.keys()}
            mean_diff_length = np.array([*diffs.values()]).mean()
            # log sequence lengths
            self.log(f"{split}_forecast/pred_mean_num_events", np.array([*mean_pred_seq_lengths.values()]).mean())
            self.log(f"{split}_forecast/mae_num_events", mean_diff_length)

            # get predicted sequence intervals
            dts_pred = {k: torch.cat([v['dt'] for v in d]) for k, d in forecast_seq.items()}
            dts_pred_means = {k: (v.mean() if len(v) else torch.tensor(np.nan, device=v.device)) for k, v in dts_pred.items()}
            dts_pred_mean = torch.tensor([x.item() for x in dts_pred_means.values()]).nanmean().item()
            # get true sequence intervals
            # TODO: A lot of NAN values. why?
            dts_true_means = {k: v['dt'].mean() for k, v in true_seq.items()}
            dts_diffs = {k: abs(dts_true_means[k]-dts_pred_means[k]) for k in dts_true_means.keys()}
            dts_true_mean = torch.stack([*dts_diffs.values()]).nanmean().item()
            self.log(f"{split}_forecast/pred_mean_time_intervals", dts_pred_mean)
            self.log(f"{split}_forecast/mae_time_intervals", dts_true_mean)

            # calculate number of events
            # calculate last date
            # calculate time interval
            # calculate number of events times margin
            # calculate number of events times volume
            end = datetime.datetime.now()
        diff = end-start
        print(diff)

    def forecast(self, split):
        input_seq, predicted_seq = self.inference_engine.extend_sequence(
            loader=getattr(self.trainer.datamodule, f"{split}_dataloader")(),
            start_date=self.trainer.datamodule.forecast_dataset.asof_time,
            end_date=self.trainer.datamodule.forecast_dataset.last_event_time,
            n=2,
            return_input=True,
        )

        # trim sequences to only include forecasts
        assert len(input_seq) == len(predicted_seq)

        forecast_seq = trim_forecasts(input_seq, predicted_seq)
        return forecast_seq

    def get_true_data_forecast(self, split):
        asof_time = datetime2float(self.trainer.datamodule.dataset.asof_time)
        true_forecast_seq = {}
        for idx, batch in enumerate(getattr(self.trainer.datamodule, f"{split}_forecast_dataloader")()):
            for start, end in zip(batch["offsets"][:-1], batch["offsets"][1:]):
                usr_id = int(batch["USER_PROFILE_ID"][start + 1].item())
                past_asof = (batch["t"][start:end] > asof_time).sum().item()
                true_forecast_seq[usr_id] = {k: v[end-past_asof:end] for k, v in batch.items()}
        return true_forecast_seq


def trim_forecasts(input_seq, predicted_seq):
    # trim sequences to only include forecasts
    assert len(input_seq) == len(predicted_seq)

    data_seq = {}
    for batch_idx, batch in enumerate(input_seq):
        for start, end in zip(batch["offsets"][:-1], batch["offsets"][1:]):
            usr_id = int(batch["USER_PROFILE_ID"][start + 1].item())
            data_seq[usr_id] = {k: v[start:end] for k, v in batch.items()}

    pred_seq = collections.defaultdict(list)
    for batch_idx, batch in enumerate(predicted_seq):
        for start, end in zip(batch["offsets"][:-1], batch["offsets"][1:]):
            usr_id = int(batch["USER_PROFILE_ID"][start + 1].item())
            n_obs = len(data_seq[usr_id]["t"])  # number of observations including start token
            pred_seq[usr_id].append({k: v[(start + n_obs): end] for k, v in batch.items()})


    return pred_seq
