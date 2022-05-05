import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Tuple
import numpy as np

import pytest
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.loggers import CSVLogger

from neural_lifetimes import run_model
from neural_lifetimes.data.datamodules.sequence_datamodule import SequenceDataModule
from neural_lifetimes.data.datasets.btyd import BTYD, GenMode
from neural_lifetimes.models.modules import ClassicModel
from neural_lifetimes.utils.data import Tokenizer, FeatureDictionaryEncoder, TargetCreator

from ..test_datasets.datamodels import EventprofilesDataModel

DISCRETE_START_TOKEN = "StartToken"


@pytest.fixture
def log_dir() -> str:
    with TemporaryDirectory() as f:
        yield f


# @pytest.fixture
# def data_dir() -> str:
#     return str(Path(__file__).parents[2] / "examples")


@pytest.fixture
def data_and_model() -> Tuple[SequenceDataModule, ClassicModel]:
    # create btyd data and dependent modules
    data_model = EventprofilesDataModel()

    dataset = BTYD.from_modes(
        modes=[GenMode(a=1, b=3, r=1, alpha=15)],
        num_customers=100,
        seq_gen_dynamic=True,
        start_date=datetime.datetime(2019, 1, 1, 0, 0, 0),
        start_limit_date=datetime.datetime(2019, 6, 15, 0, 0, 0),
        end_date=datetime.datetime(2021, 1, 1, 0, 0, 0),
        data_dir=str(Path(__file__).parents[2] / "examples"),
        continuous_features=data_model.cont_feat,
        discrete_features=data_model.discr_feat,
    )

    discr_values = dataset.get_discrete_feature_values("<StartToken>")

    transform = FeatureDictionaryEncoder(data_model.cont_feat, discr_values)
    target_transform = TargetCreator(cols=data_model.target_cols + data_model.cont_feat + data_model.discr_feat)
    tokenizer = Tokenizer(data_model.cont_feat, discr_values, 100, np.nan, "<StartToken>", np.nan)

    datamodule = SequenceDataModule(
        dataset=dataset,
        transform=transform,
        target_transform=target_transform,
        tokenizer=tokenizer,
        test_size=0.2,
        batch_points=256,
        min_points=1,
    )

    # create model
    model = ClassicModel(
        feature_encoder_config=transform.config_dict(),
        rnn_dim=256,
        drop_rate=0.5,
        bottleneck_dim=32,
        lr=0.001,
        target_cols=target_transform.cols,
        vae_sample_z=True,
        vae_sampling_scaler=1.0,
        vae_KL_weight=0.01,
    )
    return datamodule, model


@pytest.mark.slow
class TestRunModel:
    @pytest.mark.parametrize("run_mode", ("train", "test", "none"))
    def test_defaults_with_run(
        self,
        log_dir: str,
        data_and_model: Tuple[LightningDataModule, LightningModule],
        run_mode: str,
    ) -> None:
        datamodule, model = data_and_model
        run_model(
            datamodule,
            model,
            log_dir=log_dir,
            num_epochs=2,
            run_mode=run_mode,
        )

    def test_custom_trainer_kwargs(
        self,
        log_dir: str,
        data_and_model: Tuple[LightningDataModule, LightningModule],
    ) -> None:
        datamodule, model = data_and_model
        trainer = run_model(
            datamodule,
            model,
            log_dir=log_dir,
            checkpoint_path=None,
            run_mode="none",
            trainer_kwargs={"accumulate_grad_batches": 2, "callbacks": []},
        )
        assert len(trainer.callbacks) <= 4
        # progressbar, model summary and model checkpoints are turned on by default.
        # Gradient Accummulation also enabled => max 4 callbacks expected
        assert trainer.accumulate_grad_batches == 2

    def test_custom_logger_class(
        self,
        log_dir: str,
        data_and_model: Tuple[LightningDataModule, LightningModule],
    ) -> None:
        datamodule, model = data_and_model
        loggers = [CSVLogger(save_dir=log_dir)]
        trainer = run_model(
            datamodule,
            model,
            log_dir=log_dir,
            checkpoint_path=None,
            run_mode="none",
            loggers=loggers,
        )
        assert isinstance(trainer.logger, CSVLogger)

    def test_custom_logger_kwargs(
        self,
        log_dir: str,
        data_and_model: Tuple[LightningDataModule, LightningModule],
    ) -> None:
        datamodule, model = data_and_model
        logger_kwargs = {"save_dir": "directory123", "version": 42}
        trainer = run_model(
            datamodule,
            model,
            log_dir=log_dir,
            checkpoint_path=None,
            run_mode="none",
            logger_kwargs=logger_kwargs,
        )
        assert trainer.logger.version == 42
        assert trainer.logger.save_dir == "directory123"

    def test_custom_logger_raises_error(
        self,
        log_dir: str,
        data_and_model: Tuple[LightningDataModule, LightningModule],
    ) -> None:
        datamodule, model = data_and_model
        loggers = [CSVLogger(save_dir=log_dir)]
        logger_kwargs = {"version": 42}
        with pytest.raises(AssertionError) as excinfo:
            _ = run_model(
                datamodule,
                model,
                log_dir=log_dir,
                checkpoint_path=None,
                run_mode="none",
                loggers=loggers,
                logger_kwargs=logger_kwargs,
            )
        (msg,) = excinfo.value.args
        assert msg == "If custom logger is supplied, 'logger_kwargs' argument must be 'None'"
