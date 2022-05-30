import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import pytest

from neural_lifetimes import run_model
from neural_lifetimes.data.datamodules.sequence_datamodule import SequenceDataModule
from neural_lifetimes.data.datasets.btyd import BTYD, GenMode
from neural_lifetimes.models.modules import ClassicModel
from neural_lifetimes.utils.data import FeatureDictionaryEncoder, TargetCreator, Tokenizer

from ..test_datasets.datamodels import EventprofilesDataModel

DISCRETE_START_TOKEN = "StartToken"


@pytest.fixture
def log_dir() -> str:
    with TemporaryDirectory() as f:
        yield f


@pytest.fixture
def data_dir() -> str:
    return str(Path(__file__).parents[2] / "examples")


@pytest.mark.slow
class TestClassicModel:
    @pytest.mark.parametrize("vae_sample_z", ("True", "False"))
    def test_btyd(self, vae_sample_z: bool, log_dir: str, data_dir: str) -> None:
        # create btyd data and dependent modules
        data_model = EventprofilesDataModel()

        dataset = BTYD.from_modes(
            modes=[GenMode(a=1, b=3, r=1, alpha=15)],
            num_customers=100,
            seq_gen_dynamic=True,
            start_date=datetime.datetime(2019, 1, 1, 0, 0, 0),
            start_limit_date=datetime.datetime(2019, 6, 15, 0, 0, 0),
            end_date=datetime.datetime(2021, 1, 1, 0, 0, 0),
            data_dir=data_dir,
            continuous_features=data_model.cont_feat,
            discrete_features=data_model.discr_feat,
        )

        discr_values = dataset.get_discrete_feature_values("<StartToken>")

        tokenizer = Tokenizer(
            data_model.cont_feat,
            discr_values,
            100,
            np.nan,
            "<StartToken>",
            datetime.datetime(1970, 1, 1, 0, 0, 0),
            np.nan,
        )
        transform = FeatureDictionaryEncoder(data_model.cont_feat, discr_values)

        target_transform = TargetCreator(cols=(data_model.target_cols + data_model.cont_feat + data_model.discr_feat))
        datamodule = SequenceDataModule(
            dataset=dataset,
            tokenizer=tokenizer,
            transform=transform,
            target_transform=target_transform,
            test_size=0.2,
            batch_points=256,
            min_points=1,
        )

        # create model
        model = ClassicModel(
            feature_encoder_config=transform.config_dict(),
            rnn_dim=256,
            emb_dim=256,
            drop_rate=0.5,
            bottleneck_dim=32,
            lr=0.001,
            target_cols=target_transform.cols,
            vae_sample_z=vae_sample_z,
            vae_sampling_scaler=1.0,
            vae_KL_weight=0.01,
        )

        run_model(
            datamodule,
            model,
            log_dir=log_dir,
            num_epochs=2,
            val_check_interval=2,
            limit_val_batches=2,
        )

        # TODO add tests for: different rnn_dims, drop_rates, bottleneck_dims.
        # TODO add failure tests for negative learning rate, negative sacler and negative KL weights
