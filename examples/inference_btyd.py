import pytorch_lightning as pl
from pathlib import Path
import datetime
import numpy as np

from neural_lifetimes.data.dataloaders.sequence_loader import SequenceLoader
from neural_lifetimes.data.datamodules import SequenceDataModule
import neural_lifetimes as nl
from neural_lifetimes.data.datasets.btyd import BTYD, GenMode
from neural_lifetimes.utils.data import Tokenizer, TargetCreator, FeatureDictionaryEncoder

from examples import eventsprofiles_datamodel


checkpoint_path = (
    "/Users/Cornelius.Emde/Repositories/NeuralLifetimes/examples/version_4/epoch=0-step=10-val_loss/total=61.19.ckpt"
)

LOG_DIR = str(Path(__file__).parent)
data_dir = str(Path(__file__).parent.absolute())
START_TOKEN_DISCR = "<StartToken>"
COLS = eventsprofiles_datamodel.target_cols + eventsprofiles_datamodel.cont_feat + eventsprofiles_datamodel.discr_feat
START_DATE = datetime.datetime(2019, 1, 1, 0, 0, 0)
END_DATE = datetime.datetime(2021, 1, 1, 0, 0, 0)

if __name__ == "__main__":
    pl.seed_everything(9473)

    dataset = BTYD.from_modes(
        modes=[
            GenMode(a=1.5, b=20, r=1, alpha=14),
            GenMode(a=2, b=50, r=2, alpha=6),
        ],
        num_customers=1000,
        mode_ratios=[2.5, 1],  # generate equal number of transactions from each mode
        seq_gen_dynamic=False,
        start_date=START_DATE,
        start_limit_date=datetime.datetime(2019, 6, 15, 0, 0, 0),
        end_date=END_DATE,
        data_dir=data_dir,
        continuous_features=eventsprofiles_datamodel.cont_feat,
        discrete_features=eventsprofiles_datamodel.discr_feat,
        track_statistics=True,
    )

    # btyd_dataset[:]
    # print(f"Expected Num Transactions per mode: {btyd_dataset.expected_num_transactions_from_priors()}")
    # print(f"Expected p churn per mode: {btyd_dataset.expected_p_churn_from_priors()}")
    # print(f"Expected time interval per mode: {btyd_dataset.expected_time_interval_from_priors()}")
    # print(f"Truncated sequences: {btyd_dataset.truncated_sequences}")

    # btyd_dataset.plot_tracked_statistics().show()

    discrete_values = dataset.get_discrete_feature_values(
        start_token=START_TOKEN_DISCR,
    )

    encoder = FeatureDictionaryEncoder(eventsprofiles_datamodel.cont_feat, discrete_values)

    inference = nl.ModelInference(model_filename=checkpoint_path, encoder=encoder)

    tokenizer = Tokenizer(
        continuous_features=eventsprofiles_datamodel.cont_feat,
        discrete_features=discrete_values,
        start_token_continuous=np.nan,
        start_token_discrete=START_TOKEN_DISCR,
        start_token_other=np.nan,
        max_item_len=100,
    )

    target_transform = TargetCreator(cols=COLS)

    data_loader = SequenceLoader(
        dataset,
        encoder,
        target_transform,
        tokenizer,
        100,
        1,
    )

    # datamodule = SequenceDataModule(
    #     dataset=dataset,
    #     tokenizer=tokenizer,
    #     transform=encoder,
    #     target_transform=target_transform,
    #     test_size=0.2,
    #     batch_points=1024,
    #     min_points=1,
    # )

    raw_data, extended_sequences = inference.extend_sequence(
        data_loader,
        start_date=END_DATE,
        end_date=END_DATE + datetime.timedelta(days=90),
        return_input=True,
    )

    print("@")
