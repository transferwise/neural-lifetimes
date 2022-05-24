import datetime
import random
from typing import Any, List, Dict, Optional, Union
from uuid import uuid4

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from neural_lifetimes.data.dataloaders.sequence_loader import SequenceLoader
from neural_lifetimes.models.modules.classic_model import ClassicModel
from neural_lifetimes.utils.data.feature_encoder import FeatureDictionaryEncoder
from neural_lifetimes.utils import datetime2float, float2datetime

class ModelInference:
    """
    Simulate sequences from scratch or extend sequences from a model artifact.

    A sequence is simulated/extended iteratively by adding one event at the end of the
    sequence each time. To simulate an event, the current sequence is used as the model
    input and the distributions outputted by the model are used to sample the next
    event. The sampled event is added to the sequence and the resulting sequence is used
    as an input in the following iteration. The process ends if a sequence reaches the
    `end_date` or if the customer churns.

    Args:
        model_filename (str): Filepath of a trained model artifact.
            It should be a ``.ckpt`` file.

    Attributes:
        model (ClassicModel): Model instance.
    """

    def __init__(
        self,
        model: Optional[pl.LightningModule] = None,
        model_filename: Optional[str] = None,
        encoder: Optional[FeatureDictionaryEncoder] = None,
        # device: Optional[torch.device] = None
    ):
        if model is None:
            assert model_filename is not None and encoder is not None
            self.model = ClassicModel.load_from_checkpoint(model_filename, feature_encoder_config=encoder.config_dict())
        else:
            assert model_filename is None and encoder is None
            self.model = model
        # self.device = self.model.device if device is None else device

    def predict(self, loader: SequenceLoader, n_samples=1, return_input=True):
        """Samples the following event for every sequence in a SequenceLoader from the output distributions
        of the model.

        Args:
            loader: Sequence Loader
            n_samples: number of samples taken from the output distributions
            return_input: if True the function returns both the input and output if False just the input

        Returns: for every sequence of events in the loader it returns `n_samples` samples for the distributions
            outputted by the model
        """
        predictions = []
        raw_data = []
        for batch in loader:
            out = [self.model(batch) for i in range(n_samples)]
            out_stack = {k: torch.stack([out[i][k] for i in range(n_samples)]) for k in out[0].keys()}

            raw_data.append(batch)
            predictions.append(out_stack)

        if return_input:
            return raw_data, predictions
        else:
            return predictions

    def get_start_token(self, start_token_discr, start_token_cont):

        st = np.array([start_token_discr])
        st_discr = {
            k: torch.tensor(self.model.net.encoder.emb.enc[k].transform(st).reshape(-1))
            for k in self.model.encoder.emb.enc.keys()
        }
        st_cont = {k: torch.tensor([float(start_token_cont)]) for k in self.model.encoder.emb.continuous_features}

        token_event = {**st_discr, **st_cont}
        token_event["dt"] = torch.tensor([float(start_token_cont)])
        # token_event['offsets'] = np.array([0,1])

        return token_event

    def _sample_ouput(self, batch, n: int = 1):

        model_out = self.model(batch)
        sampling = {
            k: self.model.head_map[k].distribution(v.detach()).sample((n,)).flatten()
            for k, v in model_out.items()
            if k in self.model.head_map.keys()
        }
        data = {k[5:]: v for k, v in sampling.items() if k[:5] == "next_"}
        data["churn"] = sampling["p_churn"]

        return data

    def _simulate_initial_events(
        self,
        n: int,
        start_date: datetime.datetime,
        end_date: datetime.datetime,
        start_token_discr: str,
        start_token_cont: int,
    ):

        token_event = self.get_start_token(start_token_discr, start_token_cont)
        token_event["churn"] = torch.tensor([0])

        initial_events = {k: v.repeat(n) for k, v in token_event.items()}

        delta = end_date - start_date
        int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
        initial_events["t"] = np.array(
            [start_date + datetime.timedelta(seconds=random.randrange(int_delta)) for _ in range(n)]
        )
        initial_events["offsets"] = np.arange(n + 1)

        initial_events["token_event"] = torch.ones(n)
        initial_events["USER_PROFILE_ID"] = np.array([str(uuid4()) for i in range(n)])

        return initial_events

    def _sequence_simulation(self, batch: dict, end_date: datetime.datetime):

        # Set pre_encoded to True temporarily in case it wasn't, restore value at the end
        # pre_encoded = self.model.encoder.emb.pre_encoded
        # self.model.encoder.emb.pre_encoded = True

        seqs_ongoing = batch.copy()
        seqs_finished = []

        while len(seqs_ongoing["offsets"]) > 1:

            data = self._sample_ouput(seqs_ongoing)
            # model_out = self.model(seqs_ongoing)
            # sampling = {k: self.model.head_map[k].distribution(v).sample((1,)).flatten()
            #             for k, v in modelout.items() if k in self.model.head_map.keys()}
            # data = {k[5:]: v for k, v in sampling.items() if k[:5] == 'next_'}
            # data['churn'] = sampling['p_churn']

            sequences = split_batch(seqs_ongoing)
            offsets = seqs_ongoing["offsets"]
            num_seqs = len(sequences)

            new_sequences = []
            churn_state = []
            last_date = []
            for i, seq in enumerate(sequences):
                next_t = torch.tensor([seq["t"][-1] + data["dt"][offsets[i + 1] - 1]], device=seq["t"].device)
                new_seq = {}

                if next_t < datetime2float(end_date):
                    new_seq["t"] = torch.cat((seq["t"], next_t))

                    for k in seq.keys():
                        if k in data.keys():
                            new_seq[k] = torch.cat((seq[k], data[k][offsets[i + 1] - 1 : offsets[i + 1]]))

                    new_seq["USER_PROFILE_ID"] = torch.tensor(
                        [seq["USER_PROFILE_ID"][0]] * len(new_seq["dt"]), device=new_seq["dt"].device
                    )
                    new_seq["token_event"] = torch.cat(
                        (seq["token_event"], torch.tensor([0], device=seq["token_event"].device))
                    )
                    new_seq["offsets"] = torch.tensor([0, len(new_seq["t"])])
                else:
                    new_seq = seq.copy()

                new_sequences.append(new_seq)
                churn_state.append(new_seq["churn"][-1])
                last_date.append(next_t)

            seqs_ongoing = [
                new_sequences[i]
                for i in range(num_seqs)
                if churn_state[i] == 0 and last_date[i] < datetime2float(end_date)
            ]
            seqs_finished.extend(
                [new_sequences[i] for i in range(num_seqs) if churn_state[i] == 1 or last_date[i] >= datetime2float(end_date)]
            )

            seqs_ongoing = build_batch(seqs_ongoing)

        # self.model.encoder.emb.pre_encoded = pre_encoded

        out = build_batch(seqs_finished)

        return out

    def _infer_churn_status(
        self,
        batch: dict,
        sim_start_date: datetime.datetime,
        n: int = 1,
    ):

        data = self._sample_ouput(batch, n)
        sim_start_date = datetime2float(sim_start_date)
        sequences = split_batch(batch)
        num_batch_seqs = len(sequences)
        num_repeated_seqs = num_batch_seqs * n
        offsets = torch.cat(
            [torch.tensor([0], device=batch["offsets"].device)]
            + [batch["offsets"][1:] + i * batch["offsets"][-1] for i in range(n)]
        )

        original_sequences = []
        appended_sequences = []

        churn_last_event = []
        churn_next_event = []
        last_date = []

        for i in range(num_repeated_seqs):
            seq = sequences[i % num_batch_seqs]
            seq_next = append_event_to_seq(seq, data, offsets[i + 1])
            seq_original = {k: seq[k] for k in seq_next.keys()}

            original_sequences.append(seq_original)
            appended_sequences.append(seq_next)

            churn_last_event.append(seq_next["churn"][-2])
            churn_next_event.append(seq_next["churn"][-1])
            last_date.append(seq_next["t"][-1])

        seqs_ongoing = [
            appended_sequences[i]
            for i in range(num_repeated_seqs)
            if churn_last_event[i] == 0 and churn_next_event[i] == 0 and last_date[i] > sim_start_date
        ]
        seqs_finished = [
            original_sequences[i]
            if churn_last_event[i] == 0 or last_date[i] < sim_start_date
            else appended_sequences[i]
            for i in range(num_repeated_seqs)
            if churn_last_event[i] == 1 or churn_next_event[i] == 1 or last_date[i] < sim_start_date
        ]

        print(
            f"Seqs ongoing: {len(seqs_finished)}. Seqs finished: {len(seqs_finished)}."
            + f"Mean churn_last_real_event {torch.stack(churn_last_event).mean():.2f}."
            + f"Mean churn_next_event {torch.stack(churn_next_event).mean():.2f}."
            + f"Sampled date before Start simulation {torch.stack([t < sim_start_date for t in last_date]).to(torch.float32).mean()}"
        )
        seqs_ongoing_batch = build_batch(seqs_ongoing)
        seqs_finished_batch = build_batch(seqs_finished)

        return seqs_ongoing_batch, seqs_finished_batch

    def simulate_sequences(
        self,
        n: int,
        start_date: datetime.datetime,
        start_date_limit: datetime.datetime,
        end_date: datetime.datetime,
        start_token_discr: str,
        start_token_cont: int,
    ):
        """
        Simulate sequences from scratch.

        The `n` sequences start with an event randomly sampled between
        `start_date` and  `start_date_limit`. The sequences of events are build by sampling
        from the model distribution ouputs. The sequence is initialized with a Starting Token event.
        A sequence will end when if either the user churns or if an event happens after the
        `end_date`.

        Args:
            n (int): Number of sequences to simulate.
            start_date (datetime.datetime): Earliest start date of a sequence.
            start_date_limit (datetime.datetime): Latest start date of a sequence.
            end_date (datetime.datetime): Latest end date of a sequence.
            start_token_discr (str): Starting token event for discrete variables.
            start_token_cont (int): Starting token event for continuous variables.

        Returns:
            dict: Dictionary containing the simulated sequences, where keys are those of the sequences plus
                ``"offset"``.

        """
        initial_events = self._simulate_initial_events(
            n, start_date, start_date_limit, start_token_discr, start_token_cont
        )

        sequences = self._sequence_simulation(initial_events, end_date)

        return sequences

    def extend_sequence(
        self,
        loader: SequenceLoader,
        start_date: datetime.datetime = datetime.datetime(2021, 1, 1, 0, 0, 0),
        end_date: datetime.datetime = datetime.datetime(2021, 4, 1, 0, 0, 0),
        n: int = 1,
        return_input: bool = True,
    ):
        """
        Extends sequences from already exisiting sequences using the given model.

        Given a SequenceLoader the function extends the event sequences between two dates: `start_date` and
        `end_date`. The sequences are extended in a stochastic way by sampling from the distributions outputted
        by the model.
        A sequence will end when if either the user churns or if an event happens after the `end_date`.

        Args:
            loader (SequenceLoader): SequenceLoader that contains the sequences to extend.
            start_date (datetime.datetime): Earliest date from the first event appended to the sequences
            end_date (datetime.datetime): The sequences will be extended
            n (int): number of samples per sequence
            return_input (bool): bool describing if the input sequences should also be returned.

        Returns:
            dict: Dictionary containing the simulated sequences, where keys are those of the sequences plus
                ``"offset"``.

        """
        # continuous_features = self.model.emb.continuous_features
        # discrete_features = self.model.emb.discrete_features.keys()

        extended_sequences = []
        raw_data = []
        for i, batch in enumerate(loader):
            encoded_batch = to_device(batch, self.model.device)
            # encoded_batch["t"] = tensor2npdatetime(batch["t"])

            token_event = torch.zeros(len(batch["t"]), device=self.model.device)
            token_event[batch["offsets"][:-1]] = 1
            encoded_batch["token_event"] = token_event

            encoded_batch["churn"] = torch.zeros(len(batch["t"]), device=self.model.device)

            batch_ongoing, batch_churned = self._infer_churn_status(encoded_batch, start_date, n)
            extend_seq = self._sequence_simulation(batch_ongoing, end_date)

            if return_input:
                raw_data.append(batch)

            all_sequences = merge_batches([extend_seq, batch_churned])
            extended_sequences.append(all_sequences)

        if return_input:
            return raw_data, extended_sequences
        else:
            return extended_sequences

    def _simulate_initial_events(
        self,
        n: int,
        start_date: datetime.datetime,
        end_date: datetime.datetime,
        start_token_discr: str,
        start_token_cont: int,
    ):

        token_event = self.get_start_token(start_token_discr, start_token_cont)
        token_event["churn"] = torch.tensor([0])

        initial_events = {k: v.repeat(n) for k, v in token_event.items()}

        delta = end_date - start_date
        int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
        initial_events["t"] = np.array(
            [start_date + datetime.timedelta(seconds=random.randrange(int_delta)) for _ in range(n)]
        )
        initial_events["offsets"] = np.arange(n + 1)

        initial_events["token_event"] = torch.ones(n)
        initial_events["USER_PROFILE_ID"] = np.array([str(uuid4()) for i in range(n)])

        return initial_events


def split_batch(batch: Dict):
    keys = [k for k in batch.keys() if k[:5] != "next_"]
    out = [{k: batch[k][s:e] for k in keys} for s, e in zip(batch["offsets"][:-1], batch["offsets"][1:])]
    return out


def build_batch(seqs: List):

    if len(seqs) == 0:
        return {"offsets": torch.tensor([0])}

    seq_concat = {}
    dev = seqs[0]["t"].device
    assert dev == torch.device("cuda:0")  # TODO remove this!!
    for k in seqs[0].keys():
        if isinstance(seqs[0][k], torch.Tensor):
            seq_concat[k] = torch.cat([s[k].to(dev) for s in seqs])
        else:
            seq_concat[k] = np.concatenate([s[k] for s in seqs])

    seq_lengths = np.array([0] + [len(s["dt"]) for s in seqs])
    seq_concat["offsets"] = torch.from_numpy(np.cumsum(seq_lengths))

    return seq_concat


def merge_batches(batches: List, keys: List = None):

    if len(batches) == 0:
        return {"offsets": torch.tensor([0])}

    real_batches = [batch for batch in batches if batch["offsets"].shape[0] > 1]

    if keys is None:
        keys = real_batches[0].keys()

    merged = {
        k: torch.cat([batch[k] for batch in real_batches])
        if isinstance(real_batches[0][k], torch.Tensor)
        else np.concatenate([batch[k] for batch in real_batches])
        for k in keys
    }

    off_cumsum = np.cumsum([0] + [batch["offsets"][-1] for batch in real_batches])
    merged["offsets"] = torch.cat(
        [torch.tensor([0])] + [batch["offsets"][1:] + off_cumsum[i] for i, batch in enumerate(real_batches)]
    )

    return merged

def add_delta_to_numpy(t: np.ndarray, delta_hours: float) -> np.ndarray:
    return t + np.array(datetime.timedelta(hours=delta_hours)).astype(np.timedelta64)


def append_event_to_seq(seq: dict, data: dict, i: int):
    new_seq = {}

    next_t = torch.tensor([seq["t"][-1] + data["dt"][i - 1]], device=seq["t"].device)
    new_seq["t"] = torch.cat([seq["t"], next_t])

    for k in seq.keys():
        if k in data.keys():
            if k == "churn":
                new_seq[k] = torch.cat((seq[k][:-1].to(data[k].device), data[k][i - 2 : i]))
            else:
                new_seq[k] = torch.cat((seq[k], data[k][i - 1].unsqueeze(0)))

    new_seq["USER_PROFILE_ID"] = torch.tensor([seq["USER_PROFILE_ID"][-1]] * len(new_seq["dt"]))
    try:
        new_seq["token_event"] = torch.cat((seq["token_event"], torch.tensor([0], device=seq["token_event"].device)))
    except KeyError:
        pass
    return new_seq


def to_device(batch, device):
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


# class InferenceEngine:
#     def __init__(self, model: nn.Module) -> None:
#         super().__init__()
#         self.model = model

#     def __call__(
#         self, batch: Dict[str, torch.Tensor], start_date: datetime.datetime, end_date: datetime.datetime, n: int
#     ) -> Any:
#         # generate forecasts
#         batch_ongoing, batch_churned = self._infer_churn_status(batch, start_date, n, True)
#         extend_seq = self._sequence_simulation(batch_ongoing, end_date)

#     def _infer_churn_status(
#         self,
#         batch: Dict[str, torch.Tensor],
#         sim_start_date: datetime.datetime,
#         n: int = 1,
#         discrete_are_logits: bool = False,
#     ):

#         data = self._sample_ouput(batch, n, discrete_are_logits)

#         sequences = split_batch(batch)
#         num_batch_seqs = len(sequences)
#         num_repeated_seqs = num_batch_seqs * n
#         offsets = torch.cat([torch.tensor([0])] + [batch["offsets"][1:] + i * batch["offsets"][-1] for i in range(n)])

#         original_sequences = []
#         appended_sequences = []

#         churn_last_event = []
#         churn_next_event = []
#         last_date = []

#         for i in range(num_repeated_seqs):
#             seq = sequences[i % num_batch_seqs]
#             seq_next = append_event_to_seq(seq, data, offsets[i + 1])
#             seq_original = {k: seq[k] for k in seq_next.keys()}

#             original_sequences.append(seq_original)
#             appended_sequences.append(seq_next)

#             churn_last_event.append(seq_next["churn"][-2])
#             churn_next_event.append(seq_next["churn"][-1])
#             last_date.append(seq_next["t"][-1])

#         seqs_ongoing = [
#             appended_sequences[i]
#             for i in range(num_repeated_seqs)
#             if churn_last_event[i] == 0 and churn_next_event[i] == 0 and last_date[i] > sim_start_date
#         ]
#         seqs_finished = [
#             original_sequences[i]
#             if churn_last_event[i] == 0 or last_date[i] < sim_start_date
#             else appended_sequences[i]
#             for i in range(num_repeated_seqs)
#             if churn_last_event[i] == 1 or churn_next_event[i] == 1 or last_date[i] < sim_start_date
#         ]

#         print(
#             f"Seqs ongoing: {len(seqs_finished)}. Seqs finished: {len(seqs_finished)}. Mean churn_last_real_event {round(np.mean(churn_last_event),2)}. Mean churn_next_event {round(np.mean(churn_next_event),2)}. Sampled date before Start simulation {np.mean([t < sim_start_date for t in last_date])}"
#         )
#         seqs_ongoing_batch = build_batch(seqs_ongoing)
#         seqs_finished_batch = build_batch(seqs_finished)

#         return seqs_ongoing_batch, seqs_finished_batch

#     def _sequence_simulation(self, batch: dict, end_date: datetime.datetime):

#         # Set pre_encoded to True temporarily in case it wasn't, restore value at the end
#         # pre_encoded = self.model.encoder.emb.pre_encoded
#         # self.model.encoder.emb.pre_encoded = True

#         seqs_ongoing = batch.copy()
#         seqs_finished = []

#         while len(seqs_ongoing["offsets"]) > 1:

#             data = self._sample_ouput(seqs_ongoing)
#             # model_out = self.model(seqs_ongoing)
#             # sampling = {k: self.model.head_map[k].distribution(v).sample((1,)).flatten()
#             #             for k, v in modelout.items() if k in self.model.head_map.keys()}
#             # data = {k[5:]: v for k, v in sampling.items() if k[:5] == 'next_'}
#             # data['churn'] = sampling['p_churn']

#             sequences = split_batch(seqs_ongoing)
#             offsets = seqs_ongoing["offsets"]
#             num_seqs = len(sequences)

#             new_sequences = []
#             churn_state = []
#             last_date = []
#             for i, seq in enumerate(sequences):
#                 next_t = add_delta_to_numpy(seq["t"][-1], data["dt"][offsets[i + 1] - 1].item())
#                 new_seq = {}

#                 if next_t < end_date:
#                     new_seq["t"] = np.append(seq["t"], next_t)

#                     for k in seq.keys():
#                         if k in data.keys():
#                             new_seq[k] = torch.cat((seq[k], data[k][offsets[i + 1] - 1 : offsets[i + 1]]))

#                     new_seq["USER_PROFILE_ID"] = np.array([seq["USER_PROFILE_ID"][0]] * len(new_seq["dt"]))
#                     new_seq["token_event"] = torch.cat((seq["token_event"], torch.tensor([0])))

#                 new_sequences.append(new_seq)
#                 churn_state.append(new_seq["churn"][-1])
#                 last_date.append(next_t)

#             seqs_ongoing = [
#                 new_sequences[i] for i in range(num_seqs) if churn_state[i] == 0 and last_date[i] < end_date
#             ]
#             seqs_finished.extend(
#                 [new_sequences[i] for i in range(num_seqs) if churn_state[i] == 1]  # or last_date[i] >= end_date
#             )

#             seqs_ongoing = build_batch(seqs_ongoing)
#         out = build_batch(seqs_finished)

#         return out

#     def _sample_ouput(
#         self, batch: Dict[str, torch.Tensor], n: int = 1, discrete_are_logits: bool = False
#     ) -> Dict[str, torch.Tensor]:

#         # assumes that batch is fully encoded. including t and dt as float values

#         model_out = self.model(batch)
#         if discrete_are_logits:
#             for k, v in model_out.items():
#                 assert torch.all(torch.isfinite(v))
#                 if k[5:] in self.model.emb.discrete_features:
#                     model_out[k] = F.softmax(v.to(torch.float32), dim=-1)  # Softmax does not reliably sum to 1 for fp16
#                 else:
#                     model_out[k] = v
#         sampling = {
#             k: self.model.head_map[k].distribution(v.detach()).sample((n,)).flatten()
#             for k, v in model_out.items()
#             if k in self.model.head_map.keys()
#         }
#         data = {k[5:]: v for k, v in sampling.items() if k[:5] == "next_"}
#         data["churn"] = sampling["p_churn"]

#         return data
