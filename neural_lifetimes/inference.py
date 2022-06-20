from typing import Dict, Any
import datetime
import random
from typing import List
from uuid import uuid4

import numpy as np
import torch

from neural_lifetimes.data.dataloaders.sequence_loader import SequenceLoader
from neural_lifetimes.models.modules.classic_model import ClassicModel


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

    def __init__(self, model_filename: str, init_kwargs: Dict[str, Any] = None):
        if init_kwargs is None:
            init_kwargs = {}
        self.model = ClassicModel.load_from_checkpoint(model_filename, **init_kwargs)
        print(f"Model loaded from: {model_filename} with args: {init_kwargs}")

    def predict(self, loader: SequenceLoader, n_samples=1, return_input=True):
        """
        Samples the following event for every sequence in a SequenceLoader from the output distributions
        of the model

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

    def _sample_ouput(self, batch):

        model_out = self.model(batch)
        sampling = {
            k: self.model.head_map[k].distribution(v.detach()).sample((1,)).flatten()
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
        pre_encoded = self.model.encoder.emb.pre_encoded
        self.model.encoder.emb.pre_encoded = True

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
                next_t = seq["t"][-1] + datetime.timedelta(hours=data["dt"][offsets[i + 1] - 1].item())
                if next_t < end_date:
                    seq["t"] = np.append(seq["t"], next_t)

                    for k in seq.keys():
                        if k in data.keys():
                            seq[k] = torch.cat((seq[k], data[k][offsets[i + 1] - 1 : offsets[i + 1]]))

                    seq["USER_PROFILE_ID"] = np.array([seq["USER_PROFILE_ID"][0]] * len(seq["dt"]))
                    seq["token_event"] = torch.cat((seq["token_event"], torch.tensor([0])))

                new_sequences.append(seq.copy())
                churn_state.append(seq["churn"][-1])
                last_date.append(next_t)

            seqs_ongoing = [
                new_sequences[i] for i in range(num_seqs) if churn_state[i] == 0 and last_date[i] < end_date
            ]
            seqs_finished.extend(
                [new_sequences[i] for i in range(num_seqs) if churn_state[i] == 1]  # or last_date[i] >= end_date
            )

            seqs_ongoing = build_batch(seqs_ongoing)

        self.model.encoder.emb.pre_encoded = pre_encoded

        out = build_batch(seqs_finished)

        return out

    def _infer_churn_status(
        self,
        batch: dict,
        sim_start_date: datetime.datetime,
    ):
        # Set pre_encoded to True temporarily in case it wasn't, restore value at the end
        pre_encoded = self.model.encoder.emb.pre_encoded
        self.model.encoder.emb.pre_encoded = True

        data = self._sample_ouput(batch)

        sequences = split_batch(batch)
        offsets = batch["offsets"]
        num_seqs = len(sequences)

        new_sequences = []
        churn_state = []
        last_date = []

        for i, seq in enumerate(sequences):
            next_t = seq["t"][-1] + datetime.timedelta(hours=data["dt"][offsets[i + 1] - 1].item())
            seq["t"] = np.append(seq["t"], next_t)

            for k in seq.keys():
                if k in data.keys():
                    seq[k] = torch.cat((seq[k], data[k][offsets[i + 1] - 1 : offsets[i + 1]]))

            seq["USER_PROFILE_ID"] = np.array([seq["USER_PROFILE_ID"][-1]] * len(seq["dt"]))
            seq["token_event"] = torch.cat((seq["token_event"], torch.tensor([0])))

            new_sequences.append(seq.copy())
            churn_state.append(seq["churn"][-1])
            last_date.append(seq["t"][-1])

        self.model.encoder.emb.pre_encoded = pre_encoded

        seqs_ongoing = [
            new_sequences[i] for i in range(num_seqs) if churn_state[i] == 0 and last_date[i] > sim_start_date
        ]
        seqs_finished = [
            new_sequences[i] for i in range(num_seqs) if churn_state[i] == 1 or last_date[i] < sim_start_date
        ]

        seqs_ongoing = build_batch(seqs_ongoing)
        seqs_finished = build_batch(seqs_finished)

        return seqs_ongoing, seqs_finished

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
            return_input (bool): bool describing if the input sequences should also be returned.

        Returns:
            dict: Dictionary containing the simulated sequences, where keys are those of the sequences plus
                ``"offset"``.

        """
        continuous_features = self.model.encoder.emb.continuous_features
        discrete_features = self.model.encoder.emb.enc.keys()

        extended_sequences = []
        raw_data = []
        for i, batch in enumerate(loader):
            encode_cont = {f: batch[f] for f in continuous_features}
            for f in encode_cont.keys():
                encode_cont[f][encode_cont[f].isnan()] = 0

            encode_discr = {name: self.model.encoder.emb.encode(name, batch[name]) for name in discrete_features}
            encode_batch = {**encode_cont, **encode_discr}

            encode_batch["USER_PROFILE_ID"] = batch["USER_PROFILE_ID"]
            encode_batch["offsets"] = batch["offsets"]
            encode_batch["dt"] = batch["dt"]

            encode_batch["t"] = tensor2datetime(batch["t"])

            token_event = torch.zeros(len(batch["t"]))
            token_event[batch["offsets"][:-1]] = 1
            encode_batch["token_event"] = token_event

            encode_batch["churn"] = torch.zeros(len(batch["t"]))

            batch_ongoing, batch_churned = self._infer_churn_status(encode_batch, start_date)
            extend_seq = self._sequence_simulation(batch_ongoing, end_date)

            raw_data.append(batch)

            extended_sequences.append({"extended_sequences": extend_seq, "inferred_churn": batch_churned})

        if return_input:
            return raw_data, extended_sequences
        else:
            return extended_sequences


def split_batch(batch: dict):
    keys = batch.keys()
    out = [{k: batch[k][s:e] for k in keys} for s, e in zip(batch["offsets"][:-1], batch["offsets"][1:])]
    return out


def build_batch(seqs: List):

    if len(seqs) == 0:
        return {"offsets": np.array([0])}

    seq_concat = {}
    for k in seqs[0].keys():
        if isinstance(seqs[0][k], torch.Tensor):
            seq_concat[k] = torch.cat([s[k] for s in seqs])
        else:
            seq_concat[k] = np.concatenate([s[k] for s in seqs])

    seq_lengths = np.array([0] + [len(s["dt"]) for s in seqs])
    seq_concat["offsets"] = np.cumsum(seq_lengths)

    return seq_concat


def datetime2tensor(x: np.ndarray):
    ts = x.astype("datetime64[us]").astype(np.int64).astype(np.float32) / (1e6 * 60 * 60)
    return ts


def tensor2datetime(x: torch.tensor):
    ts = [t.item() for t in (x.astype("float32") * (1e6 * 60 * 60)).astype("datetime64[us]")]
    return ts
