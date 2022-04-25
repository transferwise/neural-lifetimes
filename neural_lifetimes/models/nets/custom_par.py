"""Probabilistic autoregressive model."""

import copy
import logging
import pickle
import uuid

import numpy as np
import pandas as pd
import rdt
import torch
from deepecho.models.base import DeepEcho
from deepecho.sequences import assemble_sequences
from sdv.metadata import Table
from sdv.tabular.copulas import GaussianCopula
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)


def batch_to_df(sequences, columns, entity):
    df = pd.DataFrame()
    for k, v in sequences.items():
        if isinstance(v, torch.Tensor):
            list_ = list(v.numpy())
        else:
            list_ = list(v)
        if len(list_) < len(sequences["t"]):  # removes all the next_* and *_to_now features
            continue
        else:
            df[k] = list_

    df = df[columns + entity]

    return df


class PARNet(torch.nn.Module):
    """PARModel ANN model."""

    def __init__(self, data_size, context_size, hidden_size=32):
        super(PARNet, self).__init__()
        self.context_size = context_size
        self.down = torch.nn.Linear(data_size + context_size, hidden_size)
        self.rnn = torch.nn.GRU(hidden_size, hidden_size)
        self.up = torch.nn.Linear(hidden_size, data_size)

    def forward(self, x, c):
        """Forward passing computation."""
        if isinstance(x, torch.nn.utils.rnn.PackedSequence):
            x, lengths = torch.nn.utils.rnn.pad_packed_sequence(x)
            if self.context_size:
                x = torch.cat(
                    [x, c.unsqueeze(0).expand(x.shape[0], c.shape[0], c.shape[1])],
                    dim=2,
                )

            x = self.down(x)
            x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, enforce_sorted=False)
            x, _ = self.rnn(x)
            x, lengths = torch.nn.utils.rnn.pad_packed_sequence(x)
            x = self.up(x)
            x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, enforce_sorted=False)

        else:
            if self.context_size:
                x = torch.cat(
                    [x, c.unsqueeze(0).expand(x.shape[0], c.shape[0], c.shape[1])],
                    dim=2,
                )

            x = self.down(x)
            x, _ = self.rnn(x)
            x = self.up(x)

        return x


class PARModel(DeepEcho):
    def __init__(self, epochs=128, sample_size=1, cuda=True, verbose=True):
        self.epochs = epochs
        self.sample_size = sample_size

        if not cuda or not torch.cuda.is_available():
            device = "cpu"
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = "cuda"

        self.device = torch.device(device)
        self.verbose = verbose

        LOGGER.info("%s instance created", self)
        if verbose:
            print(self, "instance created")

    def __repr__(self):
        return "{}(epochs={}, sample_size={}, cuda='{}', verbose={})".format(
            self.__class__.__name__,
            self.epochs,
            self.sample_size,
            self.device,
            self.verbose,
        )

    def _idx_map(self, x, t):
        idx = 0
        idx_map = {}
        for i, t in enumerate(t):
            if t == "continuous" or t == "datetime":
                try:
                    if i == 14:
                        x[i] = [hash(k) for k in x[i]]
                    idx_map[i] = {
                        "type": t,
                        "mu": np.nanmean(x[i]),
                        "std": np.nanstd(x[i]),
                        "nulls": pd.isnull(x[i]).any(),
                        "indices": (idx, idx + 1, idx + 2),
                    }
                    idx += 3
                except Exception as e:
                    print(e)

            elif t == "count":
                idx_map[i] = {
                    "type": t,
                    "min": np.nanmin(x[i]),
                    "range": np.nanmax(x[i]) - np.nanmin(x[i]),
                    "nulls": pd.isnull(x[i]).any(),
                    "indices": (idx, idx + 1, idx + 2),
                }
                idx += 3

            elif t == "categorical" or t == "ordinal":
                idx_map[i] = {"type": t, "indices": {}}
                idx += 1
                for v in set(x[i]):
                    if pd.isnull(v):
                        v = None

                    idx_map[i]["indices"][str(v)] = idx
                    idx += 1

            else:
                raise ValueError("Unsupported type: {}".format(t))

        return idx_map, idx

    def _build(self, examples, types):

        # TODO: define better suited values
        min_length = 0
        max_length = int(1e3)

        self._fixed_length = min_length == max_length
        self._min_length = min_length
        self._max_length = max_length

        self._ctx_map, self._ctx_dims = self._idx_map(
            examples["context"], types["context"]
        )  # all possible values => per batch!
        self._data_map, self._data_dims = self._idx_map(examples["data"], types["data"])
        self._data_map["<TOKEN>"] = {
            "type": "categorical",
            "indices": {
                "<START>": self._data_dims,
                "<END>": self._data_dims + 1,
                "<BODY>": self._data_dims + 2,
            },
        }
        self._data_dims += 3

    def _data_to_tensor(self, data):
        seq_len = len(data[0]) if data != [] else 0
        X = []

        x = torch.zeros(self._data_dims)
        x[self._data_map["<TOKEN>"]["indices"]["<START>"]] = 1.0
        X.append(x)

        for i in range(seq_len):
            x = torch.zeros(self._data_dims)
            for key, props in self._data_map.items():
                if key == "<TOKEN>":
                    x[self._data_map["<TOKEN>"]["indices"]["<BODY>"]] = 1.0

                elif props["type"] in ["continuous", "timestamp"]:
                    mu_idx, sigma_idx, missing_idx = props["indices"]
                    if data[key][i].isalpha() and not (props["std"] == 0):
                        x[mu_idx] = (
                            (float(data[key][i]) - props["mu"]) / props["std"]
                            if isinstance(data[key][i], float)
                            else 0.0
                        )
                    else:
                        x[mu_idx] = 0.0

                    x[sigma_idx] = 0.0
                    x[missing_idx] = 1.0 if data[key][i] is None else 0.0

                elif props["type"] in ["count"]:
                    r_idx, p_idx, missing_idx = props["indices"]
                    x[r_idx] = (
                        0.0
                        if (data[key][i] is None or props["range"] == 0)
                        else (data[key][i] - props["min"]) / props["range"]
                    )
                    x[p_idx] = 0.0
                    x[missing_idx] = 1.0 if data[key][i] is None else 0.0

                elif props["type"] in ["categorical", "ordinal"]:  # categorical
                    value = data[key][i]
                    if pd.isnull(value):
                        value = None
                    x[props["indices"][value]] = 1.0

                else:
                    raise ValueError()

            X.append(x)

        x = torch.zeros(self._data_dims)
        x[self._data_map["<TOKEN>"]["indices"]["<END>"]] = 1.0
        X.append(x)

        return torch.stack(X, dim=0).to(self.device)

    def _context_to_tensor(self, context):
        if not self._ctx_dims:
            return None

        x = torch.zeros(self._ctx_dims)
        for key, props in self._ctx_map.items():
            if props["type"] in ["continuous", "datetime"]:
                mu_idx, sigma_idx, missing_idx = props["indices"]
                x[mu_idx] = (
                    0.0
                    if (pd.isnull(context[key]) or props["std"] == 0)
                    else (context[key] - props["mu"]) / props["std"]
                )
                x[sigma_idx] = 0.0
                x[missing_idx] = 1.0 if pd.isnull(context[key]) else 0.0

            elif props["type"] in ["count"]:
                r_idx, p_idx, missing_idx = props["indices"]
                x[r_idx] = (
                    0.0
                    if (pd.isnull(context[key]) or props["range"] == 0)
                    else (context[key] - props["min"]) / props["range"]
                )
                x[p_idx] = 0.0
                x[missing_idx] = 1.0 if pd.isnull(context[key]) else 0.0

            elif props["type"] in ["categorical", "ordinal"]:
                value = context[key]
                if pd.isnull(value):
                    value = None
                x[props["indices"][value]] = 1.0

            else:
                raise ValueError()

        return x.to(self.device)

    def _transform_sequence_index(self, sequences, dem):
        sequence_index_idx = dem._data_columns.index(dem._sequence_index)
        for sequence in sequences:
            data = sequence["data"]
            sequence_index = data[sequence_index_idx]
            diffs = np.diff(sequence_index).tolist()
            data[sequence_index_idx] = diffs[0:1] + diffs
            data.append(sequence_index[0:1] * len(sequence_index))

    def fit_sequences(self, sequence_loader, columns, types, examples, dem):
        """Fit a model to the specified sequences.

        Args:
            sequences (list):
                List of sequences. Each sequence is a single training example
                (i.e. an example of a multivariate time series with some context).
                For example, a sequence might look something like::

                    {
                        "context": [1],
                        "data": [
                            [1, 3, 4, 5, 11, 3, 4],
                            [2, 2, 3, 4,  5, 1, 2],
                            [1, 3, 4, 5,  2, 3, 1]
                        ]
                    }

                The "context" attribute maps to a list of variables which
                should be used for conditioning. These are variables which
                do not change over time.

                The "data" attribute contains a list of lists corrsponding
                to the actual time series data such that `data[i][j]` contains
                the value at the jth time step of the ith channel of the
                multivariate time series.
            context_types (list):
                List of strings indicating the type of each value in context.
                he value at `context[i]` must match the type specified by
                `context_types[i]`. Valid types include the following: `categorical`,
                `continuous`, `ordinal`, `count`, and `datetime`.
            data_types (list):
                List of strings indicating the type of each channel in data.
                Each value in the list at data[i] must match the type specified by
                `data_types[i]`. The valid types are the same as for `context_types`.
        """
        # build variable2id maps
        self._build(examples, types)

        # Initialize model with data dimension & context dimension
        self._model = PARNet(self._data_dims, self._ctx_dims).to(self.device)

        # Specify optimizer
        optimizer = torch.optim.Adam(self._model.parameters(), lr=1e-3)

        iterator = range(self.epochs)
        if self.verbose:
            iterator = tqdm(iterator)

        # Training
        for epoch in iterator:
            for sequences in sequence_loader:

                # process sequence as in _fit
                timeseries_data = batch_to_df(sequences, columns, dem._entity_columns)
                sequences = assemble_sequences(
                    timeseries_data,
                    dem._entity_columns,
                    dem._context_columns,
                    dem._segment_size,
                    dem._sequence_index,
                    drop_sequence_index=False,
                )
                if dem._sequence_index:
                    self._transform_sequence_index(sequences, dem)

                X, C = [], []
                for sequence in sequences:
                    dataFeed = np.array(sequence["data"]).tolist()
                    # if dataFeed!=[]:
                    X.append(self._data_to_tensor(dataFeed))
                    C.append(self._context_to_tensor(sequence["context"]))

                X = torch.nn.utils.rnn.pack_sequence(X, enforce_sorted=False).to(self.device)
                if self._ctx_dims:
                    C = torch.stack(C, dim=0).to(self.device)

                X_padded, seq_len = torch.nn.utils.rnn.pad_packed_sequence(X)

                Y = self._model(X, C)
                Y_padded, _ = torch.nn.utils.rnn.pad_packed_sequence(Y)

                optimizer.zero_grad()
                loss = self._compute_loss(X_padded[1:, :, :], Y_padded[:-1, :, :], seq_len)
                loss.backward()
                if self.verbose:
                    iterator.set_description("Epoch {} | Loss {}".format(epoch + 1, loss.item()))

                optimizer.step()

    def _compute_loss(self, X_padded, Y_padded, seq_len):
        """Compute the loss between X and Y.

        Given X[i,:,:], the neural network predicts the value at the next
        timestep (i+1); this prediction is provided in Y[i,:,:]. This function
        returns the loss between the predicted and actual sequence.

        .. note::
            The `i`th time series (with padding removed) can be indexed with
            `X[:seq_len[i], i, :]`.

        Args:
            X_padded (tensor):
                This contains the input to the model.
            Y_padded (tensor):
                This contains the output of the model.
            seq_len (list):
                This list contains the length of each sequence.
        """
        log_likelihood = 0.0
        _, batch_size, input_size = X_padded.shape

        for key, props in self._data_map.items():
            if props["type"] in ["continuous", "timestamp"]:
                mu_idx, sigma_idx, missing_idx = props["indices"]
                mu = Y_padded[:, :, mu_idx]
                sigma = torch.nn.functional.softplus(Y_padded[:, :, sigma_idx])
                missing = torch.nn.LogSigmoid()(Y_padded[:, :, missing_idx])

                for i in range(batch_size):
                    dist = torch.distributions.normal.Normal(mu[: seq_len[i], i], sigma[: seq_len[i], i])
                    log_likelihood += torch.sum(dist.log_prob(X_padded[-seq_len[i] :, i, mu_idx]))

                    p_true = X_padded[: seq_len[i], i, missing_idx]
                    p_pred = missing[: seq_len[i], i]
                    log_likelihood += torch.sum(p_true * p_pred)
                    log_likelihood += torch.sum((1.0 - p_true) * torch.log(1.0 - torch.exp(p_pred)))

            elif props["type"] in ["count"]:
                r_idx, p_idx, missing_idx = props["indices"]
                r = torch.nn.functional.softplus(Y_padded[:, :, r_idx]) * props["range"]
                p = torch.sigmoid(Y_padded[:, :, p_idx])
                x = X_padded[:, :, r_idx] * props["range"]
                missing = torch.nn.LogSigmoid()(Y_padded[:, :, missing_idx])

                for i in range(batch_size):
                    dist = torch.distributions.negative_binomial.NegativeBinomial(
                        r[: seq_len[i], i], p[: seq_len[i], i]
                    )
                    log_likelihood += torch.sum(dist.log_prob(x[: seq_len[i], i]))

                    p_true = X_padded[: seq_len[i], i, missing_idx]
                    p_pred = missing[: seq_len[i], i]
                    log_likelihood += torch.sum(p_true * p_pred)
                    log_likelihood += torch.sum((1.0 - p_true) * torch.log(1.0 - torch.exp(p_pred)))

            elif props["type"] in ["categorical", "ordinal"]:
                idx = list(props["indices"].values())
                log_softmax = torch.nn.functional.log_softmax(Y_padded[:, :, idx], dim=2)

                for i in range(batch_size):
                    target = X_padded[: seq_len[i], i, idx]
                    predicted = log_softmax[: seq_len[i], i]
                    target = torch.argmax(target, dim=1).unsqueeze(dim=1)
                    log_likelihood += torch.sum(predicted.gather(dim=1, index=target))

            else:
                raise ValueError()

        return -log_likelihood / (batch_size * len(self._data_map) * batch_size)

    def _tensor_to_data(self, x):
        # Force CPU on x
        x = x.to(torch.device("cpu"))

        seq_len, batch_size, _ = x.shape
        assert batch_size == 1

        data = [None] * (len(self._data_map) - 1)
        for key, props in self._data_map.items():
            if key == "<TOKEN>":
                continue

            data[key] = []
            for i in range(seq_len):
                if props["type"] in ["continuous", "datetime"]:
                    mu_idx, sigma_idx, missing_idx = props["indices"]
                    if (x[i, 0, missing_idx] > 0) and props["nulls"]:
                        data[key].append(None)
                    else:
                        data[key].append(x[i, 0, mu_idx].item() * props["std"] + props["mu"])

                elif props["type"] in ["count"]:
                    r_idx, p_idx, missing_idx = props["indices"]
                    if x[i, 0, missing_idx] > 0 and props["nulls"]:
                        data[key].append(None)
                    else:
                        sample = x[i, 0, r_idx].item() * props["range"] + props["min"]
                        data[key].append(int(sample))

                elif props["type"] in ["categorical", "ordinal"]:
                    ml_value, max_x = None, float("-inf")
                    for value, idx in props["indices"].items():
                        if x[i, 0, idx] > max_x:
                            max_x = x[i, 0, idx]
                            ml_value = value

                    data[key].append(ml_value)

                else:
                    raise ValueError()

        return data

    def _sample_state(self, x):
        log_likelihood = 0.0
        seq_len, batch_size, input_size = x.shape
        assert seq_len == 1 and batch_size == 1

        for key, props in self._data_map.items():
            if props["type"] in ["continuous", "timestamp"]:
                mu_idx, sigma_idx, missing_idx = props["indices"]
                mu = x[0, 0, mu_idx]
                sigma = torch.nn.functional.softplus(x[0, 0, sigma_idx])
                dist = torch.distributions.normal.Normal(mu, sigma)
                x[0, 0, mu_idx] = dist.sample()
                x[0, 0, sigma_idx] = 0.0
                log_likelihood += torch.sum(dist.log_prob(x[0, 0, mu_idx]))

                dist = torch.distributions.Bernoulli(torch.sigmoid(x[0, 0, missing_idx]))
                x[0, 0, missing_idx] = dist.sample()
                x[0, 0, mu_idx] = x[0, 0, mu_idx] * (1.0 - x[0, 0, missing_idx])
                log_likelihood += torch.sum(dist.log_prob(x[0, 0, missing_idx]))

            elif props["type"] in ["count"]:
                r_idx, p_idx, missing_idx = props["indices"]
                r = torch.nn.functional.softplus(x[0, 0, r_idx]) * props["range"]
                p = torch.sigmoid(x[0, 0, p_idx])
                dist = torch.distributions.negative_binomial.NegativeBinomial(r, p)
                x[0, 0, r_idx] = dist.sample()
                x[0, 0, p_idx] = 0.0
                log_likelihood += torch.sum(dist.log_prob(x[0, 0, r_idx]))
                x[0, 0, r_idx] /= props["range"]

                dist = torch.distributions.Bernoulli(torch.sigmoid(x[0, 0, missing_idx]))
                x[0, 0, missing_idx] = dist.sample()
                x[0, 0, r_idx] = x[0, 0, r_idx] * (1.0 - x[0, 0, missing_idx])
                log_likelihood += torch.sum(dist.log_prob(x[0, 0, missing_idx]))

            elif props["type"] in ["categorical", "ordinal"]:
                idx = list(props["indices"].values())
                p = torch.nn.functional.softmax(x[0, 0, idx], dim=0)
                x_new = torch.zeros(p.size()).to(self.device)
                x_new.scatter_(dim=0, index=torch.multinomial(p, 1), value=1)
                x[0, 0, idx] = x_new
                log_likelihood += torch.sum(torch.log(p) * x_new)

            else:
                raise ValueError()

        return x, log_likelihood

    def _sample_sequence(self, context, min_length, max_length):
        log_likelihood = 0.0

        x = torch.zeros(self._data_dims).to(self.device)
        x[self._data_map["<TOKEN>"]["indices"]["<START>"]] = 1.0
        x = x.unsqueeze(0).unsqueeze(0)

        for step in range(max_length):
            next_x, ll = self._sample_state(self._model(x, context)[-1:, :, :])
            x = torch.cat([x, next_x], dim=0)
            log_likelihood += ll
            if next_x[0, 0, self._data_map["<TOKEN>"]["indices"]["<END>"]] > 0.0:
                if min_length <= step + 1 <= max_length:
                    break  # received end token

                next_x[0, 0, self._data_map["<TOKEN>"]["indices"]["<BODY>"]] = 1.0
                next_x[0, 0, self._data_map["<TOKEN>"]["indices"]["<END>"]] = 0.0

        return x[1:, :, :], log_likelihood

    def sample_sequence(self, context, sequence_length=None):

        if sequence_length is not None:
            min_length = max_length = sequence_length
        else:
            min_length = self._min_length
            max_length = self._max_length

        if self._ctx_dims:
            context = self._context_to_tensor(context).unsqueeze(0)
        else:
            context = None

        best_x, best_ll = None, float("-inf")
        for _ in range(self.sample_size):
            with torch.no_grad():
                x, log_likelihood = self._sample_sequence(context, min_length, max_length)

            if log_likelihood > best_ll:
                best_x = x
                best_ll = log_likelihood

        return self._tensor_to_data(best_x)


class BaseTimeseriesModel:
    """Base class for timeseries models.

    Args:
        field_names (list[str]):
            List of names of the fields that need to be modeled
            and included in the generated output data. Any additional
            fields found in the data will be ignored and will not be
            included in the generated output.
            If ``None``, all the fields found in the data are used.
        field_types (dict[str, dict]):
            Dictinary specifying the data types and subtypes
            of the fields that will be modeled. Field types and subtypes
            combinations must be compatible with the SDV Metadata Schema.
        anonymize_fields (dict[str, str]):
            Dict specifying which fields to anonymize and what faker
            category they belong to.
        primary_key (str):
            Name of the field which is the primary key of the table.
        entity_columns (list[str]):
            Names of the columns which identify different time series
            sequences. These will be used to group the data in separated
            training examples.
        context_columns (list[str]):
            The columns in the dataframe which are constant within each
            group/entity. These columns will be provided at sampling time
            (i.e. the samples will be conditioned on the context variables).
        segment_size (int, pd.Timedelta or str):
            If specified, cut each training sequence in several segments of
            the indicated size. The size can either can passed as an integer
            value, which will interpreted as the number of data points to
            put on each segment, or as a pd.Timedelta (or equivalent str
            representation), which will be interpreted as the segment length
            in time. Timedelta segment sizes can only be used with sequence
            indexes of type datetime.
        sequence_index (str):
            Name of the column that acts as the order index of each
            sequence. The sequence index column can be of any type that can
            be sorted, such as integer values or datetimes.
        context_model (str or sdv.tabular.BaseTabularModel):
            Model to use to sample the context rows. It can be passed as a
            a string, which must be one of the following:

            * `gaussian_copula` (default): Use a GaussianCopula model.

            Alternatively, a preconfigured Tabular model instance can be
            passed.

        table_metadata (dict or metadata.Table):
            Table metadata instance or dict representation.
            If given alongside any other metadata-related arguments, an
            exception will be raised.
            If not given at all, it will be built using the other
            arguments or learned from the data.
    """

    _DTYPE_TRANSFORMERS = {
        "i": None,
        "f": None,
        "M": rdt.transformers.DatetimeTransformer(strip_constant=True),
        "b": None,
        "O": None,
    }
    _CONTEXT_MODELS = {
        "gaussian_copula": (
            GaussianCopula,
            {"categorical_transformer": "categorical_fuzzy"},
        )
    }

    _metadata = None

    def __init__(
        self,
        field_names=None,
        field_types=None,
        anonymize_fields=None,
        primary_key=None,
        entity_columns=None,
        context_columns=None,
        sequence_index=None,
        segment_size=None,
        context_model=None,
        table_metadata=None,
    ):
        if table_metadata is None:
            self._metadata = Table(
                field_names=field_names,
                primary_key=primary_key,
                field_types=field_types,
                anonymize_fields=anonymize_fields,
                dtype_transformers=self._DTYPE_TRANSFORMERS,
                sequence_index=sequence_index,
                entity_columns=entity_columns,
                context_columns=context_columns,
            )
            self._metadata_fitted = False
        else:
            null_args = (
                field_names,
                primary_key,
                field_types,
                anonymize_fields,
                sequence_index,
                entity_columns,
                context_columns,
            )
            for arg in null_args:
                if arg:
                    raise ValueError("If table_metadata is given {} must be None".format(arg.__name__))

            if isinstance(table_metadata, dict):
                table_metadata = Table.from_dict(
                    table_metadata,
                    dtype_transformers=self._DTYPE_TRANSFORMERS,
                )

            self._metadata = table_metadata
            self._metadata_fitted = table_metadata.fitted

        # Validate arguments
        if segment_size is not None and not isinstance(segment_size, int):
            if sequence_index is None:
                raise TypeError("`segment_size` must be of type `int` if " "no `sequence_index` is given.")

            segment_size = pd.to_timedelta(segment_size)

        self._context_columns = self._metadata._context_columns
        self._entity_columns = self._metadata._entity_columns
        self._sequence_index = self._metadata._sequence_index
        self._segment_size = segment_size

        context_model = context_model or "gaussian_copula"
        if isinstance(context_model, str):
            context_model = self._CONTEXT_MODELS[context_model]

        self._context_model_template = context_model

    def _fit(self, sequence_loader, timeseries_data_example):
        raise NotImplementedError()

    def _fit_context_model(self, transformed):
        template = self._context_model_template
        default_kwargs = {
            "primary_key": self._entity_columns,
            "field_types": {
                name: meta for name, meta in self._metadata.get_fields().items() if name in self._entity_columns
            },
        }
        if isinstance(template, tuple):
            context_model_class, context_model_kwargs = copy.deepcopy(template)
            if "primary_key" not in context_model_kwargs:
                context_model_kwargs["primary_key"] = self._entity_columns
                for keyword, argument in default_kwargs.items():
                    if keyword not in context_model_kwargs:
                        context_model_kwargs[keyword] = argument

            self._context_model = context_model_class(**context_model_kwargs)
        elif isinstance(template, type):
            self._context_model = template(**default_kwargs)
        else:
            self._context_model = copy.deepcopy(template)

        LOGGER.debug("Fitting context model %s", self._context_model.__class__.__name__)
        if self._context_columns:
            context = transformed[self._entity_columns + self._context_columns]
        else:
            context = transformed[self._entity_columns].copy()
            # Add constant column to allow modeling
            context[str(uuid.uuid4())] = 0

        context = context.groupby(self._entity_columns).first().reset_index()
        self._context_model.fit(context)

    def fit(self, sequence_loader, columns, types, examples):
        self._fit(sequence_loader, columns, types, examples)

    def get_metadata(self):
        """Get metadata about the table.

        This will return an ``sdv.metadata.Table`` object containing
        the information about the data that this model has learned.

        This Table metadata will contain some common information,
        such as field names and data types, as well as additional
        information that each Sub-class might add, such as the
        observed data field distributions and their parameters.

        Returns:
            sdv.metadata.Table:
                Table metadata.
        """
        return self._metadata

    def _sample(self, context=None, sequence_length=None):
        raise NotImplementedError()

    def sample(self, context=None, sequence_length=None):
        """Sample new sequences.

        Args:
            num_sequences (int):
                Number of sequences to sample. If context is
                passed, this is ignored. If not given, the
                same number of sequences as in the original
                timeseries_data is sampled.
            context (pandas.DataFrame):
                Context values to use when generating the sequences.
                If not passed, the context values will be sampled
                using the specified tabular model.
            sequence_length (int):
                If passed, sample sequences of this length. If not
                given, the sequence length will be sampled from
                the model.

        Returns:
            pandas.DataFrame:
                Table containing the sampled sequences in the same
                format as that he training data had.
        """
        num_sequences = 1  # TODO this was missing. Should it be an argument?
        if not self._entity_columns:
            if context is not None:
                raise TypeError("If there are no entity_columns, context must be None")

            context = pd.DataFrame(index=range(num_sequences or 1))
        elif context is None:
            context = self._context_model.sample(num_sequences)
            for column in self._entity_columns or []:
                if column not in context:
                    context[column] = range(len(context))

        sampled = self._sample(context, sequence_length)
        return sampled

    def save(self, path):
        """Save this model instance to the given path using pickle.

        Args:
            path (str):
                Path where the SDV instance will be serialized.
        """
        with open(path, "wb") as output:
            pickle.dump(self, output)

    @classmethod
    def load(cls, path):
        """Load a TabularModel instance from a given path.

        Args:
            path (str):
                Path from which to load the instance.

        Returns:
            TabularModel:
                The loaded tabular model.
        """
        with open(path, "rb") as f:
            return pickle.load(f)


class DeepEchoModel(BaseTimeseriesModel):
    """Base class for all the SDV Time series models based on DeepEcho."""

    _MODEL_CLASS = None
    _model_kwargs = None

    _DATA_TYPES = {
        "numerical": "continuous",
        "categorical": "categorical",
        "boolean": "categorical",
        "datetime": "datetime",
    }

    _verbose = True

    def _build_model(self):
        return self._MODEL_CLASS(**self._model_kwargs)

    def _transform_sequence_index(self, sequences):
        sequence_index_idx = self._data_columns.index(self._sequence_index)
        for sequence in sequences:
            data = sequence["data"]
            sequence_index = data[sequence_index_idx]
            diffs = np.diff(sequence_index).tolist()
            data[sequence_index_idx] = diffs[0:1] + diffs
            data.append(sequence_index[0:1] * len(sequence_index))

    def _fit(self, sequence_loader, columns, types, examples):

        self._model = self._build_model()

        # Specify output columns in self._data_columns
        self._output_columns = columns
        self._data_columns = [
            column for column in columns if column not in self._entity_columns + self._context_columns
        ]

        # Validate and fit
        dem = self
        self._model.fit_sequences(sequence_loader, columns, types, examples, dem)

    def _sample(self, context=None, sequence_length=None):
        """Sample new sequences.

        Args:
            context (pandas.DataFrame):
                Context values to use when generating the sequences.
                If not passed, the context values will be sampled
                using the specified tabular model.
            sequence_length (int):
                If passed, sample sequences of this length. If not
                given, the sequence length will be sampled from
                the model.

        Returns:
            pandas.DataFrame:
                Table containing the sampled sequences in the same
                format as that he training data had.
        """
        # Set the entity_columns as index to properly iterate over them
        if self._entity_columns:
            context = context.set_index(self._entity_columns)

        iterator = tqdm(context.iterrows(), disable=not self._verbose, total=len(context))

        output = list()
        for entity_values, context_values in iterator:
            context_values = context_values.tolist()
            sequence = self._model.sample_sequence(context_values, sequence_length)
            if self._sequence_index:
                sequence_index_idx = self._data_columns.index(self._sequence_index)
                diffs = sequence[sequence_index_idx]
                start = sequence[-1]
                sequence[sequence_index_idx] = np.cumsum(diffs) - diffs[0] + start

            # Reformat as a DataFrame
            group = pd.DataFrame(dict(zip(self._data_columns, sequence)), columns=self._data_columns)
            group[self._entity_columns] = entity_values
            for column, value in zip(self._context_columns, context_values):
                if column == self._sequence_index:
                    sequence_index = group[column]
                    group[column] = sequence_index.cumsum() - sequence_index.iloc[0] + value
                else:
                    group[column] = value

            output.append(group)

        output = pd.concat(output)
        return output[self._output_columns].reset_index(drop=True)


class PAREdit(DeepEchoModel):
    _MODEL_CLASS = PARModel

    def __init__(
        self,
        field_names=None,
        field_types=None,
        anonymize_fields=None,
        primary_key=None,
        entity_columns=None,
        context_columns=None,
        sequence_index=None,
        segment_size=None,
        context_model=None,
        table_metadata=None,
        epochs=128,
        sample_size=1,
        cuda=True,
        verbose=True,
    ):
        super().__init__(
            field_names=field_names,
            field_types=field_types,
            anonymize_fields=anonymize_fields,
            primary_key=primary_key,
            entity_columns=entity_columns,
            context_columns=context_columns,
            sequence_index=sequence_index,
            segment_size=segment_size,
            context_model=context_model,
            table_metadata=table_metadata,
        )

        self._model_kwargs = {
            "epochs": epochs,
            "sample_size": sample_size,
            "cuda": cuda,
            "verbose": verbose,
        }
        self._verbose = verbose
