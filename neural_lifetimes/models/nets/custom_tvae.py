import datetime
from collections import namedtuple

import numpy as np
import pandas as pd
import torch

# from ctgan.data_transformer import DataTransformer
from ctgan.synthesizers.base import BaseSynthesizer
from rdt.transformers import OneHotEncodingTransformer
from sklearn.mixture import BayesianGaussianMixture
from torch.nn import Linear, Module, Parameter, ReLU, Sequential
from torch.nn.functional import cross_entropy
from torch.optim import Adam

SpanInfo = namedtuple("SpanInfo", ["dim", "activation_fn"])
ColumnTransformInfo = namedtuple(
    "ColumnTransformInfo",
    [
        "column_name",
        "column_type",
        "transform",
        "transform_aux",
        "output_info",
        "output_dimensions",
    ],
)


class DataTransformer(object):
    """Data Transformer.

    Model continuous columns with a BayesianGMM and normalized to a scalar [0, 1] and a vector.
    Discrete columns are encoded using a scikit-learn OneHotEncoder.

    Args:
        max_clusters (int):
            Maximum number of Gaussian distributions in Bayesian GMM.
        weight_threshold (float):
            Weight threshold for a Gaussian distribution to be kept.
    """

    def __init__(self, max_clusters=10, weight_threshold=0.005):
        self._max_clusters = max_clusters
        self._weight_threshold = weight_threshold
        self.gm = BayesianGaussianMixture(
            n_components=self._max_clusters,
            weight_concentration_prior_type="dirichlet_process",
            weight_concentration_prior=0.001,
            n_init=1,
            warm_start=True,
        )

    def _fit_continuous(self, column_name, raw_column_data):
        """Train Bayesian GMM for continuous column."""
        self.gm.fit(raw_column_data.reshape(-1, 1))
        valid_component_indicator = self.gm.weights_ > self._weight_threshold
        num_components = valid_component_indicator.sum()

        return ColumnTransformInfo(
            column_name=column_name,
            column_type="continuous",
            transform=self.gm,
            transform_aux=valid_component_indicator,
            output_info=[SpanInfo(1, "tanh"), SpanInfo(num_components, "softmax")],
            output_dimensions=1 + num_components,
        )

    def _fit_discrete(self, column_name, raw_column_data):
        """Fit one hot encoder for discrete column."""
        ohe = OneHotEncodingTransformer(error_on_unknown=False)
        ohe.fit(raw_column_data)
        num_categories = len(ohe.dummies)

        return ColumnTransformInfo(
            column_name=column_name,
            column_type="discrete",
            transform=ohe,
            transform_aux=None,
            output_info=[SpanInfo(num_categories, "softmax")],
            output_dimensions=num_categories,
        )

    def fit(self, raw_data, discrete_columns=tuple()):
        """Fit self.

        GMM for continuous columns and One hot encoder for discrete columns.
        This step also counts the #columns in matrix data, and span information.
        """
        self.output_info_list = []
        self.output_dimensions = 0

        if not isinstance(raw_data, pd.DataFrame):
            self.dataframe = False
            raw_data = pd.DataFrame(raw_data)
        else:
            self.dataframe = True

        self._column_raw_dtypes = raw_data.infer_objects().dtypes

        self._column_transform_info_list = []
        for column_name in raw_data.columns:
            raw_column_data = raw_data[column_name].values
            if column_name in discrete_columns:
                column_transform_info = self._fit_discrete(column_name, raw_column_data)
            else:
                column_transform_info = self._fit_continuous(column_name, raw_column_data)

            self.output_info_list.append(column_transform_info.output_info)
            self.output_dimensions += column_transform_info.output_dimensions
            self._column_transform_info_list.append(column_transform_info)

    def _transform_continuous(self, column_transform_info, raw_column_data):
        self.gm = column_transform_info.transform

        valid_component_indicator = column_transform_info.transform_aux
        num_components = valid_component_indicator.sum()

        means = self.gm.means_.reshape((1, self._max_clusters))
        stds = np.sqrt(self.gm.covariances_).reshape((1, self._max_clusters))
        normalized_values = ((raw_column_data - means) / (4 * stds))[:, valid_component_indicator]
        component_probs = self.gm.predict_proba(raw_column_data)[:, valid_component_indicator]

        selected_component = np.zeros(len(raw_column_data), dtype="int")
        for i in range(len(raw_column_data)):
            component_porb_t = component_probs[i] + 1e-6
            component_porb_t = component_porb_t / component_porb_t.sum()
            selected_component[i] = np.random.choice(np.arange(num_components), p=component_porb_t)

        selected_normalized_value = normalized_values[np.arange(len(raw_column_data)), selected_component].reshape(
            [-1, 1]
        )
        selected_normalized_value = np.clip(selected_normalized_value, -0.99, 0.99)

        selected_component_onehot = np.zeros_like(component_probs)
        selected_component_onehot[np.arange(len(raw_column_data)), selected_component] = 1
        return [selected_normalized_value, selected_component_onehot]

    def _transform_discrete(self, column_transform_info, raw_column_data):
        ohe = column_transform_info.transform
        return [ohe.transform(raw_column_data)]

    def transform(self, raw_data):
        """Take raw data and output a matrix data."""
        if not isinstance(raw_data, pd.DataFrame):
            raw_data = pd.DataFrame(raw_data)

        column_data_list = []
        for column_transform_info in self._column_transform_info_list:
            column_data = raw_data[[column_transform_info.column_name]].values
            if column_transform_info.column_type == "continuous":
                column_data_list += self._transform_continuous(column_transform_info, column_data)
            else:
                assert column_transform_info.column_type == "discrete"
                column_data_list += self._transform_discrete(column_transform_info, column_data)

        return np.concatenate(column_data_list, axis=1).astype(float)

    def _inverse_transform_continuous(self, column_transform_info, column_data, sigmas, st):
        # self.gm = column_transform_info.transform
        valid_component_indicator = column_transform_info.transform_aux

        selected_normalized_value = column_data[:, 0]
        selected_component_probs = column_data[:, 1:]

        if sigmas is not None:
            sig = sigmas[st]
            selected_normalized_value = np.random.normal(selected_normalized_value, sig)

        selected_normalized_value = np.clip(selected_normalized_value, -1, 1)
        component_probs = np.ones((len(column_data), self._max_clusters)) * -100
        component_probs[:, valid_component_indicator] = selected_component_probs

        means = self.gm.means_.reshape([-1])
        stds = np.sqrt(self.gm.covariances_).reshape([-1])
        selected_component = np.argmax(component_probs, axis=1)

        std_t = stds[selected_component]
        mean_t = means[selected_component]
        column = selected_normalized_value * 4 * std_t + mean_t

        return column

    def _inverse_transform_discrete(self, column_transform_info, column_data):
        ohe = column_transform_info.transform
        return ohe.reverse_transform(column_data)

    def inverse_transform(self, data, sigmas=None):
        """Take matrix data and output raw data.

        Output uses the same type as input to the transform function.
        Either np array or pd dataframe.
        """
        st = 0
        recovered_column_data_list = []
        column_names = []
        for column_transform_info in self._column_transform_info_list:
            dim = column_transform_info.output_dimensions
            column_data = data[:, st : st + dim]

            if column_transform_info.column_type == "continuous":
                recovered_column_data = self._inverse_transform_continuous(
                    column_transform_info, column_data, sigmas, st
                )
            else:
                assert column_transform_info.column_type == "discrete"
                recovered_column_data = self._inverse_transform_discrete(column_transform_info, column_data)

            recovered_column_data_list.append(recovered_column_data)
            column_names.append(column_transform_info.column_name)
            st += dim

        recovered_data = np.column_stack(recovered_column_data_list)
        recovered_data = pd.DataFrame(recovered_data, columns=column_names).astype(self._column_raw_dtypes)
        if not self.dataframe:
            recovered_data = recovered_data.values

        return recovered_data

    def convert_column_name_value_to_id(self, column_name, value):
        discrete_counter = 0
        column_id = 0
        for column_transform_info in self._column_transform_info_list:
            if column_transform_info.column_name == column_name:
                break
            if column_transform_info.column_type == "discrete":
                discrete_counter += 1

            column_id += 1

        else:
            raise ValueError(f"The column_name `{column_name}` doesn't exist in the data.")

        one_hot = column_transform_info.transform.transform(np.array([value]))[0]
        if sum(one_hot) == 0:
            raise ValueError(f"The value `{value}` doesn't exist in the column `{column_name}`.")

        return {
            "discrete_column_id": discrete_counter,
            "column_id": column_id,
            "value_id": np.argmax(one_hot),
        }


class Encoder(Module):
    def __init__(self, data_dim, compress_dims, embedding_dim):
        super(Encoder, self).__init__()
        dim = data_dim
        seq = []
        for item in list(compress_dims):
            seq += [Linear(dim, item), ReLU()]
            dim = item
        self.seq = Sequential(*seq)
        self.fc1 = Linear(dim, embedding_dim)
        self.fc2 = Linear(dim, embedding_dim)

    def forward(self, input):
        feature = self.seq(input)
        mu = self.fc1(feature)
        logvar = self.fc2(feature)
        std = torch.exp(0.5 * logvar)
        return mu, std, logvar


class Decoder(Module):
    def __init__(self, embedding_dim, decompress_dims, data_dim):
        super(Decoder, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(decompress_dims):
            seq += [Linear(dim, item), ReLU()]
            dim = item

        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)
        self.sigma = Parameter(torch.ones(data_dim) * 0.1)

    def forward(self, input):
        return self.seq(input), self.sigma


def loss_function(recon_x, x, sigmas, mu, logvar, output_info, factor, discrete_dimension):
    st = 0
    loss = []
    for column_info in output_info:
        for span_info in column_info:
            if span_info.activation_fn != "softmax":
                ed = st + span_info.dim
                std = sigmas[st]
                loss.append(((x[:, st] - torch.tanh(recon_x[:, st])) ** 2 / 2 / (std**2)).sum())
                loss.append(torch.log(std) * x.size()[0])
                st = ed

            else:
                st = max(st, recon_x.size()[1] - discrete_dimension)
                ed = st + span_info.dim

                list_ = list(recon_x[st:ed].size())
                if not list_[0] * list_[1] == 0:
                    loss.append(
                        cross_entropy(
                            recon_x[st:ed],
                            torch.argmax(x[st:ed], dim=-1),
                            reduction="sum",
                        )
                    )
                st = ed

    # assert st == recon_x.size()[1] We no longer assert this since dimensions vary across batches
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return sum(loss) * factor / x.size()[0], KLD / x.size()[0]


class TVAESynthesizer(BaseSynthesizer):
    """TVAESynthesizer."""

    def __init__(
        self,
        embedding_dim=128,
        compress_dims=(128, 128),
        decompress_dims=(128, 128),
        l2scale=1e-5,
        batch_size=500,
        epochs=300,
        loss_factor=2,
        cuda=True,
    ):

        self.embedding_dim = embedding_dim
        self.compress_dims = compress_dims
        self.decompress_dims = decompress_dims

        self.l2scale = l2scale
        self.batch_size = batch_size
        self.loss_factor = loss_factor
        self.epochs = epochs

        if not cuda or not torch.cuda.is_available():
            device = "cpu"
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = "cuda"

        self._device = torch.device(device)

    def fit(self, train_loader, data_example, discrete_columns=tuple()):

        # Split Data Example
        continuous_columns = set(data_example.columns) - set(discrete_columns)

        example_discrete = data_example[list(discrete_columns)]
        example_continuous = data_example[continuous_columns]  # .values

        # TODO can't have a hardcoded hyperparameter buried here
        # This is a hyperparameter. Change it according to the number and distribution of continuous features.
        continuous_output_dim = 10

        # DataTransformer with pre-fit One-Hot Encoder
        self.transformer_continuous = DataTransformer()
        self.transformer_continuous.fit(example_continuous)

        self.transformer_discrete = DataTransformer()
        self.transformer_discrete.fit(example_discrete, discrete_columns)

        # Encoder and Decoder in TVAE
        data_dim = continuous_output_dim + self.transformer_discrete.output_dimensions

        self.encoder = Encoder(data_dim, self.compress_dims, self.embedding_dim).to(self._device)
        self.decoder = Decoder(self.embedding_dim, self.compress_dims, data_dim).to(self._device)

        optimizerAE = Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            weight_decay=self.l2scale,
        )

        for i in range(self.epochs):
            print(f"Start epoch: {i+1}/{self.epochs}")
            for id_, data in enumerate(train_loader):
                data = pd.DataFrame(data)
                data_continuous = data[list(continuous_columns)]  # .values

                data_discrete = data[list(discrete_columns)]
                self.transformer_continuous.fit(data_continuous)  # We only fit the continuous part of the transoformer

                transformed_discrete = self.transformer_discrete.transform(data_discrete)
                transformed_continuous = self.transformer_continuous.transform(data_continuous)
                shaped_continuous = np.concatenate(
                    [
                        transformed_continuous,
                        np.zeros(shape=(len(transformed_continuous), continuous_output_dim)),
                    ],
                    axis=1,
                )[:, :continuous_output_dim]

                transformed_data = np.concatenate([shaped_continuous, transformed_discrete], axis=1).astype(float)

                batch = torch.Tensor(transformed_data)

                # training
                optimizerAE.zero_grad()
                real = batch.to(self._device)

                mu, std, logvar = self.encoder(real)
                eps = torch.randn_like(std)
                emb = eps * std + mu
                rec, sigmas = self.decoder(emb)

                rec = torch.cat(
                    (
                        rec[:, :continuous_output_dim],
                        rec[:, -self.transformer_discrete.output_dimensions :],
                    ),
                    dim=1,
                )

                output_info_list_ = (
                    self.transformer_continuous.output_info_list + self.transformer_discrete.output_info_list
                )

                loss_1, loss_2 = loss_function(
                    rec,
                    real,
                    sigmas,
                    mu,
                    logvar,
                    output_info_list_,
                    self.loss_factor,
                    self.transformer_discrete.output_dimensions,
                )
                loss = loss_1 + loss_2
                loss.backward()
                optimizerAE.step()
                self.decoder.sigma.data.clamp_(0.01, 1.0)

                if id_ % 1000 == 0:
                    print(
                        f"Iteration {id_}/{len(train_loader)}. Loss {loss}, loss 1: {loss_1}, loss 2: {loss_2}."
                        f"Timestamp: {datetime.datetime.now()}"
                    )

    def sample(self, samples):
        self.decoder.eval()

        steps = samples // self.batch_size + 1
        data = []
        for _ in range(steps):
            mean = torch.zeros(self.batch_size, self.embedding_dim)
            std = mean + 1
            noise = torch.normal(mean=mean, std=std).to(self._device)
            fake, sigmas = self.decoder(noise)
            fake = torch.tanh(fake)
            data.append(fake.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:samples]

        # Split generated into continuous and discrete part
        sampled_continuous = data[:, : self.transformer_continuous.output_dimensions]
        sampled_discrete = data[:, -self.transformer_discrete.output_dimensions :]

        sigmas_detached = sigmas.detach().cpu().numpy()

        # Recover data using inverse transform
        recovered_continuous = self.transformer_continuous.inverse_transform(sampled_continuous, sigmas=sigmas_detached)
        recovered_discrete = self.transformer_discrete.inverse_transform(sampled_discrete, sigmas=sigmas_detached)

        recovered_data = pd.concat([recovered_continuous, recovered_discrete], axis=1)

        return recovered_data

    def set_device(self, device):
        self._device = device
        self.decoder.to(self._device)
