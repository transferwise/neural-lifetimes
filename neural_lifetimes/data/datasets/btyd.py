import datetime
import numbers
import os
import pickle
import random
from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Union
from uuid import uuid4

import numpy as np
import torch
from plotly.graph_objects import Figure
from scipy.special import hyp2f1

from neural_lifetimes.utils.date_arithmetic import date2time

from .sequence_dataset import SequenceDataset


def _is_numeric(x: Any) -> bool:
    return isinstance(x, numbers.Number)


def _is_array_like(x: Any) -> bool:
    return isinstance(x, (list, tuple))


class NotTrackedError(Exception):
    pass


class GenMode:
    """
    A class for generating modes for the BTYD model.

    To specify a mode, you must specify the following:
    - (a,b) or mean_p_churn for distribution of p_churn
    - (r,alpha) or mean_lambda for distribution of lambda
    - discreteDist for distribution of discrete events
    - contDist for distribution of continuous events

    Args:
        a (float): a parameter for the p_churn distribution. Use this with ``b``, or just ``mean_p_churn``.
        b (float): b parameter for the p_churn distribution. Use this with ``a``, or just ``mean_p_churn``.
        r (float): r parameter for the lambda distribution. Use this with ``alpha``, or just ``mean_lambda``.
        alpha (float): alpha parameter for the lambda distribution. Use this with ``r``, or just ``mean_lambda``.
        mean_p_churn (float): mean of the p_churn distribution. Use either this or ``a`` and ``b``.
        mean_lambda (float): mean of the lambda distribution. Use either this or ``r`` and ``alpha``.
        discrete_dist (dict): dictionary of discrete events
        cont_dist (dict): dictionary of continuous events

    Attributes:
        a (float): The a parameter of the BTYD model.
        b (float): The b parameter of the BTYD model.
        r (float): The r parameter of the BTYD model.
        alpha (float): The alpha parameter of the BTYD model.
        discrete_dist (dict): The discrete distribution of the BTYD model.
        cont_dist (dict): The continuous distribution of the BTYD model.
    """

    def __init__(
        self,
        a: float = 0.0,
        b: float = 0.0,
        r: float = 0.0,
        alpha: float = 0.0,
        mean_lambda: float = 0.0,
        mean_p_churn: float = 0.0,
        discrete_dist: Optional[dict] = None,
        cont_dist: Optional[dict] = None,
    ):
        # TODO Mark: Presumably we don't want to set mean_* and (a, b, r, alpha)
        self.a = a
        self.b = b
        self.r = r
        self.alpha = alpha
        self.discrete_dist = discrete_dist or {}
        self.cont_dist = cont_dist or {}
        if mean_p_churn > 0:
            # proposed conversion from mean_p_churn to a,b
            # can change if needed
            # this proposal makes mode = 0 since self.a = 1, self.b > 0
            self.a = 1.0
            self.b = 1.0 / mean_p_churn - 1.0
        if mean_lambda > 0:
            # proposed conversion from mean_lambda to r,alpha
            # can change if needed
            # this proposal makes mode = max(0, mean_lambda)
            self.r = max(1.0, mean_lambda)
            self.alpha = min(1.0, mean_lambda)


# TODO: data_dir should be removed
class BTYD(SequenceDataset):
    """
    BTYD synthetic dataset.

    Creates a synthetic dataset of customers using the Buy-Till You Die (BTYD) model.
    Note that the number of customers has to be specified. If you want to create a
    multimodal dataset, it is recommended you use the
    :mod:`from_modes <neural_lifetimes.datasets.btyd_dataset.BTYD.from_modes>` method
    and use :mod:`GenMode <neural_lifetimes.datasets.btyd_modes.GenMode>` class to create the
    modes.

    Attributes:
        continuous_features (List[str]): List of continuous features. They will be
            sampled from normal distributions.
        discrete_features (List[str]): List of discrete features. They will be sampled
            from the list of possible values uniformly.
        data_dir (str): Path to data directory.
        start_date (datetime.datetime): earliest start date in days timestamp.
        start_limit_date (datetime.datetime): latest start date in days timestamp.
        end_date (datetime.datetime): end date in days timestamp.
        num_customers (int): number of customers to be simulated. Min ca 100. Otherwise,
            Pytorch lightning won't train.
        a (List[float]): Parameter(s) for Prior of p: Beta(a,b). If a list is passed,
            then it is assumed to be a multimodal model.
        b (List[float]): Parameter(s) for Prior of p: Beta(a,b). If ``a`` is a list,
            then this must also be a list of the same length.
        r (List[float]): Parameter(s) for prior of lambda: Gamma(r,alpha). If ``a`` is a
            list, then this must also be a list of the same length.
        alpha (List[float]): Parameter(s) for prior of lambda: Gamma(r,alpha). If ``a``
            is a list, then this must also be a list of the same length.
        multimodal (bool): Inferred by checking type of ``a``. If true, use multi-modal
            distributions.
        discrete_dists (dict): If we have multimodal input then assigned distributions
            gives the distributions for some features for each mode
        cont_dists (Optional[List[Dict[str, Any]]]): A list of dictionaries setting
            the distributions of certain continuous features. Each feature is normally
            distributed.
        num_modes (int): Number of modes. Is just len(self.a).
        feature_funcs (Dict[str, Callable]): Dictionary of feature functions.
        mode_ratios (Optional[List[float]]): Gives the distribution over the modes
    """

    def __init__(
        self,
        num_customers: int,
        a: Union[float, List[float]],
        b: Union[float, List[float]],
        r: Union[float, List[float]],
        alpha: Union[float, List[float]],
        start_date: datetime.date,
        start_limit_date: datetime.date,
        end_date: datetime.date,
        continuous_features: List[str],
        discrete_features: List[str],
        seq_gen_dynamic: bool,
        data_dir: str,
        discrete_dists: Optional[List[Dict[str, Any]]] = None,
        cont_dists: Optional[List[Dict[str, Any]]] = None,
        mode_ratios: Optional[List[float]] = None,
        track_statistics: bool = False,
    ):
        """
        Create a BTYD synthetic dataset.

        Arguments:
            num_customers (int): number of customers to be simulated. Min ca 100.
                Otherwise, Pytorch lightning won't train.
            a (Union[float, List[float]]): Parameter for prior of p. Beta(a,b)
            b (Union[float, List[float]]): Parameter for prior of p. Beta(a,b)
            r (Union[float, List[float]]): Parameter for prior of lambda. Gamma(r,alpha)
            alpha (Union[float, List[float]]): Parameter for prior of lambda. Gamma(r,alpha)
            start_date (datetime.date): earliest start date in days timestamp.
            start_limit_date (datetime.date): latest start date in days timestamp.
            end_date (datetime.date): end date in days timestamp.
            discrete_dists (Optional[List[Dict[str, Any]]]): A list of dictionaries setting
                the distributions of certain discrete features.
            cont_dists (Optional[List[Dict[str, Any]]]): A list of dictionaries setting the
                distributions of certain continuous features. Each feature is normally
                distributed.
            mode_ratios (Optional[List[float]]): Gives the distribution over the modes
            track_statistics bool: Track a range of statistics should about generated data.
        """
        # settings
        self.continuous_feature_names = continuous_features
        self.discrete_feature_names = discrete_features

        self.data_dir = data_dir
        self.start_date = date2time(start_date)  # start date in days timestamp
        self.start_limit_date = date2time(start_limit_date)  # start date in days timestamp
        self.end_date = date2time(end_date)  # end date in days timestamp
        self.seq_gen_dynamic = seq_gen_dynamic

        # dataset length
        self.num_customers = num_customers

        # BTYD parameters
        if _is_numeric(a):
            assert (
                _is_numeric(b) and _is_numeric(r) and _is_numeric(a)
            ), "All statistical parameters should be lists or all should be scalars."
            self.a, self.b = [a], [b]
            self.r, self.alpha = [r], [alpha]
        else:
            assert (
                _is_array_like(a) and _is_array_like(b) and _is_array_like(r) and _is_array_like(alpha)
            ), "All statistical parameters should be lists or all should be scalars."
            self.a, self.b = a, b
            self.r, self.alpha = r, alpha

        assert (
            len(self.a) == len(self.b) == len(self.alpha) == len(self.r)
        ), "Parameter lists have different lengths. Must be equal."

        if mode_ratios is not None:
            assert len(self.a) == len(mode_ratios), (
                f"`mode_ratios` has to match the number of modes in length. Received {len(self.a)} modes and"
                f" {len(mode_ratios)} `mode_ratio`s instead."
            )

        # we have asserted that all are iterable or not, so checking a is sufficient to
        # see if we want a multimodal model
        # TODO Cornelius: This is incorrect
        self.multimodal = _is_array_like(a)
        self.num_modes = len(self.a)

        self.discrete_dists = discrete_dists if discrete_dists is not None else [{} for _ in range(self.num_modes)]
        self.cont_dists = cont_dists if cont_dists is not None else [{} for _ in range(self.num_modes)]
        self.mode_ratios = mode_ratios if self.multimodal else [1.0]
        self.feature_funcs = self._get_feature_dict()
        self.cache = {}

        self.track_statistics = track_statistics
        if self.track_statistics:
            self._num_generated_sequences = 0
            self._num_truncated_sequences = 0
            self._range_transaction_days = deque(maxlen=self.num_customers)
            self._start_dates = deque(maxlen=self.num_customers)
            self._end_dates = deque(maxlen=self.num_customers)
            self._all_transactions = deque(maxlen=self.num_customers * 100)
            self._num_events_per_customer = deque(maxlen=self.num_customers)

    @classmethod
    def from_modes(
        cls,
        num_customers: int,
        modes: List[GenMode],
        start_date: datetime.date,
        start_limit_date: datetime.date,
        end_date: datetime.date,
        continuous_features: List[str],
        discrete_features: List[str],
        seq_gen_dynamic: bool,
        data_dir: str,
        mode_ratios: Optional[List[float]] = None,
        track_statistics: bool = False,
    ):
        """
        Initialize BTYD dataset from modes.

        See :mod:`GenMode <neural_lifetimes.datasets.btyd_modes.GenMode>` for creating the
        modes.

        Args:
            num_customers (int): number of customers to be simulated. Min ca 100.
                Otherwise, Pytorch lightning won't train.
            modes (List[GenMode]): List of modes.
            start_date (datetime.date): earliest start date in days timestamp.
            start_limit_date (datetime.date): latest start date in days timestamp.
            end_date (datetime.date): end date in days timestamp.
            continuous_features (List[str]): List of continuous features.
            discrete_features (List[str]): List of discrete features.
            seq_gen_dynamic (bool): If true, generate sequences dynamically.
            mode_ratios (Optional): List of mode_ratios of modes. If ``None``, then all modes
                are assumed to have equal probability.
            track_statistics bool: Track a range of statistics should about generated data.
        Returns:
            BTYD: BTYD dataset.
        """
        a = []
        b = []
        r = []
        alpha = []
        discrete_dists = []
        cont_dists = []
        for mode in modes:
            a.append(mode.a)
            b.append(mode.b)
            r.append(mode.r)
            alpha.append(mode.alpha)
            discrete_dists.append(mode.discrete_dist)
            cont_dists.append(mode.cont_dist)

        if mode_ratios is None:
            mode_ratios = [1.0 for _ in range(len(modes))]
        mode_ratios = [r / sum(mode_ratios) for r in mode_ratios]

        return cls(
            num_customers,
            a,
            b,
            r,
            alpha,
            start_date,
            start_limit_date,
            end_date,
            continuous_features,
            discrete_features,
            seq_gen_dynamic,
            data_dir,
            discrete_dists=discrete_dists,
            cont_dists=cont_dists,
            mode_ratios=mode_ratios,
            track_statistics=track_statistics,
        )

    @property
    def asof_time(self) -> datetime.datetime:
        """
        Get the time of the last possible transaction.

        Needed for compatibility with SequenceLoader.
        """
        return self.end_date

    def __len__(self):
        return self.num_customers

    # def __getitem__(self, customer_id: int) -> int:
    #     # TODO Mark: Why just return the length? Seems odd for this method
    #     if customer_id not in self.cache:
    #         # cache to return a consistent response to get_bulk
    #         self.cache[customer_id] = self._single_sample()
    #     return self._length_of_cached_transaction_sequence(customer_id)

    def get_seq_len(self, customer_id: int) -> int:
        if customer_id not in self.cache:
            self.cache[customer_id] = self._single_sample()
        return len(next(iter(self.cache[customer_id].values())))

    # def _length_of_cached_transaction_sequence(self, customer_id: int) -> int:
    #     """
    #     Get the length of the transaction sequence which has been cached for a given
    #     customer_id.

    #     Args:
    #         customer_id (int): id of the customer for whom we've already generated a
    #             sequence of transactions.

    #     Returns:
    #         int: length of the generated sequence
    #     """
    #     return len(next(iter(self.cache[customer_id].values())))

    def _single_item_from_cache(self, idx):
        if idx not in self.cache:
            self.cache[idx] = self._single_sample()

        if self.seq_gen_dynamic:
            return self.cache.pop(idx)
        else:
            return self.cache[idx]

    def _load_batch(self, customer_ids: Sequence[int]) -> List[Dict[str, np.ndarray]]:
        """
        Get transactions for a sequence of customer ids from the cache.

        If seq_gen_dynamic is set, the customer_id will be popped from the cache.

        Args:
            customer_ids (Sequence[int]): A sequence of customer_ids

        Returns:
            List[Dict[str, np.ndarray]]: A list of records of transactions per customer
                in the order provided by the input sequence.

        Raises:
            KeyError: A customer_id provided was not present in the cache.
        """
        # TODO Mark: Looks like we need to ask for the length before we can get the
        # cached value?
        # load into cache

        return [self._single_item_from_cache(i) for i in customer_ids]

    def timestamp2float(self, timestamp: datetime.datetime) -> float:
        return self.timedelta2float(timestamp - self.start_date)

    def float2timestamp(self, time: float) -> datetime.datetime:
        return self.start_date + datetime.timedelta(days=time)

    def timedelta2float(self, timedelta: datetime.timedelta) -> float:
        return timedelta.days + timedelta.seconds / 60 / 60 / 24

    def float2timedelta(self, time: float) -> datetime.timedelta:
        days = int(time)
        seconds = (time - days) * 24 * 60 * 60
        return datetime.timedelta(days=days, seconds=seconds)

    @property
    def time_interval(self) -> float:
        """The time interval between start and end dates given in the unit of the underlying stochastic process: Hours.

        Returns:
            float: time interval
        """
        return self.timedelta2float(self.end_date - self.start_date)

    def _feature_select_cont(self, size: int, mode: int, feature_name: str) -> np.ndarray:
        """
        TODO doc.

        Args:
            size (int): number of transactions
            mode (int): mode of particular user
            feature_name (str): name of the continuous feature

        Returns:
            np.ndarray: a normally distributed array
        """
        if self.multimodal and feature_name in self.cont_dists[mode]:
            return np.random.normal(
                size=size,
                loc=self.cont_dists[mode][feature_name][0],
                scale=self.cont_dists[mode][feature_name][1],
            )
        else:
            return np.random.normal(size=size)

    def _feature_select_discrete(
        self,
        size: int,
        mode: int,
        val: np.ndarray,
        feature_name: str,
    ) -> np.ndarray:
        """
        TODO doc.

        Args:
            size (int): size of returned array
            mode (int): mode of particular user
            val (np.ndarray): list of potential values which the feature can take
            feature_name (str): name of the discrete feature

        Returns:
            np.ndarray[str]: An array of values from val sampled according to the
            distribution of the specified feature
        """
        p = None
        if self.multimodal and feature_name in self.discrete_dists[mode]:
            # We want p to give the probability of each element of val
            p = np.zeros(len(val))
            for i in range(len(val)):
                if val[i] in self.discrete_dists[mode][feature_name]:
                    p[i] = self.discrete_dists[mode][feature_name][val[i]]

        return np.random.choice(val.reshape(-1), size=size, replace=True, p=p)

    def get_discrete_feature_values(self, start_token: Any = None) -> Dict[str, np.ndarray]:
        """
        Load the discrete feature values from the data pickle.

        Args:
            start_token (Any, optional): If not None, include a start_token in each of the feature lists.
                Defaults to None.

        Returns:
            Dict[str, np.ndarray]: A mapping between feature names and allowed feature values.
        """
        with open(os.path.join(self.data_dir, "discrete_values.pkl"), "rb") as f:
            discrete_feat_values = pickle.load(f)

        discrete_feat_values["btyd_mode"] = np.array(range(len(self.a)))
        discrete_feat_values = {k: discrete_feat_values[k] for k in self.discrete_feature_names}

        if start_token is not None:
            discrete_feat_values = {k: np.append(v, start_token) for k, v in discrete_feat_values.items()}

        return discrete_feat_values

    def _get_feature_dict(self):
        """
        Return a dictionary of feature functions.

        Returns:
            Dict[str, Callable]: Dictionary of discrete, continuous features, and user
                ids.

        Note:
            Requires discrete_values.pkl to get the discrete values. Continuous features
            are specified in the corresponding settings files.
        """
        discrete_funcs = {}
        discrete_feat_values = self.get_discrete_feature_values()
        for k, v in discrete_feat_values.items():
            discrete_funcs[k] = lambda input_: self._feature_select_discrete(input_[0], input_[1], v, k)

        # get continuous values
        continuous_features = self.continuous_feature_names
        cont_funcs = {}
        for feature in continuous_features:
            cont_funcs[feature] = lambda input_: self._feature_select_cont(input_[0], input_[1], feature)

        # user id
        special_feat = {"USER_PROFILE_ID": lambda input_: np.full(shape=input_[0], fill_value=str(uuid4()))}

        return {**cont_funcs, **discrete_funcs, **special_feat}

    def _single_sample(self) -> Dict[str, np.ndarray]:
        """
        Produce a series of BTYD transactions sampled according to the provided distribution parameters and modes.

        Returns:
            Dict[str, np.ndarray]: A mapping from features to arrays describing the
                transaction series.
        """
        mode = np.random.choice(range(self.num_modes), p=self.mode_ratios)

        # sample lambdas and p's
        lambda_ = np.random.gamma(self.r[mode], 1 / self.alpha[mode])
        p = np.random.beta(self.a[mode], self.b[mode])

        lambda_ = max(lambda_, 1 / (365 * 24))  # TODO Why 365*24? should it be changed to 365 now that time is in days.

        # sample actual data
        start_delta = self.start_limit_date - self.start_date
        start_delta_seconds = int(self.timedelta2float(start_delta) * 24 * 60 * 60)
        transactions = [self.start_date + datetime.timedelta(seconds=random.randrange(start_delta_seconds))]

        j = 1
        alive = True
        while alive:
            u = np.random.uniform()
            if u > p:
                new_transaction = datetime.timedelta(days=np.random.exponential(1.0 / lambda_)) + transactions[-1]
                if new_transaction > self.end_date:
                    alive = False
                    if self.track_statistics:
                        self._num_truncated_sequences += 1
                else:
                    transactions.append(new_transaction)
                    j += 1
            else:
                alive = False

        if self.track_statistics:
            # update state variable to track dataset generation
            self._num_generated_sequences += 1

            self._start_dates.append(transactions[0])
            self._end_dates.append(transactions[-1])
            self._range_transaction_days.append(self.timedelta2float(transactions[-1] - transactions[0]))
            self._num_events_per_customer.append(len(transactions))
            self._all_transactions.extend(transactions)

        transactions = np.array(transactions)
        features = {k: func((len(transactions), mode)) for k, func in self.feature_funcs.items()}

        out = {
            "t": transactions,
            **features,
            "lambda": np.array([lambda_] * len(transactions)),
            "p": np.array([p] * len(transactions)),
            "btyd_mode": np.array([mode] * len(transactions)),
        }

        return out

    # TODO Add support for calculating these from observed p and lambdas per customer.
    def expected_num_transactions_from_parameters(self, p: float, lambda_: float) -> float:
        """Compute the expected number of transactions for a customer with parameters `p` and `lambda`.

        Args:
            p (float): The churn probability
            lambda_ (float): The rate parameter

        Returns:
            float: Expected number of transactions
        """
        with torch.no_grad():
            return 1 / p - 1 / p * torch.exp(-lambda_ * p * torch.tensor(self.time_interval))

    def expected_p_churn_from_priors(
        self,
        a: Optional[List[float]] = None,
        b: Optional[List[float]] = None,
    ) -> List[float]:
        """Compute the expected churn probablity from priors per mode.

        Args:
            a (Optional[Union[float, List[float]]], optional): Parameter of the Beta distribution. Defaults to None.
                If None, use self.a.
            b (Optional[Union[float, List[float]]], optional): Parameter of the Beta distribution. Defaults to None.
                If None, use self.b.
        Returns:
            List[float]: List containing expected churn probabilities.
        """
        assert (a is None) == (b is None)

        if a is None:
            as_, bs = self.a, self.b
        else:
            as_, bs = a, b

        e_per_mode = []

        for a, b in zip(as_, bs):
            assert a > 0 and b > 0, "Parameters must have positive values."
            e_per_mode.append(a / (a + b))
        return e_per_mode

    def expected_time_interval_from_priors(
        self,
        r: Optional[Union[float, List[float]]] = None,
        alpha: Optional[Union[float, List[float]]] = None,
    ) -> List[float]:
        """Compute the expected time between transactions from priors per mode.

        Args:
            a (Optional[Union[float, List[float]]], optional): Parameter of the Gamma distribution. Defaults to None.
                If None, use self.r.
            b (Optional[Union[float, List[float]]], optional): Parameter of the Gamma distribution. Defaults to None.
                If None, use self.alpha.
        Returns:
            List[float]: List containing expected time intervals.
        """
        assert (r is None) == (alpha is None)

        if r is None:
            rs, alphas = self.r, self.alpha
        else:
            rs, alphas = r, alpha

        e_per_mode = []

        for r, alpha in zip(rs, alphas):
            assert r > 0 and alpha > 0, "Parameters must have positive values."
            e_per_mode.append(alpha / r)
        return e_per_mode

    def expected_num_transactions_from_priors(
        self,
        a: Optional[List[float]] = None,
        b: Optional[List[float]] = None,
        r: Optional[List[float]] = None,
        alpha: Optional[List[float]] = None,
        time_interval: Optional[float] = None,
    ) -> List[float]:
        """
        Compute the expected number of transactions per customer from priors per mode.

        If parameters are provided, these will be used to to calculated the expected value.

        Otherwise, the internal priors are used.

        See Fader et al. (http://brucehardie.com/papers/018/fader_et_al_mksc_05.pdf) for details.

        Args:
            a (Optional[List[float]]): Parameter of the Beta distribution.
                If None, use self.a.
            b (Optional[List[float]]): Parameter of the Beta distribution.
                If None, use self.b.
            r (Optional[List[float]]): Parameter of the Gamma
                distribution. If None, use self.r.
            alpha (Optional[List[float]]): Parameter of the Gamma
                distribution. If None, use self.alpha.
            time_interval (Optional[float]): The time interval for the expected number of transactions.
                These determine the integration bounds. Must be finite.

        Returns:
            List[float]: Expected customer lifetime values.
        """
        assert (a is None) == (b is None) == (r is None) == (alpha is None) == (time_interval is None)

        if a is None:
            as_, bs = self.a, self.b
            rs, alphas = self.r, self.alpha
            time_interval = [self.time_interval] * len(self.a)
        else:
            as_, bs = a, b
            rs, alphas = r, alpha

        e_per_mode = []
        eps = 0.00001
        for a, b, r, alpha, t_i in zip(as_, bs, rs, alphas, time_interval):
            z1 = (a + b - 1) / (a - 1 + eps)
            z2 = (alpha / (alpha + t_i)) ** r
            z3 = hyp2f1(r, b, a + b - 1, t_i / (alpha + t_i))
            # hyp2f1(r,b;a+b-1;t/(t+alpha))
            z4 = 1 - z2 * z3
            e_per_mode.append(z1 * z4)
        return e_per_mode

    def expected_num_transactions_from_priors_and_history(
        self,
        a: Optional[List[float]] = None,
        b: Optional[List[float]] = None,
        r: Optional[List[float]] = None,
        alpha: Optional[List[float]] = None,
        time_interval: Optional[float] = None,
        asof_time: Optional[float] = None,
        num_transactions: Optional[List[float]] = None,
        last_transaction: Optional[List[float]] = None,
    ) -> List[float]:
        """Compute the expected number of transactions per customer from priors and past transactions per customer.

        If parameters are provided, these will be used to to calculated the expected value. Otherwise, the internal
        data are used.

        See Fader et al. (http://brucehardie.com/papers/018/fader_et_al_mksc_05.pdf) Eq 10 for details.

        Args:
            a (Optional[List[float]]): Parameter of the Beta distribution.
                If None, use self.a.
            b (Optional[List[float]]): Parameter of the Beta distribution.
                If None, use self.b.
            r (Optional[List[float]]): Parameter of the Gamma
                distribution. If None, use self.r.
            alpha (Optional[List[float]]): Parameter of the Gamma
                distribution. If None, use self.alpha.
            time_interval (Optional[float]): The time interval for the expected number of transactions.
                These determine the integration bounds. Must be finite.
            asof_time (Optional[float] = None): The time from which the predictions should be done. Default is None.
            num_transactions (Optional[List[float]]): The number of transactions observed per customer.
            last_transaction (Optional[List[float]]): The date (as float) of the last transaction per customer.

        Returns:
            List[float]: Expected number of transactions per customer.
        """
        assert (
            (a is None)
            == (b is None)
            == (r is None)
            == (alpha is None)
            == (time_interval is None)
            == (asof_time is None)
            == (num_transactions is None)
            == (last_transaction is None)
        )

        if a is None:
            modes = [v["btyd_mode"][0] for v in self.cache.values()]
            a_list = [self.a[m] for m in modes]
            b_list = [self.b[m] for m in modes]
            r_list = [self.r[m] for m in modes]
            alpha_list = [self.alpha[m] for m in modes]
            t = self.time_interval
            num_transactions = [len(v["t"]) for v in self.cache.values()]
            last_transaction = [self.timestamp2float(v["t"][-1]) for v in self.cache.values()]
            T = self.timestamp2float(self.asof_time)

        else:
            assert len(a) == len(b) == len(r) == len(alpha) == len(num_transactions) == len(last_transaction)
            a_list, b_list = a, b
            r_list, alpha_list = r, alpha
            t = time_interval
            T = asof_time

        e_per_customer = []
        eps = 0.00001
        for a, b, r, alpha, x, tx in zip(a_list, b_list, r_list, alpha_list, num_transactions, last_transaction):
            z1 = (a + b + x - 1) / (a - 1 + eps)
            z2 = ((alpha + T) / (alpha + T + t)) ** (r + x)
            z3 = hyp2f1(r + x, b + x, a + b + x - 1, t / (alpha + T + t + eps))  # noqa: E226
            z4 = 1 - z2 * z3
            z5 = z1 * z4
            z6 = a / (b + x - 1 + eps)  # noqa: E226
            z7 = ((alpha + T) / (alpha + tx + eps)) ** (r + x)  # noqa: E226
            z8 = 1 + z6 * z7 if x > 0 else 1
            e_per_customer.append(z5 / z8)
        return e_per_customer

    def _not_tracked_warning(self, var) -> None:
        raise NotTrackedError(f"`{var}` not tracked. Set `track_statistics=True`")

    @property
    def truncated_sequences(self) -> float:
        """float: returns the ratio of simulated sequences that were truncated due to the end date."""
        if self.track_statistics:
            return self._num_truncated_sequences / self._num_generated_sequences
        else:
            self._not_tracked_warning("truncated_sequences")
            return -1.0

    @property
    def num_generated_sequences(self) -> int:
        """The number of sequences generated so far.

        Returns:
            int: number of sequences generated so far.
        """
        if self.track_statistics:
            return self._num_generated_sequences
        else:
            self._not_tracked_warning("num_generated_sequences")
            return -1

    @property
    def start_dates_from_start_date(self) -> List[float]:
        if self.track_statistics:
            start_offsets_start_dates = [x - self.start_date for x in self._start_dates]
            start_offsets_start_dates = [x.days + x.seconds / (60 * 60 * 24) for x in start_offsets_start_dates]
            return list(start_offsets_start_dates)
        else:
            self._not_tracked_warning("start_dates_from_start_date")
            return [-1.0]

    @property
    def end_dates_from_end_date(self) -> List[float]:
        if self.track_statistics:
            end_offsets_end_dates = [x - self.end_date for x in self._end_dates]
            end_offsets_end_dates = [x.days + x.seconds / (60 * 60 * 24) for x in end_offsets_end_dates]
            return list(end_offsets_end_dates)
        else:
            self._not_tracked_warning("end_dates_from_end_date")
            return [-1.0]

    @property
    def all_dates_from_start_date(self) -> List[float]:
        if self.track_statistics:
            start_offsets = [x - self.start_date for x in self._all_transactions]
            start_offsets = [x.days + x.seconds / (60 * 60 * 24) for x in start_offsets]
            return list(start_offsets)
        else:
            self._not_tracked_warning("all_dates_from_start_date")
            return [-1.0]

    @property
    def range_transaction_days(self) -> List[float]:
        if self.track_statistics:
            return list(self._range_transaction_days)
        else:
            self._not_tracked_warning("range_transaction_days")
            return [-1.0]

    @property
    def num_events_per_customer(self) -> List[float]:
        if self.track_statistics:
            return list(self._num_events_per_customer)
        else:
            self._not_tracked_warning("num_events_per_customer")
            return [-1.0]

    def plot_tracked_statistics(self, nbins: int = 100) -> Figure:
        """Plot all tracked statistics about the BTYD dataset.

        For example, use:
        ```python
        dataset = BTYD(..., track_statistics=True) # create dataset
        dataset[0:1000] # generate 1000 examples by requesting them
        dataset.plot_tracked_statistics().show() # call show method on plotly.go.Figure
        ```

        Args:
            nbins (int): The number of bins for the histograms
        Returns:
            Figure: Plotly figure.
        """
        # PLOTLY is slow to import, thus only importing it once it is needed
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        if self.num_generated_sequences == 0:
            raise IndexError("No data generated yet. Consider running `BTYD()[:]` to create samples.")

        def d2s(obj: datetime.datetime) -> str:
            return obj.strftime("%Y-%m-%d")

        subtitles = [
            "Time span between first and last event per customer",
            "Number of events per customer",
            "Dates of first events per customer",
            "Dates of last events per customer",
            "Dates of all events per customer",
            "Statistics",
        ]
        fig = make_subplots(rows=3, cols=2, start_cell="top-left", subplot_titles=subtitles)

        statistics = [
            f"Number of generated sequences: {self.num_generated_sequences}",
            f"Truncated Sequences: {100*self.truncated_sequences:.2f}%",
            f"Earliest start date: {d2s(self.start_date)}",
            f"Last start date: {d2s(self.start_limit_date)}",
            f"End date: {d2s(self.end_date)}",
            f"Time between earliest and last start date: {(self.start_limit_date-self.start_date).days} days",
            f"Time between earliest start and end date: {(self.end_date-self.start_date).days} days",
        ]

        fig.update_layout(title_text="Buy Till You Die - Dataset Statistics", title_x=0.5)

        fig.add_trace(
            go.Histogram(x=self.range_transaction_days, nbinsx=nbins, name="Range"),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Histogram(x=self.num_events_per_customer, nbinsx=nbins, name="Events"),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Histogram(x=self.start_dates_from_start_date, nbinsx=nbins, name="Start Dates"),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Histogram(x=self.end_dates_from_end_date, nbinsx=nbins, name="End Dates"),
            row=2,
            col=2,
        )

        fig.add_trace(
            go.Histogram(x=self.all_dates_from_start_date, nbinsx=nbins, name="All Dates"),
            row=3,
            col=1,
        )

        fig.add_trace(go.Scatter(name="Statistics"), row=3, col=2)

        def annotate(fig, text, y):
            fig.add_annotation(
                xref="x domain",
                yref="y domain",
                x=0.01,
                y=y,
                text=text,
                showarrow=False,
                row=3,
                col=2,
            )

        [annotate(fig, text, 1 - 0.15 * i) for i, text in enumerate(statistics)]

        fig.update_xaxes(title_text="Days", showgrid=True, row=1, col=1)
        fig.update_xaxes(title_text="# Events", showgrid=True, row=1, col=2)
        fig.update_xaxes(
            title_text=f"Days since earliest start date: {d2s(self.start_date)}",
            showgrid=True,
            row=2,
            col=1,
        )
        fig.update_xaxes(
            title_text=f"Days relative to the end date: {d2s(self.end_date)}",
            showgrid=True,
            row=2,
            col=2,
        )
        fig.update_xaxes(
            title_text=f"Days since earliest start date: {d2s(self.start_date)}",
            showgrid=True,
            row=3,
            col=1,
        )
        fig.update_xaxes(showgrid=False, range=[1, 2], row=3, col=2)
        fig.update_yaxes(showgrid=False, range=[1, 2], row=3, col=2)

        fig.update_layout(showlegend=False)

        return fig


PRESET_MODES = {
    "credit_cards": GenMode(a=1, b=3, r=5, alpha=15),  # TODO: write discrete features
    "forex": GenMode(a=5, b=2, r=5, alpha=15)
    # + Other Preset Modes that we write up
}
