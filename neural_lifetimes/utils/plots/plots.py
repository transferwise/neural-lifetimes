import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np


def plot_transactions(
    dataset,
    year_limit=3,
    title=None,
    log=False,
    start_date=None,
    end_date=None,
    ax=None,
):
    """
    Plots all the transactions.

    Args:
        dataset (List[Dict[str, List[str]]]): List of transactions.
        year_limit (int): Limit the number of years to plot. Default is 3.
        title (str): Title of the plot. If None, uses
            ``f"Last Transaction of each customer (n={len(dataset)})"``.
        log (bool): If True, plots the number of transactions in log scale. Default is
            False.
        start_date (datetime.datetime): If not None, only transactions after this date
            are plotted.
        end_date (datetime.datetime): If not None, only transactions before this date
            are plotted.
        ax (matplotlib.axes.Axes): If not None, plots the transactions on this axis.
            Otherwise, creates a new figure with ``figsize=(16,6)``.

    Returns:
        matplotlib.axes.Axes: The axis with the plot.
    """
    # start_date, end_date = dataset.start_date, dataset.end_date

    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 6))

    total_t = np.array([])
    # max_t = np.array([t['t'].max() for t in dataset]) \\ use this if you want
    #   to plot last transaction
    for t in dataset:
        total_t = np.append(total_t, t["t"])
    mdates.date2num(total_t)

    upper_limit = start_date + year_limit * (end_date - start_date)
    upper_limit = datetime.datetime.combine(upper_limit, datetime.time())
    max_t_trunc = np.array([x for x in total_t if x <= upper_limit])
    t_num_trunc = mdates.date2num(max_t_trunc)

    ax.hist(t_num_trunc, bins=50, color="lightblue", log=log)

    # plt.locator_params(axis="x", nbins=10)
    ax.locator_params(axis="x", nbins=10)

    if title is None:
        ax.set_title(f"Number of transactions (n={len(total_t)})")  # Last transaction of every customer
    else:
        ax.set_title(title)
    if log:
        ax.set_ylabel("# Transactions (Log Scale)")  # Customers (if using last_t)
    else:
        ax.set_ylabel("# Transactions")  # Customers (if using last_t)
    ax.axvline(x=start_date, color="green")
    ax.axvline(x=end_date, color="red")

    # plt.xlim(start_date - datetime.timedelta(days=1), upper_limit)
    ax.set_xlim(start_date - datetime.timedelta(days=1), upper_limit)

    return ax


def plot_timeline(dates, names, title="Customer events", split=None, ax=None):
    """
    Display a timeline of events for a customer.

    Args:
        dates (List[datetime.datetime]): List of dates.
        names (List[str]): List of names.
        title (str): Title of the plot. Default is ``"Customer events"``.
        split (datetime.datetime): If not ``None``, splits the plot by adding vertical
            lines.
        ax (matplotlib.axes.Axes): If not ``None``, plots the transactions on this axis.
            Otherwise, creates a new figure with ``figsize=(16,6)``.

    Returns:
        matplotlib.axes.Axes: The axis with the plot.
    """
    # Choose some nice levels
    levels = np.tile([-5, 5, -3, 3, -1, 1], int(np.ceil(len(dates) / 6)))[: len(dates)]

    # Create figure and plot a stem plot with the date
    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 6), constrained_layout=True)
    ax.set(title=title)

    ax.vlines(dates, 0, levels, color="tab:red")  # The vertical stems.
    ax.plot(dates, np.zeros_like(dates), "-o", color="k", markerfacecolor="w")  # Baseline and markers on it.

    # annotate lines
    for d, l, r in zip(dates, levels, names):
        ax.annotate(
            r,
            xy=(d, l),
            xytext=(-3, np.sign(l) * 3),
            textcoords="offset points",
            horizontalalignment="right",
            verticalalignment="bottom" if l > 0 else "top",
        )

    # format xaxis with 1 month intervals
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    # ax.get_xtickslabels(rotation=30, ha="right")

    # remove y axis and spines
    ax.yaxis.set_visible(False)
    ax.spines[["left", "top", "right"]].set_visible(False)

    if split is not None:
        # plt.axvline(split)
        ax.axvline(split)

    ax.margins(y=0.1)

    return ax


def _add_to_timeline(dates, names, ax, color):
    levels = np.tile([-5, 5, -3, 3, -1, 1], int(np.ceil(len(dates) / 6)))[: len(dates)]

    ax.vlines(dates, 0, levels, color=f"tab:{color}")  # The vertical stems.
    ax.plot(dates, np.zeros_like(dates), "-o", color="k", markerfacecolor="w")  # Baseline and markers on it.

    # annotate lines
    for d, l, r in zip(dates, levels, names):
        ax.annotate(
            r,
            xy=(d, l),
            xytext=(-3, np.sign(l) * 3),
            textcoords="offset points",
            horizontalalignment="right",
            verticalalignment="bottom" if l > 0 else "top",
        )
    return ax


def plot_freq_transactions(dataset, title=None, log=False, ax=None):
    """
    Plots the frequency of transactions of each customer.

    Args:
        dataset (List[Dict[str, List[str]]]): List of transactions.
        title (str): Title of the plot. If ``None`` (default), uses
            ``f"Average # days between transaction by # customers (n={len(dataset)})"``.
        log (bool): If ``True``, plots the number of transactions in log scale. Default
            is ``False``.
        ax (matplotlib.axes.Axes): If not ``None``, plots the transactions on this axis.
            Otherwise, creates a new figure with ``figsize=(16,6)``.

    Returns:
        matplotlib.axes.Axes: The axis with the plot.
    """
    dts = np.array([(t["t"][1:] - t["t"][:-1]).mean().days if len(t["t"]) > 1 else -1 for t in dataset])

    mean_dts = dts[dts != -1]
    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 6))

    ax.hist(mean_dts, color="blue", bins=100, log=log)

    # plt.locator_params(axis="x", nbins=10)
    ax.locator_params(axis="x", nbins=10)

    if title is None:
        ax.set_title(f"Average # days between transaction by # customers (n={len(dataset)})")
    else:
        ax.set_title(title)
    if log:
        ax.set_ylabel("# Customers (Log Scale)")
    else:
        ax.set_ylabel("# Customers")

    return ax


def plot_num_transactions(df, title=None, log=False, ax=None):
    """
    Plots the number of transactions of each customer.

    Args:
        df (pd.DataFrame): DataFrame with columns ``transactions`` recording the number
            of transactions.
        title (str): Title of the plot. If ``None`` (default), uses
            ``f"Average # transactions by # customers (n={len(df)})"``.
        log (bool): If ``True``, plots the number of transactions in log scale. Default
            is ``False``.
        ax (matplotlib.axes.Axes): If not ``None``, plots the transactions on this axis.
            Otherwise, creates a new figure with ``figsize=(16,6)``.

    Returns:
        matplotlib.axes.Axes: The axis with the plot.
    """
    counts = np.array(df["transactions"])

    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 6))

    ax.hist(counts, color="pink", bins=100, log=log)

    # plt.locator_params(axis="x", nbins=10)
    ax.locator_params(axis="x", nbins=10)

    if title is None:
        ax.set_title(f"Number of transactions by # customers (n={len(df.transactions)})")
    else:
        ax.set_title(title)
    if log:
        ax.set_ylabel("# Customers (Log Scale)")
    else:
        ax.set_ylabel("# Customers")

    return ax


def plot_cont_feature(dataset, feature, title=None, log=False, ax=None):
    """
    Plots the distribution of a continuous feature.

    Args:
        dataset (List[Dict[str, List[str]]]): List of transactions.
        feature (str): Name of the feature to plot.
        title (str): Title of the plot. If ``None`` (default), uses
            ``f"Feature {feature} by # events (n={len(dataset)})"``.
        log (bool): If ``True``, plots the number of transactions in log scale. Default
            is ``False``.
        ax (matplotlib.axes.Axes): If not ``None``, plots the transactions on this axis.
            Otherwise, creates a new figure with ``figsize=(16,6)``.

    Returns:
        matplotlib.axes.Axes: The axis with the plot.
    """
    # counts = np.array([df[feature] for df in dataset])
    # counts = counts.ravel()
    counts = np.array([])
    for df in dataset:
        counts = np.append(counts, df[feature])

    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 6))

    ax.hist(counts, bins=100, log=log)

    # plt.locator_params(axis="x", nbins=10)
    ax.locator_params(axis="x", nbins=10)

    if title is None:
        ax.set_title(f"Feature {feature} by # events (n={len(dataset)})")
    else:
        ax.set_title(title)
    if log:
        ax.set_ylabel("# Events (Log Scale)")
    else:
        ax.set_ylabel("# Events")

    return ax


def plot_cont_features_pd(dataset, feature, title=None, ax=None):
    """
    Plots the distribution of a continuous feature from a pandas DataFrame.

    Args:
        dataset (pd.DataFrame): DataFrame with columns ``transactions`` recording the
            number of transactions.
        feature (str): Name of the feature to plot.
        title (str): Title of the plot. If ``None`` (default), uses
            ``f"Feature {feature} by # events (n={len(dataset)})"``.
        ax (matplotlib.axes.Axes): If not ``None``, plots the transactions on this axis.
            Otherwise, creates a new figure with ``figsize=(16,6)``.

    Returns:
        matplotlib.axes.Axes: The axis with the plot.
    """
    counts = np.array(dataset[feature])

    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 6))

    ax.hist(counts, bins=100)

    # plt.locator_params(axis="x", nbins=10)
    ax.locator_params(axis="x", nbins=10)

    if title is None:
        ax.set_title(f"Feature {feature} by # events (n={len(dataset)})")
    else:
        ax.set_title(title)
    ax.set_ylabel("# Events")

    return ax


def plot_discr_feature(df, feature, feature_names=None, title=None, log=False, ax=None):
    """
    Plots the distribution of a discrete feature.

    Args:
        df (pd.DataFrame): DataFrame with the feature to plot.
        feature (str): Name of the feature to plot.
        feature_names (List[str]): Names of the categories of the feature. If ``None``
            (default), uses ``df[feature].unique()``.
        title (str): Title of the plot. If ``None`` (default), uses
            ``f"Feature {feature} by # events (n={len(df)})"``.
        log (bool): If ``True``, plots the number of transactions in log scale. Default
            is ``False``.
        ax (matplotlib.axes.Axes): If not ``None``, plots the transactions on this axis.
            Otherwise, creates a new figure with ``figsize=(16,6)``.

    Returns:
        matplotlib.axes.Axes: The axis with the plot.
    """
    unique, counts = np.unique(np.array(df[feature]), return_counts=True)

    if feature_names is None:
        feature_names = list(range(len(unique)))

    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(feature_names, counts)

    if title is None:
        ax.set_title(f"Feature {feature} by # events (n={len(df)})")
    else:
        ax.set_title(title)
    if log:
        ax.set_ylabel("# Events (Log Scale)")
    else:
        ax.set_ylabel("# Events")

    return ax
