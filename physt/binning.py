import numpy as np


def numpy_like(data, bins=10, range=None, **kwargs):
    """Create same bins as numpy.histogram would do.

    See Also
    --------
    numpy.histogram
    """
    if isinstance(bins, int):
        if range:
            return np.linspace(range[0], range[1], bins + 1)
        else:
            start = data.min()
            stop = data.max()
            return np.linspace(start, stop, bins + 1)
    elif np.iterable(bins):
        return np.asarray(bins).flatten()
    else:
        # Some numpy edge case
        _, bins = np.histogram(data, bins, **kwargs)
        return bins


def exponential(data, bins=None, range=None, **kwargs):
    if bins is None:
        bins = ideal_bin_count(data)
    if range is None:
        range = (np.log10(data.min()), np.log10(data.max()))
    return np.logspace(range[0], range[1], bins)


def quantile(data, bins=None, min_quantile=0.0, max_quantile=1.0):
    if bins is None:
        bins = ideal_bin_count(data)
    return np.percentile(data, np.linspace(min_quantile * 100, max_quantile * 100, bins + 1))


def ideal_bin_count(data, method="default"):
    n = data.size
    if method == "default":
        if n <= 32:
            return 7
        else:
            return ideal_bin_count(data, "sturges")
    elif method == "sturges":
        return np.ceil(np.log2(n)) + 1
    elif method == "rice":
        return np.ceil(2 * np.power(n, 1 / 3))


methods = {
    "numpy_like" : numpy_like,
    "exponential" : exponential
}

bincount_methods = ["default", "sturges", "rice"]