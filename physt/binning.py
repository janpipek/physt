import numpy as np


def numpy_like(data=None, bins=10, range=None, **kwargs):
    """Create same bins as numpy.histogram would do.

    Parameters
    ----------
    data: array_like, optional
        This is optional if both bins and range are set
    bins: int or array_like
    range: tuple
        minimum and maximum

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


def exponential(data=None, bins=None, range=None, **kwargs):
    if bins is None:
        bins = ideal_bin_count(data)
    if range is None:
        range = (np.log10(data.min()), np.log10(data.max()))
    return np.logspace(range[0], range[1], bins+1)


def quantile(data, bins=None, qrange=(0.0, 1.0)):
    """Binning scheme based on quantile ranges."""
    if bins is None:
        bins = ideal_bin_count(data)
    return np.percentile(data, np.linspace(qrange[0] * 100, qrange[1] * 100, bins + 1))


def ideal_bin_count(data, method="default"):
    """A theoretically ideal bin count.

    Returns
    -------
    bincount: int
    """
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


def fixed_width(data, bin_width, align=True):
    if bin_width <= 0:
        raise RuntimeError("Bin width must be > 0.")
    if align == True:
        align = bin_width
    min = data.min()
    if align:
        min = (min // align) * align
    bincount = np.ceil((data.max() - min) / bin_width).astype(int)
    return np.arange(bincount + 1) * bin_width + min


binning_methods = {
    "numpy_like" : numpy_like,
    "exponential" : exponential,
    "quantile": quantile,
    "fixed_width": fixed_width,
}

bincount_methods = ["default", "sturges", "rice"]