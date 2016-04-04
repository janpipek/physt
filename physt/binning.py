import numpy as np
from .bin_utils import make_bin_array


def calculate_bins(array, _=None, *args, **kwargs):
    """Find optimal binning from arguments."""
    if _ is None:
        bin_count = kwargs.pop("bins", ideal_bin_count(data=array))
        bins = numpy_like(array, bin_count, *args, **kwargs)
    elif isinstance(_, int):
        bins = numpy_like(array, _, *args, **kwargs)
    elif isinstance(_, str):
        method = binning_methods[_]
        bins = method(array, *args, **kwargs)
    elif callable(_):
        bins = _(array, *args, **kwargs)
    elif np.iterable(_):
        bins = _
    else:
        raise RuntimeError("Binning {0} not understood.".format(_))
    return make_bin_array(bins)


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

try:
    from astropy.stats.histogram import histogram, knuth_bin_width, freedman_bin_width, scott_bin_width, bayesian_blocks
    import warnings
    warnings.filterwarnings("ignore", module="astropy\..*")

    def astropy_blocks(data, range=None, **kwargs):
        if range is not None:
            data = data[(data >= range[0]) & (data <= range[1])]
        edges = bayesian_blocks(data)
        return edges

    def astropy_knuth(data, range=None, **kwargs):
        if range is not None:
            data = data[(data >= range[0]) & (data <= range[1])]
        _, edges = knuth_bin_width(data, True)
        return edges

    def astropy_scott(data, range=None, **kwargs):
        if range is not None:
            data = data[(data >= range[0]) & (data <= range[1])]
        _, edges = scott_bin_width(data, True)
        return edges

    def astropy_freedman(data, range=None, **kwargs):
        if range is not None:
            data = data[(data >= range[0]) & (data <= range[1])]
        _, edges = freedman_bin_width(data, True)
        return edges

    binning_methods["astropy_blocks"] = astropy_blocks
    binning_methods["astropy_knuth"] = astropy_knuth
    binning_methods["astropy_scott"] = astropy_scott
    binning_methods["astropy_freedman"] = astropy_freedman
except:
    pass     # astropy is not required


def ideal_bin_count(data, method="default"):
    """A theoretically ideal bin count.

    Parameters
    ----------
    data: array_like or None
        Data to work on. Most methods don't use this.
    method: str
        Name of the method to apply, available values:
          - default
          - sqrt
          - sturges
          - doane

    Returns
    -------
    bincount: int
        Number of bins, always >= 1

    See also
    --------
    - https://en.wikipedia.org/wiki/Histogram
    """
    n = data.size
    if n < 1:
        return 1
    if method == "default":
        if n <= 32:
            return 7
        else:
            return ideal_bin_count(data, "sturges")
    elif method == "sqrt":
        return np.ceil(np.sqrt(n))
    elif method == "sturges":
        return np.ceil(np.log2(n)) + 1
    elif method == "doane":
        if n < 3:
            return 1
        from scipy.stats import skew
        sigma = np.sqrt(6 * (n-2) / (n + 1) * (n + 3))
        return np.ceil(1 + np.log2(n) + np.log2(1 + np.abs(skew(data)) / sigma))
    elif method == "rice":
        return np.ceil(2 * np.power(n, 1 / 3))


bincount_methods = ["default", "sturges", "rice", "sqrt", "doane"]