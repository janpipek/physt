"""Different binning algorithms/schemas for the histograms."""

import numpy as np
from .bin_utils import make_bin_array


def calculate_bins(array, _=None, *args, **kwargs):
    """Find optimal binning from arguments.

    Parameters
    ----------
    array: arraylike
    _: int or str or Callable or arraylike or Iterable

    Returns
    -------
    numpy.ndarray
    """
    if _ is None:
        bin_count = kwargs.pop("bins", ideal_bin_count(data=array))
        bins = numpy_bins(array, bin_count, *args, **kwargs)
    elif isinstance(_, int):
        bins = numpy_bins(array, _, *args, **kwargs)
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


def numpy_bins(data=None, bins=10, range=None, **kwargs):
    """Binning schema working as numpy.histogram.

    Parameters
    ----------
    data: array_like, optional
        This is optional if both bins and range are set
    bins: int or array_like
    range: tuple
        minimum and maximum

    Returns
    -------
    numpy.ndarray

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


def exponential_bins(data=None, bins=None, range=None, **kwargs):
    """Binning schema with exponentially distributed bins.

    Parameters
    ----------
    bins: Optional[int]
        Number of bins
    range: Optional[tuple]
        The log10 of first left edge and last right edge.

    Returns
    -------
    numpy.ndarray

    See also
    --------
    numpy.logspace
    """
    if bins is None:
        bins = ideal_bin_count(data)
    if range is None:
        range = (np.log10(data.min()), np.log10(data.max()))
    return np.logspace(range[0], range[1], bins+1)


def quantile_bins(data, bins=None, qrange=(0.0, 1.0)):
    """Binning schema based on quantile ranges.

    This binning finds equally spaced quantiles. This should lead to
    all bins having roughly the same frequencies.

    Note: weights are not (yet) take into account for calculating
    quantiles.

    Parameters
    ----------
    bins: Optional[int]
        Number of bins
    qrange: Optional[tuple]
        Two floats as minimum and maximum quantile (default: 0.0, 1.0)

    Returns
    -------
    numpy.ndarray
    """
    if bins is None:
        bins = ideal_bin_count(data)
    return np.percentile(data, np.linspace(qrange[0] * 100, qrange[1] * 100, bins + 1))


def fixed_width_bins(data, bin_width, align=True):
    """Binning schema with predefined bin width.

    Parameters
    ----------
    bin_width: float
    align: bool or float
        Align to a "friendly" initial value. If Ture

    Returns
    -------
    numpy.ndarray
    """
    if bin_width <= 0:
        raise RuntimeError("Bin width must be > 0.")
    if align == True:
        align = bin_width
    min = data.min()
    max = data.max()
    if align:
        min = (min // align) * align
        max = np.ceil(max / align) * align
    bincount = np.round((max - min) / bin_width).astype(int)   # (max - min) should be int or very close to it
    return np.arange(bincount + 1) * bin_width + min


binning_methods = {
    "numpy_like" : numpy_bins,
    "exponential" : exponential_bins,
    "quantile": quantile_bins,
    "fixed_width": fixed_width_bins,
}

try:
    from astropy.stats.histogram import histogram, knuth_bin_width, freedman_bin_width, scott_bin_width, bayesian_blocks
    import warnings
    warnings.filterwarnings("ignore", module="astropy\..*")

    def astropy_bayesian_blocks(data, range=None, **kwargs):
        """Binning schema based on Bayesian blocks (from astropy).

        Computationally expensive for large data sets.

        Parameters
        ----------
        data: arraylike
        range: Optional[tuple]

        Returns
        -------
        numpy.ndarray

        See also
        --------
        astropy.stats.histogram.bayesian_blocks
        astropy.stats.histogram.histogram
        """
        if range is not None:
            data = data[(data >= range[0]) & (data <= range[1])]
        edges = bayesian_blocks(data)
        return edges

    def astropy_knuth(data, range=None, **kwargs):
        """Binning schema based on Knuth's rule (from astropy).

        Computationally expensive for large data sets.

        Parameters
        ----------
        data: arraylike
        range: Optional[tuple]

        Returns
        -------
        numpy.ndarray

        See also
        --------
        astropy.stats.histogram.knuth_bin_width
        astropy.stats.histogram.histogram
        """
        if range is not None:
            data = data[(data >= range[0]) & (data <= range[1])]
        _, edges = knuth_bin_width(data, True)
        return edges

    def astropy_scott(data, range=None, **kwargs):
        """Binning schema based on Scott's rule (from astropy).

        Parameters
        ----------
        data: arraylike
        range: Optional[tuple]

        Returns
        -------
        numpy.ndarray

        See also
        --------
        astropy.stats.histogram.scott_bin_width
        astropy.stats.histogram.histogram
        """
        if range is not None:
            data = data[(data >= range[0]) & (data <= range[1])]
        _, edges = scott_bin_width(data, True)
        return edges

    def astropy_freedman(data, range=None, **kwargs):
        """Binning schema based on Freedman-Diaconis rule (from astropy).

        Parameters
        ----------
        data: arraylike
        range: Optional[tuple]

        Returns
        -------
        numpy.ndarray

        See also
        --------
        astropy.stats.histogram.freedman_bin_width
        astropy.stats.histogram.histogram
        """
        if range is not None:
            data = data[(data >= range[0]) & (data <= range[1])]
        _, edges = freedman_bin_width(data, True)
        return edges

    binning_methods["astropy_blocks"] = astropy_bayesian_blocks
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
        See https://en.wikipedia.org/wiki/Histogram for the description

    Returns
    -------
    int
        Number of bins, always >= 1
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