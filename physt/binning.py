"""Different binning algorithms/schemas for the histograms."""

import numpy as np
from .bin_utils import make_bin_array


def calculate_bins(array, _=None, *args, **kwargs):
    """Find optimal binning from arguments.

    Parameters
    ----------
    array: arraylike
        Data from which the bins should be decided (sometimes used, sometimes not)
    _: int or str or Callable or arraylike or Iterable
        To-be-guessed parameter that specifies what kind of binning should be done
    check_nan: bool
        Check for the presence of nan's in array? Default: True
    range: tuple
        Limit values to a range. Some of the binning methods also (subsequently)
        use this parameter for the bin shape.

    Returns
    -------
    numpy.ndarray
        A two-dimensional array with pairs of bin edges (not necessarily consecutive).

    """
    if kwargs.pop("check_nan", True):
        if np.any(np.isnan(array)):
            raise RuntimeError("Cannot calculate bins in presence of NaN's.")
    if "range" in kwargs:   # TODO: re-consider the usage of this parameter
        array = array[(array >= kwargs["range"][0]) & (array <= kwargs["range"][1])]
    if _ is None:
        bin_count = kwargs.pop("bins", ideal_bin_count(data=array))
        bins = numpy_bins(array, bin_count, *args, **kwargs)
    elif isinstance(_, int):
        bins = numpy_bins(array, _, *args, **kwargs)
    elif isinstance(_, str):
        # What about the ranges???
        if _ in bincount_methods:
            bin_count = ideal_bin_count(array, method=_)
            bins = numpy_bins(array, bin_count, *args, **kwargs)
        elif _ in binning_methods:
            method = binning_methods[_]
            bins = method(array, *args, **kwargs)
        else:
            raise RuntimeError("No binning method {0} available.".format(_))
    elif callable(_):
        bins = _(array, *args, **kwargs)
    elif np.iterable(_):
        bins = _
    else:
        raise RuntimeError("Binning {0} not understood.".format(_))
    return make_bin_array(bins)


def calculate_bins_nd(array, bins=None, *args, **kwargs):
    """Find optimal binning from arguments (n-dimensional variant)

    Usage similar to `calculate_bins`.
    """
    if kwargs.pop("check_nan", True):
        if np.any(np.isnan(array)):
            raise RuntimeError("Cannot calculate bins in presence of NaN's.")
    n, dim = array.shape

    # Prepare bins
    if isinstance(bins, (list, tuple)):
        if len(bins) != dim:
            raise RuntimeError("List of bins not understood.")
    else:
        bins = [bins] * dim

    # Prepare arguments
    args = list(args)
    range_ = kwargs.pop("range", None)
    if range_:
        range_n = np.asarray(range_)
        if range_n.shape == (2,):
            range_ = dim * [range_]
        elif range_n.shape == (dim, 2):
            range_ = range_
        else:
            raise RuntimeError("1 or d tuples expected for the range")
    for i in range(len(args)):
        if isinstance(args[i], (list, tuple)):
            if len(args[i]) != dim:
                raise RuntimeError("Argument not understood.")
        else:
            args[i] = dim * [args[i]]
    for key in list(kwargs.keys()):
        if isinstance(kwargs[key], (list, tuple)):
            if len(kwargs[key]) != dim:
                raise RuntimeError("Argument not understood.")
        else:
            kwargs[key] = dim * [kwargs[key]]

    if range_:
        kwargs["range"] = range_

    bins = [
        calculate_bins(array[:, i], bins[i],
                       *(arg[i] for arg in args),
                       **{k : kwarg[i] for k, kwarg in kwargs.items()})
        for i in range(dim)
        ]
    return bins


def numpy_bins(data=None, bins=10, range=None, **kwargs):
    """Binning schema working as numpy.histogram.

    Parameters
    ----------
    data: array_like, optional
        This is optional if both bins and range are set
    bins: int or array_like
    range: Optional[tuple]
        (min, max)

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
        (min, max)

    Returns
    -------
    numpy.ndarray

    See also
    --------
    numpy.logspace - note that our range semantics is different
    """
    if bins is None:
        bins = ideal_bin_count(data)
    if range:
        range = (np.log10(range[0]), np.log10(range[1]))
    else:
        range = (np.log10(data.min()), np.log10(data.max()))
    return np.logspace(range[0], range[1], bins+1)


def quantile_bins(data, bins=None, qrange=(0.0, 1.0), **kwargs):
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
    # TODO: Accept range in some way?
    if bins is None:
        bins = ideal_bin_count(data)
    return np.percentile(data, np.linspace(qrange[0] * 100, qrange[1] * 100, bins + 1))


def fixed_width_bins(data, bin_width, align=True, range=None, **kwargs):
    """Binning schema with predefined bin width.

    Parameters
    ----------
    bin_width: float
    align: bool or float
        The left-most & right-most edge will be aligned to a multiple of this.
        If True, bin_width will be assumed
    range: Optional[tuple]
        (min, max)

    Returns
    -------
    numpy.ndarray
    """
    # TODO: Introduce range
    if bin_width <= 0:
        raise RuntimeError("Bin width must be > 0.")
    if align == True:
        align = bin_width
    min = range[0] if range else data.min()
    max = range[1] if range else data.max()
    if align:
        min = (min // align) * align
        max = np.ceil(max / align) * align
    bincount = np.round((max - min) / bin_width).astype(int)   # (max - min) should be int or very close to it
    return np.arange(bincount + 1) * bin_width + min


def integer_bins(data, range=None, **kwargs):
    """Binning schema with bins centered around integers.

    Designed for integer values. Bins are centered around integers
    like [0.5, 1.5)

    Parameters
    ----------
    range: Optional[Tuple[int]]
        min (included) and max integer (excluded) bin

     Returns
    -------
    numpy.ndarray
    """
    if range:
        min = int(range[0]) - 0.5
        max = int(range[1]) - 0.5
    else:
        min = np.floor(data.min() - 0.5) + 0.5
        max = np.ceil(data.max() + 0.5) - 0.5
    bincount = np.round(max - min)
    return np.arange(bincount + 1) + min


def human_bins(data, bins=10, range=None, **kwargs):
    """Binning schema with bins automatically optimized human-friendly widths.
    
    Parameters
    ----------
    bins: Optional[int]
        Number of bins
    range: Optional[tuple]
        (min, max)

    Returns
    -------
    numpy.ndarray        
    """
    min_ = range[0] if range else data.min()
    max_ = range[1] if range else data.max()
    bw = (max_ - min_) / bins
    subscales = np.array([0.5, 1, 2, 2.5, 5, 10])
    power = np.floor(np.log10(bw)).astype(int)
    best_index = np.argmin(np.abs(np.log(subscales * (10 ** power) / bw)))
    width = (10 ** power) * subscales[best_index]
    return fixed_width_bins(data, width, range=range)
    

binning_methods = {
    "numpy" : numpy_bins,
    "exponential" : exponential_bins,
    "quantile": quantile_bins,
    "fixed_width": fixed_width_bins,
    "integer": integer_bins,
    "human" : human_bins
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