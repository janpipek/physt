"""Different binning algorithms/schemas for the histograms."""

import numpy as np
from .bin_utils import make_bin_array, is_consecutive, to_numpy_bins, is_rising


class BinningBase(object):
    """Abstract base class for binning.

    Inheriting
    ----------
    - define at least one of the following properties: bins, numpy_bins (cached intercomputation exists)
    - if you modify the bins, be sure to put _bins and _numpy_bins into proper state (in some cases, None is sufficient)
    - the construct method should prepare bins etc.
    - checking of proper bins should be done in __init__
    """
    def __init__(self, bins=None, numpy_bins=None, includes_right_edge=False, integrity_check=True, adaptive=False):
        # TODO: Incorporate integrity_check parameter
        self._consecutive = None
        if bins is not None:
            if numpy_bins is not None:
                raise RuntimeError("Cannot specify numpy_bins and bins at the same time.")
            bins = make_bin_array(bins)
            if not is_rising(bins):
                raise RuntimeError("Bins must be in rising order.")
            # TODO: Test for consecutiveness?
        elif numpy_bins is not None:
            numpy_bins = to_numpy_bins(numpy_bins)
            if not np.all(numpy_bins[1:] > numpy_bins[:-1]):
                raise RuntimeError("Bins must be in rising order.")
            self._consecutive = True
        self._bins = bins
        self._numpy_bins = numpy_bins
        self._includes_right_edge = includes_right_edge
        if adaptive and not self.adaptive_allowed:
            raise RuntimeError("Adaptivity not allowed for {0}".format(self.__class__.__name__))
        if adaptive and includes_right_edge:
            raise RuntimeError("Adaptivity does not work together with right-edge inclusion.")
        self._adaptive = adaptive

    adaptive_allowed = False
    inconsecutive_allowed = False
    # TODO: adding allowed?

    @property
    def includes_right_edge(self):
        return self._includes_right_edge

    def is_consecutive(self, rtol=1.e-5, atol=1.e-8):
        if self.inconsecutive_allowed:
            if self._consecutive is None:
                if self._numpy_bins is not None:
                    self._consecutive = True
                self._consecutive = is_consecutive(self.bins, rtol, atol)
            return self._consecutive
        else:
            return True

    def is_adaptive(self):
        return self._adaptive

    @property
    def bins(self):
        if self._bins is None:
            self._bins = make_bin_array(self.numpy_bins)
        return self._bins

    @property
    def bin_count(self):
        return self.bins.shape[0]

    @property
    def numpy_bins(self):
        if self._numpy_bins is None:
            self._numpy_bins = to_numpy_bins(self.bins)
        return self._numpy_bins

    @classmethod
    def construct(cls, data, *args, **kwargs):
        raise NotImplementedError()

    def as_static(self):
        """Convert binning to a static form.

        Returns
        -------
        StaticBinning
            A new static binning with a copy of bins.
        """
        return StaticBinning(self.bins.copy(), self.includes_right_edge)

    # TODO: implement as_adaptive()?


class StaticBinning(BinningBase):
    inconsecutive_allowed = True

    @classmethod
    def construct(cls, data=None, bins=None, **kwargs):
        return cls(bins=bins, **kwargs)


class NumpyBinning(BinningBase):
    """Binning schema working as numpy.histogram.
    """
    def __init__(self, numpy_bins, includes_right_edge=True):
        # Check: rising
        super(NumpyBinning, self).__init__(numpy_bins=numpy_bins, includes_right_edge=includes_right_edge)

    @classmethod
    def construct(cls, data, bins=10, range=None, *args, **kwargs):
        """
        Parameters
        ----------
        data: array_like, optional
            This is optional if both bins and range are set
        bins: int or array_like
        range: Optional[tuple]
            (min, max)
        includes_right_edge: Optional[bool]
            default: True

        Returns
        -------
        numpy.ndarray

        See Also
        --------
        numpy.histogram
        """
        if isinstance(bins, int):
            if range:
                bins = np.linspace(range[0], range[1], bins + 1)
            else:
                start = data.min()
                stop = data.max()
                bins = np.linspace(start, stop, bins + 1)
        elif np.iterable(bins):
            bins = np.asarray(bins)
        else:
            # Some numpy edge case
            _, bins = np.histogram(data, bins, **kwargs)
        return cls(bins)

    @property
    def numpy_bins(self):
        return self._numpy_bins


class FixedWidthBinning(BinningBase):
    """Binning schema with predefined bin width."""
    adaptive_allowed = True

    def __init__(self, bin_width, bin_count=0, min=-np.nan, includes_right_edge=False, adaptive=False):
        super(FixedWidthBinning, self).__init__(adaptive=adaptive, includes_right_edge=includes_right_edge)

        if bin_width <= 0:
            raise RuntimeError("Bin width must be > 0.")
        self._bin_width = float(bin_width)

        if bin_count < 0:
            raise RuntimeError("Bin count must be >= 0.")
        self._bin_count = int(bin_count)
        self._min = min
        self._bins = None
        self._numpy_bins = None
        # TODO: We need to introduce shift, if we want to support adaptivity for IntegerBinning

    @classmethod
    def construct(cls, data=None, bin_width=1, range=None, includes_right_edge=False, **kwargs):
        """
        Parameters
        ----------
        bin_width: float
        range: Optional[tuple]
            (min, max)

        Returns
        -------
        numpy.ndarray
        """
        align = bin_width    # TODO: make this available...

        # Get the range
        if range:
            val_min, val_max = range
        elif data is not None:
            val_min, val_max = data.min(), data.max()
        else:
            val_min, val_max = None, None
            min = -np.nan
            bin_count = 0

        if val_min is not None:
            min = np.floor(val_min / align) * align
            max = np.ceil(val_max / align) * align
            if max == val_max and not includes_right_edge:
                max += align
            bin_count = np.round((max - min) / bin_width).astype(int)  # (max - min) should be int or very close to it

        return cls(bin_width=bin_width, bin_count=bin_count, min=min, includes_right_edge=includes_right_edge, **kwargs)

    @property
    def numpy_bins(self):
        if self._numpy_bins is None:
            if self._bin_count == 0:
                return np.zeros((0, 2), dtype=float)
            self._numpy_bins = self._min + self._bin_width * np.arange(self._bin_count + 1)
        return self._numpy_bins


class ExponentialBinning(BinningBase):
    """Binning schema with exponentially distributed bins."""
    adaptive_allowed = False

    # TODO: Implement adaptivity

    def __init__(self, log_min, log_width, bin_count, includes_right_edge=False):
        adaptive = False
        super(ExponentialBinning, self).__init__(includes_right_edge=includes_right_edge, adaptive=adaptive)
        self._log_min = log_min
        self._log_width = log_width
        self._bin_count = bin_count

    @property
    def numpy_bins(self):
        if self._bin_count == 0:
            return np.ndarray((0), dtype=float)
        if self._numpy_bins is None:
            log_bins = self._log_min + np.arange(self._bin_count + 1) * self._log_width
            self._numpy_bins = 10 ** log_bins
        return self._numpy_bins

    @classmethod
    def construct(cls, data=None, bins=None, range=None, **kwargs):
        """
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
        log_width = (range[1] - range[0]) / bins
        return cls(log_min=range[0], log_width=log_width, bin_count=bins, **kwargs)


class HumanBinning(FixedWidthBinning):
    """Binning schema with bins automatically optimized to human-friendly widths.

    Typical widths are: 1.0, 25,0, 0.02, 500, 2.5e-7, ...
    """
    def __init__(self, *args, **kwargs):
        raise RuntimeError("HumanBinning does not allow instances.")

    subscales = np.array([0.5, 1, 2, 2.5, 5, 10])

    @classmethod
    def construct(cls, data=None, bins=None, range=None, **kwargs):
        """
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
        if bins is None:
            bins = ideal_bin_count(data)
        # TODO data or range check
        min_ = range[0] if range else data.min()
        max_ = range[1] if range else data.max()
        bw = (max_ - min_) / bins

        power = np.floor(np.log10(bw)).astype(int)
        best_index = np.argmin(np.abs(np.log(cls.subscales * (10 ** power) / bw)))
        bin_width = (10 ** power) * cls.subscales[best_index]
        return FixedWidthBinning.construct(bin_width=bin_width, data=data, range=range, **kwargs)


class QuantileBinning(StaticBinning):
    """Binning schema based on quantile ranges.

    This binning finds equally spaced quantiles. This should lead to
    all bins having roughly the same frequencies.

    Note: weights are not (yet) take into account for calculating
    quantiles.
    """

    def __init__(self, *args, **kwargs):
        raise RuntimeError("QuantileBinning does not allow instances.")

    inconsecutive_allowed = False

    @classmethod
    def construct(cls, data=None, bins=10, qrange=(0.0, 1.0), **kwargs):
        """
        Parameters
        ----------
        bins: sequence or Optional[int]
            Number of bins
        qrange: Optional[tuple]
            Two floats as minimum and maximum quantile (default: 0.0, 1.0)

        Returns
        -------
        numpy.ndarray
        """
        # TODO: Accept range in some way?
        if np.isscalar(bins):
            bins = np.linspace(qrange[0] * 100, qrange[1] * 100, bins + 1)
        bins = np.percentile(data, bins)
        return StaticBinning.construct(bins=make_bin_array(bins), includes_right_edge=True)


class IntegerBinning(FixedWidthBinning):
    """Binning schema with bins centered around integers.

    Designed for integer values. Bins are centered around integers
    like [0.5, 1.5)"""

    @classmethod
    def construct(cls, data=None, range=None, **kwargs):
        """
        Parameters
        ----------
        range: Optional[Tuple[int]]
            min (included) and max integer (excluded) bin

         Returns
        -------
        numpy.ndarray
        """
        # TODO: Enable running with empty data
        if range:
            min = int(range[0]) - 0.5
            max = int(range[1]) - 0.5
        elif data is not None:
            min = np.floor(data.min() - 0.5) + 0.5
            max = np.ceil(data.max() + 0.5) - 0.5

        bincount = np.round(max - min)
        # TODO: Include shift parameter
        return FixedWidthBinning(bin_width=1, bin_count=bincount, min=min, includes_right_edge=False, adaptive=kwargs.get("adaptive", False))


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

#
# binning_methods = {
#     "numpy" : numpy_bins,
#     "exponential" : exponential_bins,
#     "quantile": quantile_bins,
#     "fixed_width": fixed_width_bins,
#     "integer": integer_bins,
#     "human" : human_bins
# }

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