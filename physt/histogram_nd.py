from __future__ import absolute_import
from . import bin_utils
from .histogram_base import HistogramBase
import numpy as np


class HistogramND(HistogramBase):
    """Multi-dimensional histogram data.

    Attributes
    ----------

    """

    def __init__(self, dimension, binnings, frequencies=None, **kwargs):
        # Bins + checks
        if len(binnings) != dimension:
            raise RuntimeError("bins must be a sequence of {0} schemas".format(dimension))

        # self.ndim = dimension
        self._binnings = binnings

        # Frequencies + checks
        if frequencies is None:
            self._frequencies = np.zeros(self.shape)
        else:
            frequencies = np.asarray(frequencies)
            if frequencies.shape != self.shape:
                raise RuntimeError("Values must have same dimension as bins.")
            if np.any(frequencies < 0):
                raise RuntimeError("Cannot have negative frequencies.")
            self._frequencies = frequencies

        # Missed values
        self.keep_missed = kwargs.get("keep_missed", True)
        self._missed = np.array([kwargs.get("missed", 0)])

        # Names etc.
        self.name = kwargs.get("name", None)
        self.axis_names = kwargs.get("axis_names", ["axis{0}".format(i) for i in range(self.ndim)])
        if len(self.axis_names) != self.ndim:
            raise RuntimeError("The length of axis names must be equal to histogram dimension.")

        # Errors + checks
        self._errors2 = kwargs.get("errors2")
        if self._errors2 is None:
            self._errors2 = self._frequencies.copy()
        else:
            self._errors2 = np.asarray(self._errors2)
        if self._errors2.shape != self._frequencies.shape:
            raise RuntimeError("Errors must have same dimension as frequencies.")
        if np.any(self._errors2 < 0):
            raise RuntimeError("Cannot have negative squared errors.")

    # Not supported yet
    _stats = None

    @property
    def bins(self):
        """Matrix of bins.

        Returns
        -------
        Iterable[np.ndarray]
            Two-dimensional array of bin edges, shape=(n, 2)
        """
        return [binning.bins for binning in self._binnings]

    @property
    def numpy_bins(self):
        return [binning.numpy_bins for binning in self._binnings]  

    def __getitem__(self, item):
        raise NotImplementedError()

    # Missing: cumulative_frequencies - does it make sense

    def get_bin_widths(self, axis = None):  # -> Base
        if axis is not None:
            return self.get_bin_right_edges(axis) - self.get_bin_left_edges(axis)
        else:
            return np.meshgrid(*[self.get_bin_widths(i) for i in range(self.ndim)], indexing='ij')

    @property
    def bin_sizes(self):
        # TODO: Some kind of caching?
        sizes = self.get_bin_widths(0)
        for i in range(1, self.ndim):
            sizes = np.outer(sizes, self.get_bin_widths(i))
        return sizes

    @property
    def total_size(self):
        """The default size of the bin space."""
        return np.product([self.get_bin_widths(i) for i in range(self.ndim)])

    def get_bin_left_edges(self, axis=None):
        if axis is not None:
            return self.bins[axis][:, 0]
        else:
            return np.meshgrid(*[self.get_bin_left_edges(i) for i in range(self.ndim)], indexing='ij')

    def get_bin_right_edges(self, axis=None):
        if axis is not None:
            return self.bins[axis][:, 1]
        else:
            return np.meshgrid(*[self.get_bin_right_edges(i) for i in range(self.ndim)], indexing='ij')

    def get_bin_centers(self, axis=None):
        if axis is not None:
            return (self.get_bin_right_edges(axis) + self.get_bin_left_edges(axis)) / 2
        else:
            return np.meshgrid(*[self.get_bin_centers(i) for i in range(self.ndim)], indexing='ij')

    def find_bin(self, value, axis=None):  # TODO: Check!
        if axis is not None:
            ixbin = np.searchsorted(self.get_bin_left_edges(axis), value, side="right")
            if ixbin == 0:
                return None
            elif ixbin == self.shape[axis]:
                if value <= self.get_bin_right_edges(axis)[-1]:
                    return ixbin - 1
                else:
                    return None
            elif value < self.get_bin_right_edges(axis)[ixbin - 1]:
                return ixbin - 1
            elif ixbin == self.shape[axis]:
                return None
            else:
                return None
        else:
            ixbin = tuple(self.find_bin(value[i], i) for i in range(self.ndim))
            if None in ixbin:
                return None
            else:
                return ixbin 

    def fill(self, value, weight=1):
        for i, binning in enumerate(self._binnings):
            if binning.is_adaptive():
                #print("adaptive, forcing", value[i])
                bin_map = binning.force_bin_existence(value[i])
                #print("map", bin_map)
                self._reshape_data(binning.bin_count, bin_map, i)

        ixbin = self.find_bin(value)
        if ixbin is None and self.keep_missed:
            self._missed += weight
        else:
            self._frequencies[ixbin] += weight
            self.errors2[ixbin] += weight ** 2
        return ixbin

    def fill_n(self, values, weights=None, dropna=True):
        values = np.asarray(values)
        if dropna:
            values = values[~np.isnan(values)]
        for i, binning in enumerate(self._binnings):
            if binning.is_adaptive():
                map = self._binning.force_bin_existence(values[:,i])
                self._reshape_data(binning.bin_count, map, i)
        frequencies, errors2, underflow, overflow, stats = calculate_frequencies(values, self.ndim, self.bins,
                                                                              weights=weights)
        self._frequencies += frequencies
        self._errors2 += errors2
        # TODO: check that adaptive does not produce under-/over-flows?
        self.underflow += underflow
        self.overflow += overflow
        for key in self._stats:
            self._stats[key] += stats.get(key, 0.0)    

    def copy(self, include_frequencies=True):
        if include_frequencies:
            frequencies = np.copy(self.frequencies)
            missed = self.missed
            errors2 = np.copy(self.errors2)
        else:
            frequencies = None
            missed = 0
            errors2 = None
        return self.__class__(dimension=self.ndim, binnings=[binning.copy() for binning in self._binnings],
                              frequencies=frequencies, errors2=errors2,
                              name=self.name, axis_names=self.axis_names[:],
                              keep_missed=self.keep_missed, missed=missed)

    def projection(self, *axes, **kwargs):
        axes = list(axes)
        for i, ax in enumerate(axes):
            if isinstance(ax, str):
                if not ax in self.axis_names:
                    raise RuntimeError("Invalid axis name for projection: " + ax)
                axes[i] = self.axis_names.index(ax)
        if not axes:
            raise RuntimeError("No axis selected for projection")
        if len(axes) != len(set(axes)):
            raise RuntimeError("Duplicate axes in projection")
        invert = list(range(self.ndim))
        for ax in axes:
            invert.remove(ax)
        axes = tuple(axes)
        invert = tuple(invert)
        frequencies = self.frequencies.sum(axis=invert)
        errors2 = self.errors2.sum(axis=invert)
        name = kwargs.pop("name", self.name)
        axis_names = [name for i, name in enumerate(self.axis_names) if i in axes]
        bins = [bins for i, bins in enumerate(self._binnings) if i in axes]
        if len(axes) == 1:
            from .histogram1d import Histogram1D
            return Histogram1D(binning=bins[0], frequencies=frequencies, errors2=errors2,
                               axis_name=axis_names[0], name=name)
        elif len(axes) == 2:
            return Histogram2D(binnings=bins, frequencies=frequencies, errors2=errors2,
                               axis_names=axis_names, name=name)
        else:
            return HistogramND(dimension=len(axes), binnings=bins, frequencies=frequencies, errors2=errors2,
                               axis_names=axis_names, name=name)

    def __eq__(self, other):
        """Equality comparison

        """
        # TODO: Describe allclose
        # TODO: Think about softer alternatives (like compare method)
        if not isinstance(other, self.__class__):
            return False
        if not self.ndim == other.ndim:
            return False
        for i in range(self.ndim):
            if not np.allclose(other.bins[i], self.bins[i]):
                return False
        if not np.allclose(other.errors2, self.errors2):
            return False
        if not np.allclose(other.frequencies, self.frequencies):
            return False
        if not other.missed == self.missed:
            return False
        if not other.name == self.name:
            return False
        if not other.axis_names == self.axis_names:
            return False
        return True

    # def to_xarray(self):
    #     raise NotImplementedError()

    # @classmethod
    # def from_xarray(cls, arr):
    #     raise  NotImplementedError()

    # def to_json(self, path=None):
    #     raise NotImplementedError()

    # @classmethod
    # def from_json(cls, text=None, path=None):
    #     return NotImplementedError

    def __repr__(self):
        s = "{0}(bins={1}, total={2}".format(
            self.__class__.__name__, self.shape, self.total)
        if self.missed:
            s += ", missed={0}".format(self.missed)
        s += ")"
        return s


class Histogram2D(HistogramND):
    """Specialized 2D variant of the general HistogramND class.

    Its only addition is the plot() method
    """

    def __init__(self, binnings, frequencies=None, **kwargs):
        kwargs.pop("dimension", None)
        super(Histogram2D, self).__init__(dimension=2, binnings=binnings, frequencies=frequencies, **kwargs)


def calculate_frequencies(data, ndim, binnings, weights=None, dtype=None):
    """

    Parameters
    ----------

    dtype: Optional[type]
        Underlying type for the histogram. If weights are specified, default is float. Otherwise int64
    """
    data = np.asarray(data)

    edges_and_mask = [binning.numpy_bins_with_mask for binning in binnings]
    edges = [em[0] for em in edges_and_mask]
    masks = [em[1] for em in edges_and_mask]

    if dtype is None:
        dtype = np.int64 if weights is None else np.float    

    ixgrid = np.ix_(*masks) # Indexer to select parts we want

    if weights is not None:
        import numbers
        if issubclass(dtype, numbers.Integral):
            raise RuntimeError("Histograms with weights cannot have integral dtype")
        weights = np.asarray(weights, dtype=dtype)
        if weights.shape != (data.shape[0],):
            raise RuntimeError("Invalid weights shape.")
        total_weight = weights.sum()
    else:
        total_weight = data.shape[0]

    # TODO: Right edges are not taken into account because they fall into inf bin

    if data.shape[0]:
        frequencies, _ = np.histogramdd(data, edges, weights=weights)
        frequencies = frequencies.astype(dtype)      # Automatically copy
        frequencies = frequencies[ixgrid]
        missing = total_weight - frequencies.sum()
        if weights is not None:
            err_freq, _ = np.histogramdd(data, edges, weights=weights ** 2)
            errors2 = err_freq[ixgrid].astype(dtype) # Automatically copy
        else:
            errors2 = frequencies.copy()        
    else:
        frequencies = None
        missing = 0
        errors2 = None

    return frequencies, errors2, missing

