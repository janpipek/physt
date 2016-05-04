from . import bin_utils
import numpy as np


class HistogramND(object):
    def __init__(self, dimension, bins, frequencies=None, **kwargs):
        self._dimension = dimension

        # Bins + checks
        if len(bins) != self._dimension:
            raise RuntimeError("bins must be a sequence of {0} schemas".format(self._dimension))
        self._bins = [bin_utils.make_bin_array(bins_i) for bins_i in bins]

        # Frequencies + checks
        if frequencies is None:
            self._frequencies = np.zeros(self.shape)
        else:
            frequencies = np.array(frequencies, dtype=float)
            if frequencies.shape != self.shape:
                raise RuntimeError("Values must have same dimension as bins.")
            if np.any(frequencies < 0):
                raise RuntimeError("Cannot have negative frequencies.")
            self._frequencies = frequencies

        # Missed values
        self.keep_missed = kwargs.get("keep_missed", True)
        self.missed = kwargs.get("missed", 0)

        # Names etc.
        self.name = kwargs.get("name", None)
        self.axis_names = kwargs.get("axis_names", ["axis{0}".format(i) for i in range(self._dimension)])
        if len(self.axis_names) != self._dimension:
            raise RuntimeError("The length of axis names must be equal to histogram dimension.")

        # Errors + checks
        self._errors2 = np.asarray(kwargs.get("errors2", self.frequencies.copy()))
        if self._errors2.shape != self._frequencies.shape:
            raise RuntimeError("Errors must have same dimension as frequencies.")
        if np.any(self._errors2 < 0):
            raise RuntimeError("Cannot have negative squared errors.")

    @property
    def ndim(self):
        return self._dimension

    @property
    def shape(self):
        return tuple(bins_i.shape[0] for bins_i in self._bins)

    def __getitem__(self, item):
        raise NotImplementedError()

    @property
    def bins(self):
        return self._bins

    @property
    def numpy_bins(self):
        raise NotImplementedError()

    @property
    def frequencies(self):  # Ok -> Base
        return self._frequencies

    @property
    def densities(self):    # OK -> Base
        return (self._frequencies / self.bin_sizes) / self.total

    # Missing: cumulative_frequencies - does it make sense

    @property
    def errors2(self):      # OK -> Base
        return self._errors2

    def errors(self):       # OK -> Base
        return np.sqrt(self._errors2)

    # Missing: Missed - as attribute

    @property
    def total(self):        # OK -> Base
        return self._frequencies.sum()

    def get_bin_widths(self, axis):  # -> Base
        return self.get_bin_right_edges(axis) - self.get_bin_left_edges(axis)

    @property
    def bin_sizes(self):
        # TODO: Some kind of caching?
        sizes = self.get_bin_widths(0)
        for i in range(1, self._dimension):
            sizes = np.outer(sizes, self.get_bin_widths(i))
        return sizes

    @property
    def total_size(self):
        """The default size of the bin space."""
        return np.product([self.get_bin_widths(i) for i in range(self._dimension)])

    def get_bin_left_edges(self, axis):
        raise NotImplementedError()

    def get_bin_right_edges(self, axis):
        raise NotImplementedError()

    # TODO: bin_centers property?

    def get_bin_centers(self, axis=None):
        if axis is not None:
            return (self.get_bin_right_edges(axis) + self.get_bin_left_edges(axis)) / 2
        else:
            raise NotImplementedError()

    def find_bin(self, value, axis=None):  # TODO: Check!
        if axis is not None:
            ixbin = np.searchsorted(self.get_bin_left_edges(axis), value, side="right")
            if ixbin == 0:
                return None
            elif ixbin == self.shape[axis]:
                if value <= self.get_bin_right_edges(axis)[-1]:
                    return ixbin - 1
                else:
                    return self.shape[axis]
            elif value < self.get_bin_right_edges(axis)[ixbin - 1]:
                return ixbin - 1
            elif ixbin == self.shape[axis]:
                return None
            else:
                return None
        else:
            ixbin = [self.find_bin(value[i], i) for i in range(self._dimension)]
            if None in ixbin:
                return None
            else:
                return ixbin

    def fill(self, value, weight=1):
        ixbin = self.find_bin(value)
        if ixbin is None and self.keep_missed:
            self.missed += weight
        else:
            self._frequencies[ixbin] += weight
            self.errors2[ixbin] += weight ** 2
        return ixbin

    def copy(self, include_frequencies=True):
        if include_frequencies:
            frequencies = np.copy(self.frequencies)
            missed = self.missed
            errors2 = self.errors2
        else:
            frequencies = None
            missed = 0
            errors2 = None
        return self.__class__(bins=[np.copy(bins) for bins in self.bins],
                              frequencies=frequencies, errors2=errors2,
                              name=self.name, axis_names=self.axis_names[:],
                              keep_missed=self.keep_missed, missed=missed)

    # TODO: same_bins...

    def __eq__(self, other):
        """Equality comparison

        """
        # TODO: Describe allclose
        # TODO: Think about softer alternatives (like compare method)
        if not isinstance(other, self.__class__):
            return False
        if not self.ndim == other.ndim:
            return False
        for i in range(self._dimension):
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

    def __add__(self, other):
        new = self.copy()
        new += other
        return new

    def __sub__(self, other):
        new = self.copy()
        new -= other
        return new

    def __mul__(self, other):
        new = self.copy()
        new *= other
        return new

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        new = self.copy()
        new /= other
        return new

    def __iadd__(self, other):
        raise NotImplementedError()

    def __imul__(self, other):
        raise NotImplementedError()

    def __isub__(self, other):
        raise NotImplementedError()

    def __itruediv__(self, other):
        raise NotImplementedError()

    def __array__(self):
        """Convert to numpy array.

        Returns
        -------
        numpy.ndarray
            The array of frequencies

        See also
        --------
        frequencies
        """
        return self.frequencies

    # TODO: to_dataframe() ?

    def to_xarray(self):
        raise NotImplementedError()

    @classmethod
    def from_xarray(cls, arr):
        raise  NotImplementedError()

    def to_json(self, path=None):
        raise NotImplementedError()

    @classmethod
    def from_json(cls, text=None, path=None):
        return NotImplementedError


class Histogram2D(HistogramND):
    def __init__(self, bins, frequencies=None, **kwargs):
        super(Histogram2D, self).__init__(2, bins, frequencies, **kwargs)

    def plot(self, *args):
        raise NotImplementedError()


def calculate_frequencies(data, ndim, bins, weights=None):
    data = np.asarray(data)

    edges_and_mask = [bin_utils.to_numpy_bins_with_mask(bins[i]) for i in range(ndim)]
    edges = [em[0] for em in edges_and_mask]
    masks = [em[1] for em in edges_and_mask]

    ixgrid = np.ix_(*masks) # Indexer to select parts we want

    if weights is not None:
        weights = np.asarray(weights)
        if weights.shape != (data.shape[0],):
            raise RuntimeError("Invalid weights shape.")
        total_weight = weights.sum()
    else:
        total_weight = data.shape[0]

    frequencies, _ = np.histogramdd(data, edges, weights=weights)

    frequencies = frequencies[ixgrid].copy()

    missing = total_weight - frequencies.sum()

    if weights is not None:
        err_freq, _ = np.histogramdd(data, edges, weights=weights ** 2)
        errors2 = err_freq[ixgrid].copy()
    else:
        errors2 = frequencies.copy()

    return frequencies, missing, errors2

