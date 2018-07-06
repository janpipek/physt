"""Multi-dimensional histograms."""
from __future__ import absolute_import, division

import numpy as np

from .histogram_base import HistogramBase


class HistogramND(HistogramBase):
    """Multi-dimensional histogram data.

    Attributes
    ----------

    """

    def __init__(self, dimension, binnings, frequencies=None, **kwargs):
        """Constructor

        Parameters
        ----------
        dimension: int

        binnings: Iterable[physt.binnings.BinningBase]
            The binnings for all axes.
        frequencies: Optional[array_like]
            The bin contents.
        errors2: Optional[array_like]
            Quadratic errors of individual bins. If not set, defaults to frequencies.
        keep_missed: bool
        missed: int or float (dtype?)
        name: Optional[str]
        axis_names: Optional[Iterable[str]]
        """

        # Bins + checks
        if len(binnings) != dimension:
            raise RuntimeError("bins must be a sequence of {0} schemas".format(dimension))
        missed = kwargs.pop("missed", 0)

        HistogramBase.__init__(self, binnings, frequencies, **kwargs)

        if len(self.axis_names) != self.ndim:
            raise RuntimeError("The length of axis names must be equal to histogram dimension.")

        # Missed values
        self._missed = np.array([missed], dtype=self.dtype)

    # Not supported yet
    _stats = None

    @property
    def bins(self):
        """Matrix of bins.

        Returns
        -------
        list[np.ndarray]
            Two sets of array bins.
        """
        return [binning.bins for binning in self._binnings]

    @property
    def binnings(self):
        """The binnings.

        Note: Please, do not try to update the objects themselves.

        Returns
        -------
        list[physt.binnings.BinningBase]
        """
        return self._binnings

    @property
    def numpy_bins(self):
        """Numpy-like bins (if available)

        Returns
        -------
        list[np.ndarray]
        """
        return [binning.numpy_bins for binning in self._binnings]

    def select(self, axis, index, force_copy=False):
        """Select in an axis.

        Parameters
        ----------
        axis: int or str
            Axis, in which we select.
        index: int or slice
            Index of bin (as in numpy).
        force_copy: bool
            If True, identity slice force a copy to be made.

        Returns
        -------
        HistogramND or Histogram2D or Histogram1D (or others in special cases)
        """
        if index == slice(None) and not force_copy:
            return self

        axis_id = self._get_axis(axis)
        array_index = [slice(None, None, None) for i in range(self.ndim)]
        array_index[axis_id] = index

        frequencies = self._frequencies[array_index].copy()
        errors2 = self._errors2[array_index].copy()

        if isinstance(index, int):
            return self._reduce_dimension([ax for ax in range(self.ndim) if ax != axis_id], frequencies, errors2)
        elif isinstance(index, slice):
            if index.step is not None and index.step < 0:
                raise IndexError("Cannot change the order of bins")
            copy = self.copy()
            copy._frequencies = frequencies
            copy._errors2 = errors2
            copy._binnings[axis_id] = self._binnings[axis_id][index]
            return copy
        else:
            raise ValueError("Invalid index.")

    def __getitem__(self, index):
        """Select subset of histogram.
        
        Parameters
        ----------
        index: int or slice or iterable
            One or more indices to select in subsequent axes.

        Returns
        -------
        HistogramBase or tuple
            Depending on the parameters, a sub-histogram or content of one bin are returned.

        Indexing shares semantics with Numpy arrays, however

        Always returns a new object.
        """
        # TODO: Enable views
        if isinstance(index, (int, slice)):
            return self.select(0, index)
        elif isinstance(index, tuple):
            if len(index) > self.ndim:
                raise IndexError("Too many indices ({0}) to select from {1}D histogram".
                                 format(len(index), self.ndim))

            # Scalar case => return (bin edges), (frequency)
            if len(index) == self.ndim and all((isinstance(i, int) for i in index)):
                return (
                    tuple((self.get_bin_left_edges(i)[j], self.get_bin_right_edges(i)[j]) for i, j in enumerate(index)),
                    self._frequencies[index]
                )
            current = self
            for i, subindex in enumerate(index):              
                current = current.select(i + current.ndim - self.ndim, subindex, force_copy=False)
            if current is self:
                current = current.copy()
            return current
        else:
            raise ValueError("Invalid index.")

    # Missing: cumulative_frequencies - does it make sense?

    def get_bin_widths(self, axis=None):  # -> Base
        if axis is not None:
            return self.get_bin_right_edges(axis) - self.get_bin_left_edges(axis)
        else:
            return np.meshgrid(*[self.get_bin_widths(i) for i in range(self.ndim)], indexing='ij')

    @property
    def bin_sizes(self):
        # TODO: Some kind of caching?
        sizes = self.get_bin_widths(0)
        for i in range(1, self.ndim):
            sizes = np.multiply.outer(sizes, self.get_bin_widths(i))
        return sizes

    @property
    def total_size(self):
        """The total size of the bin space.

        Returns
        -------
        float

        Note
        ----
        Perhaps not optimized, but should work also with transformed axes
        """
        return np.sum(self.bin_sizes)

    def get_bin_edges(self, axis=None):
        # TODO: test for non-numpy ones
        if axis is not None:
            return self.numpy_bins[self._get_axis(axis)]
        else:
            edges = [self.get_bin_edges(i) for i in range(self.ndim)]
            return np.meshgrid(*edges, indexing='ij')

    def get_bin_left_edges(self, axis=None):
        if axis is not None:
            return self.bins[self._get_axis(axis)][:, 0]
        else:
            edges = [self.get_bin_left_edges(i) for i in range(self.ndim)]
            return np.meshgrid(*edges, indexing='ij')

    def get_bin_right_edges(self, axis=None):
        if axis is not None:
            return self.bins[self._get_axis(axis)][:, 1]
        else:
            edges = [self.get_bin_right_edges(i) for i in range(self.ndim)]
            return np.meshgrid(*edges, indexing='ij')

    def get_bin_centers(self, axis=None):
        if axis is not None:
            return (self.get_bin_right_edges(axis) + self.get_bin_left_edges(axis)) / 2
        else:
            return np.meshgrid(*[self.get_bin_centers(i) for i in range(self.ndim)], indexing='ij')

    def find_bin(self, value, axis=None):  # TODO: Check!
        """Index(indices) of bin corresponding to a value.

        Parameters
        ----------
        value: array_like
            Value with dimensionality equal to histogram
        axis: Optional[int]
            If set, find axis along an axis. Otherwise, find bins along all axes.
            None = outside the bins

        Returns
        -------
        int or tuple or None
            If axis is specified, a number. Otherwise, a tuple. If not available, None.
        """
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

    def fill(self, value, weight=1, **kwargs):
        self._coerce_dtype(type(weight))
        for i, binning in enumerate(self._binnings):
            if binning.is_adaptive():
                bin_map = binning.force_bin_existence(value[i])
                self._reshape_data(binning.bin_count, bin_map, i)
        ixbin = self.find_bin(value, **kwargs)
        if ixbin is None and self.keep_missed:
            self._missed += weight
        else:
            self._frequencies[ixbin] += weight
            self._errors2[ixbin] += weight ** 2
        return ixbin

    def fill_n(self, values, weights=None, dropna=True, columns=False):
        """Add more values at once.

        Parameters
        ----------
        values: array_like
            Values to add. Can be array of shape (count, ndim) or
            array of shape (ndim, count) [use columns=True] or something
            convertible to it
        weights: array_like
            Weights for values (optional)
        dropna: bool
            Whether to remove NaN values. If False and such value is met,
            exception is thrown.
        columns: bool
            Signal that the data are transposed (in columns, instead of rows).
            This allows to pass list of arrays in values.
        """
        values = np.asarray(values)
        if values.ndim != 2:
            raise RuntimeError("Expecting 2D array of values.")
        if columns:
            values = values.T
        if values.shape[1] != self.ndim:
            raise RuntimeError("Expecting array with {0} columns".format(self.ndim))
        if dropna:
            values = values[~np.isnan(values).any(axis=1)]
        if weights is not None:
            weights = np.asarray(weights)
            # TODO: Check for weights size?
            self._coerce_dtype(weights.dtype)
        for i, binning in enumerate(self._binnings):
            if binning.is_adaptive():
                map = binning.force_bin_existence(values[:, i])   # TODO: Add to some test
                self._reshape_data(binning.bin_count, map, i)
        frequencies, errors2, missed = calculate_frequencies(values, self.ndim,
                                                             self._binnings, weights=weights)
        self._frequencies += frequencies
        self._errors2 += errors2
        self._missed[0] += missed

    def _get_projection_axes(self, *axes):
        axes = list(axes)
        for i, axis in enumerate(axes):
            if isinstance(axis, str):
                if axis not in self.axis_names:
                    raise RuntimeError("Invalid axis name for projection: " + axis)
                axes[i] = self.axis_names.index(axis)
        if not axes:
            raise RuntimeError("No axis selected for projection")
        if len(axes) != len(set(axes)):
            raise RuntimeError("Duplicate axes in projection")
        invert = list(range(self.ndim))
        for axis in axes:
            invert.remove(axis)
        axes = tuple(axes)
        invert = tuple(invert)
        return (axes, invert)

    def _reduce_dimension(self, axes, frequencies, errors2, **kwargs):
        name = kwargs.pop("name", self.name)
        axis_names = [name for i, name in enumerate(self.axis_names) if i in axes]
        bins = [bins for i, bins in enumerate(self._binnings) if i in axes]
        if len(axes) == 1:
            from .histogram1d import Histogram1D
            klass = kwargs.get("type", Histogram1D)
            return klass(binning=bins[0], frequencies=frequencies, errors2=errors2,
                         axis_name=axis_names[0], name=name)
        elif len(axes) == 2:
            klass = kwargs.get("type", Histogram2D)
            return klass(binnings=bins, frequencies=frequencies, errors2=errors2,
                         axis_names=axis_names, name=name)
        else:
            klass = kwargs.get("type", HistogramND)
            return klass(dimension=len(axes), binnings=bins, frequencies=frequencies,
                         errors2=errors2, axis_names=axis_names, name=name)

    def accumulate(self, axis):
        """Calculate cumulative frequencies along a certain axis.

        Parameters
        ----------
        axis: int or str

        Returns
        -------
        new_hist : HistogramND or Histogram2D
            Histogram of the same type & size
        """
        # TODO: Merge with Histogram1D.cumulative_frequencies
        # TODO: Deal with errors and totals etc.
        new_one = self.copy()
        axis_id, _ = self._get_projection_axes(axis)
        new_one._frequencies = np.cumsum(new_one.frequencies, axis_id[0])
        return new_one

    def projection(self, *axes, **kwargs):
        """Reduce dimensionality by summing along axis/axes.

        Parameters
        ----------
        axes: Iterable[int or str]
            List of axes for the new histogram. Could be either
            numbers or names. Must contain at least one axis.
        name: Optional[str] # TODO: Check
            Name for the projected histogram (default: same)
        type: Optional[type] # TODO: Check
            If set, predefined class for the projection

        Returns
        -------
        HistogramND or Histogram2D or Histogram1D (or others in special cases)
        """
        # TODO: rename to project in 0.4
        axes, invert = self._get_projection_axes(*axes)
        frequencies = self.frequencies.sum(axis=invert)
        errors2 = self.errors2.sum(axis=invert)
        return self._reduce_dimension(axes, frequencies, errors2, **kwargs)

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


class Histogram2D(HistogramND):
    """Specialized 2D variant of the general HistogramND class.

    In contrast to general HistogramND, it is plottable.
    """

    def __init__(self, binnings, frequencies=None, **kwargs):
        kwargs.pop("dimension", None)
        super(Histogram2D, self).__init__(dimension=2, binnings=binnings,
                                          frequencies=frequencies, **kwargs)

    @property
    def T(self):
        """Histogram with swapped axes.

        Returns
        -------
        Histogram2D - a copy with swapped axes
        """
        a_copy = self.copy()
        a_copy._binnings = list(reversed(a_copy._binnings))
        a_copy.axis_names = list(reversed(a_copy.axis_names))
        a_copy._frequencies = a_copy._frequencies.T
        a_copy._errors2 = a_copy._errors2.T
        return a_copy

    def partial_normalize(self, axis=0, inplace=False):
        """Normalize in rows or columns.

        Parameters
        ----------
        axis: int or str
            Along which axis to sum (numpy-sense)
        inplace: bool
            Update the object itself

        Returns
        -------
        hist : Histogram2D
        """
        axis = self._get_axis(axis)
        if not inplace:
            copy = self.copy()
            copy.partial_normalize(axis, inplace=True)
            return copy
        else:
            self._coerce_dtype(float)
            if axis == 0:
                divisor = self._frequencies.sum(axis=0)
            else:
                divisor = self._frequencies.sum(axis=1)[:, np.newaxis]
            divisor[divisor == 0] = 1             # Prevent division errors
            self._frequencies /= divisor
            self._errors2 /= (divisor * divisor)  # Has its limitations
            return self

    def numpy_like(self):
        return self.frequencies, self.numpy_bins[0], self.numpy_bins[1]


def calculate_frequencies(data, ndim, binnings, weights=None, dtype=None):
    """"Get frequencies and bin errors from the data (n-dimensional variant).

    Parameters
    ----------
    data : array_like
        2D array with ndim columns and row for each entry.
    ndim : int
        Dimensionality od the data.
    binnings:
        Binnings to apply in all axes.
    weights : Optional[array_like]
        1D array of weights to assign to values.
        (If present, must have same length as the number of rows.)
    dtype : Optional[type]
        Underlying type for the histogram.
        (If weights are specified, default is float. Otherwise int64.)

    Returns
    -------
    frequencies : array_like
    errors2 : array_like
    missing : scalar[dtype]
    """

    # TODO: Remove ndim
    # TODO: What if data is None

    # Prepare numpy array of data
    if data is not None:
        data = np.asarray(data)
        if data.ndim != 2:
            raise RuntimeError("histogram_nd.calculate_frequencies requires 2D input data.")
            # TODO: If somewhere, here we would check ndim

    # Guess correct dtype and apply to weights
    if weights is None:
        if not dtype:
            dtype = np.int64
        if data is not None:
            weights = np.ones(data.shape[0], dtype=dtype)
    else:
        weights = np.asarray(weights)
        if data is None:
            raise RuntimeError("Weights specified but data not.")
        else:
            if data.shape[0] != weights.shape[0]:
                raise RuntimeError("Different number of entries in data and weights.")
        if dtype:
            dtype = np.dtype(dtype)
            if dtype.kind in "iu" and weights.dtype.kind == "f":
                raise RuntimeError("Integer histogram requested but float weights entered.")
        else:
            dtype = weights.dtype

    edges_and_mask = [binning.numpy_bins_with_mask for binning in binnings]
    edges = [em[0] for em in edges_and_mask]
    masks = [em[1] for em in edges_and_mask]

    ixgrid = np.ix_(*masks)  # Indexer to select parts we want

    # TODO: Right edges are not taken into account because they fall into inf bin

    if data.shape[0]:
        frequencies, _ = np.histogramdd(data, edges, weights=weights)
        frequencies = frequencies.astype(dtype)      # Automatically copy
        frequencies = frequencies[ixgrid]
        missing = weights.sum() - frequencies.sum()
        err_freq, _ = np.histogramdd(data, edges, weights=weights ** 2)
        errors2 = err_freq[ixgrid].astype(dtype)  # Automatically copy
    else:
        frequencies = None
        missing = 0
        errors2 = None

    return frequencies, errors2, missing
