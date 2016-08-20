from __future__ import absolute_import
import numpy as np


class HistogramBase(object):
    """Behaviour shared between all histogram classes.

    The most important daughter classes are:
    - Histogram1D
    - HistogramND

    The methods you should override:
    - fill
    - fill_n (optional)
    - copy

    Attributes
    ----------
    _binnings : Iterable[BinningBase]
        Schema for binning(s)
    _frequencies : array_like
        Bin contents
    _errors2 : array_like
        Square errors associated with the bin contents
    _meta_data : dict
    name: str
        Name to be displayed for the histogram

    """

    @property
    def shape(self):
        """Shape of histogram's data.

        Returns
        -------
        tuple[int]
            One-element tuple
        """
        return tuple(bins.bin_count for bins in self._binnings)

    @property
    def ndim(self):
        """Dimensionality of histogram's data.

        i.e. the number of axes along which we bin the values.

        Returns
        -------
        int
        """
        return len(self._binnings)

    @property
    def dtype(self):
        """Data type of the histogram.

        Returns
        -------
        np.dtype
        """
        return self._frequencies.dtype

    @dtype.setter
    def dtype(self, value):
        """Change data type of the histogram.

        Allowed conversions:
        - from integral to float types
        - between the same category of type (float/integer)
        - from float types to integer if weights are trivial

        Parameters
        ----------
        value: np.dtype or something convertible to it.
        """
        value = np.dtype(value)
        if value.kind not in "iuf":
            raise RuntimeError("Unsupported dtype. Only integer/floating-point types are supported.")
        ok = False
        if np.issubdtype(value, np.integer):
            if np.issubdtype(self.dtype, np.integer):
                ok = True
            elif np.array_equal(self._frequencies, self._errors2):
                ok = True
        elif np.issubdtype(value, np.float):
            ok = True
        if ok:
            self._frequencies = self._frequencies.astype(value)
            self._errors2 = self._errors2.astype(value)
            self._missed = self._missed.astype(value)
            # TODO: Overflows and underflows and stuff...
        else:
            raise RuntimeError("Cannot change histogram dtype.")

    def _coerce_dtype(self, other_dtype):
        """Possibly change the type to allow correct operations with other operand.

        Parameters
        ----------
        other_dtype : np.dtype or type
        """
        other_dtype = np.dtype(other_dtype)
        if other_dtype.kind == self.dtype.kind:
            pass
        elif self.dtype.kind in "iu":
            self.dtype = other_dtype
        else:
            pass # not changing - already float

    @property
    def bin_count(self):
        """Total number of bins.

        Returns
        -------
        int
        """
        return np.product(self.shape)

    @property
    def frequencies(self):
        """Frequencies (values) of the histogram.

        Returns
        -------
        np.ndarray
            Array of bin frequencies
        """
        return self._frequencies

    @property
    def densities(self):
        """Frequencies normalized by bin sizes.

        Useful when bins are not of the same size.

        Returns
        -------
        np.ndarray
        """
        return self._frequencies / self.bin_sizes

    def normalize(self, inplace=False):
        """Normalize the histogram, so that the total weight is equal to 1.

        Returns
        -------
        HistogramBase

        See also
        --------
        densities

        """
        if inplace:
            self /= self.total
            return self
        else:
            return self / self.total

    @property
    def errors2(self):
        """Squares of the bin errors.

        Returns
        -------
        np.ndarray
        """
        return self._errors2

    @property
    def errors(self):
        """Bin errors.

        Returns
        -------
        np.ndarray
        """
        return np.sqrt(self.errors2)

    @property
    def total(self):
        """Total number (sum of weights) of entries excluding underflow and overflow.

        Returns
        -------
        float
        """
        return self._frequencies.sum()

    @property
    def missed(self):
        """Total number (weight) of entries that missed the bins.

        Returns
        -------
        float
        """
        return self._missed.sum()

    def is_adaptive(self):
        """Whether the binning can be changed with operations.

        Returns
        -------
        bool
        """
        return all(binning.is_adaptive() for binning in self._binnings)

    def set_adaptive(self, value=True):
        """Change the histogram binning to (non)adaptive.

        This requires binning in all dimensions to allow this.

        Parameters
        ----------
        value : bool
        """
        for binning in self._binnings:
            binning.set_adaptive(value)

    def change_binning(self, new_binning, bin_map, axis=0):
        """Set new binnning and update the bin contents according to a map.

        Fills frequencies and errors with 0.
        It's the caller's responsibility to provide correct binning and map.

        Parameters
        ----------
        new_binning: physt.binnings.BinningBase
        bin_map: Iterable[tuple]
            tuples contain bin indices (old, new)
        axis: int
            What axis does the binning describe(0..ndim-1)
        """
        axis = int(axis)
        if axis < 0 or axis >= self.ndim:
            raise RuntimeError("Axis must be in range 0..(ndim-1)")
        self._reshape_data(new_binning.bin_count, bin_map, axis)
        self._binnings[axis] = new_binning

    def merge_bins(self, amount=None, min_frequency=None, axis=None, inplace=True):
        """Reduce the number of bins and add their content:

        Parameters
        ----------
        amount: int
            How many adjacent bins to join together.
        min_frequency: float
            Try to have at least this value in each bin
            (this is not enforce e.g. for minima between high bins)
        axis: int or None
            On which axis to do this (None => all)
        inplace:
            Whether to modify this histogram or return a new one

        Returns
        -------
        HistogramBase or None
            if inplace, return
        """
        if not inplace:
            histogram = self.copy()
            histogram.merge_bins(amount, min_frequency=min_frequency, axis=axis)
            return histogram
        elif axis is None:
            for i in range(self.ndim):
                self.merge_bins(amount=amount, min_frequency=min_frequency, axis=i)
        else:
            if amount is not None:
                if not amount == int(amount):
                    raise RuntimeError("Amount must be integer")
                bin_map = [(i, i // amount) for i in range(self.shape[axis])]
            elif min_frequency is not None:
                if self.ndim == 1:
                    check = self.frequencies
                else:
                    check = self.projection(axis).frequencies
                bin_map = []
                current_new = 0
                current_sum = 0
                for i, freq in enumerate(check):
                    if freq >= min_frequency and current_sum > 0:
                        current_sum = 0
                        current_new += 1
                    bin_map.append((i, current_new))
                    current_sum += freq
                    if current_sum > min_frequency:
                        current_sum = 0
                        current_new += 1
            else:
                raise NotImplementedError("Not yet implemented.")
            new_binning = self._binnings[axis].apply_bin_map(bin_map)
            self.change_binning(new_binning, bin_map, axis=axis)

    def _reshape_data(self, new_size, bin_map, axis=0):
        """Reshape data to match new binning schema.

        Fills frequencies and errors with 0.

        Parameters
        ----------
        new_size: int
        bin_map: Iterable[(old, new)] or int or None
            If None, we can keep the data unchanged.
            If int, it is offset by which to shift the data (can be 0)
            If iterable, pairs specify which old bin should go into which new bin
        axis: int
            On which axis to apply
        """
        if bin_map is None:
            return
        else:
            new_shape = list(self.shape)
            new_shape[axis] = new_size
            new_frequencies = np.zeros(new_shape, dtype=self._frequencies.dtype)
            new_errors2 = np.zeros(new_shape, dtype=self._frequencies.dtype)
            self._apply_bin_map(
                old_frequencies=self._frequencies, new_frequencies=new_frequencies,
                old_errors2=self._errors2, new_errors2=new_errors2,
                bin_map=bin_map, axis=axis)
            self._frequencies = new_frequencies
            self._errors2 = new_errors2

    def _apply_bin_map(self, old_frequencies, new_frequencies, old_errors2, new_errors2, bin_map, axis=0):
        """Fill new data arrays using a map.

        Parameters
        ----------
        old_frequencies : np.ndarray
            Source of frequencies data
        new_frequencies : np.ndarray
            Target of frequencies data
        old_errors2 : np.ndarray
            Source of errors data
        new_errors2 : np.ndarray
            Target of errors data
        bin_map: Iterable[(old, new)] or int or None
            As in _reshape_data
        axis: int
            On which axis to apply

        See also
        --------
        HistogramBase._reshape_data
        """
        if old_frequencies is not None and old_frequencies.shape[axis] > 0:
            if isinstance(bin_map, int):
                new_index = [slice(None) for i in range(self.ndim)]
                new_index[axis] = slice(bin_map, bin_map + old_frequencies.shape[axis])
                new_frequencies[new_index] += old_frequencies
                new_errors2[new_index] += old_errors2
            else:
                for (old, new) in bin_map:      # Generic enough
                    new_index = [slice(None) for i in range(self.ndim)]
                    new_index[axis] = new
                    old_index = [slice(None) for i in range(self.ndim)]
                    old_index[axis] = old
                    new_frequencies[new_index] += old_frequencies[old_index]
                    new_errors2[new_index] += old_errors2[old_index]

    def has_same_bins(self, other):
        """Whether two histograms share the same binning.

        Returns
        -------
        bool
        """
        if self.shape != other.shape:
            return False
        elif self.ndim == 1:
            return np.allclose(self.bins, other.bins)
        elif self.ndim > 1:
            for i in range(self.ndim):
                if not np.allclose(self.bins[i], other.bins[i]):
                    return False
            return True

    # Unused?
    # def has_compatible_bins(self, other):
    #     # By default, the bins must be the same
    #     # Overridden
    #     return self.has_same_bins()

    def fill_n(self, values, weights=None):
        """Add more values at once.

        This (default) implementation uses a simple loop to add values using `fill` method

        Parameters
        ----------
        values: Iterable
            Values to add
        weights: Optional[Iterable]
            Optional values to assign to each value

        Note
        ----
        This method should be overloaded with a more efficient one.
        """
        if weights is not None:
            if weights.shape != values.shape[0]:
                raise RuntimeError("Wrong shape of weights")
        for i, value in enumerate(values):
            if weights is not None:
                self.fill(value, weights[i])
            else:
                self.fill(value)

    @property
    def plot(self):
        """Proxy to plotting.

        This attribute is a special proxy to plotting. In the most
        simple cases, it can be used as a method. For more sophisticated
        use, see the documentation for physt.plotting package.

        Returns
        -------
        physt.plotting.PlottingProxy
        """
        from .plotting import PlottingProxy
        return PlottingProxy(self)

    def __add__(self, other):
        new = self.copy()
        new += other
        return new

    def __radd__(self, other):
        if other == 0:    # Enable sum()
            return self
        else:
            return self + other

    def __iadd__(self, other):
        if np.isscalar(other):
            raise RuntimeError("Cannot add constant to histograms.")
        if other.ndim != self.ndim:
            raise RuntimeError("Cannot add histograms with different dimensions.")
        elif self.has_same_bins(other):
            # print("Has same!!!!!!!!!!")
            self._coerce_dtype(other.dtype)
            self._frequencies += other.frequencies
            self._errors2 += other.errors2
            self._missed += other._missed
        elif self.is_adaptive():
            if other.missed > 0:
                raise RuntimeError("Cannot adapt histogram with missed values.")
            try:
                other = other.copy()
                other.set_adaptive(True)

                self._coerce_dtype(other.dtype)

                # TODO: Fix state after exception
                # maps1 = []
                maps2 = []
                for i in range(self.ndim):
                    new_bins = self._binnings[i].copy()

                    map1, map2 = new_bins.adapt(other._binnings[i])
                    self.change_binning(new_bins, map1, axis=i)
                    other.change_binning(new_bins, map2, axis=i)
                self._frequencies += other.frequencies
                self._errors2 += other.errors2

            except:
                raise # RuntimeError("Cannot find common binning for added histograms.")
        else:
            raise RuntimeError("Incompatible binning")

        if self._stats and other._stats:
            for key in self._stats:
                self._stats[key] += other._stats[key]
        return self

    def __sub__(self, other):
        new = self.copy()
        new -= other
        return new

    def __isub__(self, other):
        return self.__iadd__(other * (-1))

    def __mul__(self, other):
        new = self.copy()
        new *= other
        return new

    def __imul__(self, other):
        if not np.isscalar(other):
            raise RuntimeError("Histograms may be multiplied only by a constant.")
        if np.issubdtype(self.dtype, int) and np.issubdtype(type(other), float):
            self.dtype = float
        self._frequencies *= other
        self._errors2 *= other ** 2
        self._missed *= other
        if self._stats:
            self._stats["sum"] *= other
            self._stats["sum2"] *= other ** 2
        return self

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        new = self.copy()
        new /= other
        return new

    def __itruediv__(self, other):
        if not np.isscalar(other):
            raise RuntimeError("Histograms may be divided only by a constant.")
        self._coerce_dtype(np.float64)
        self._frequencies /= other
        self._errors2 /= other ** 2
        self._missed /= other
        if self._stats:
            self._stats["sum"] /= other
            self._stats["sum2"] /= other ** 2
        return self

    def __array__(self):
        """Convert to numpy array.

        Returns
        -------
        np.ndarray
            The array of frequencies

        See also
        --------
        frequencies
        """
        return self.frequencies
