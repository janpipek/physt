"""HistogramBase - base for all histogram classes."""
from __future__ import absolute_import, division
import numpy as np
from .binnings import as_binning


class HistogramBase(object):
    """Histogram base class.

    Behaviour shared by all histogram classes.

    The most important daughter classes are:
    - Histogram1D
    - HistogramND

    There are also special histogram types that are modifications of these classes.

    The methods you should override:
    - fill
    - fill_n (optional)
    - copy
    - _update_dict (optional)

    Underlying data type is int64 / float  or an explicitly specified
    other type (dtype).

    Attributes
    ----------
    _binnings : Iterable[BinningBase]
        Schema for binning(s)
    _frequencies : array_like
        Bin contents
    _errors2 : array_like
        Square errors associated with the bin contents
    _meta_data : dict
        All meta-data (names, user-custom values, ...). Anything can be put in.
        When exported, all information is kept.
    _dtype : np.dtype
        Type of the frequencies and also errors (int64, float64 or user-overridden)
    _missed : array_like
        Various storage for missed values in different histogram types
        (1 value for multi-dimensional, 3 values for one-dimensional)

    Invariants
    ----------
    - Frequencies in the histogram should always be non-negative.
    Many operations rely on that, but it is not always enforced.
    (TODO: Fix this?)

    See Also
    --------
    histogram1d
    histogram_nd
    special

    """

    def __init__(self, binnings, frequencies=None, errors2=None, **kwargs):
        """Constructor

        All keyword arguments not listed below become items in the _meta_data
        dictionary.

        Parameters
        ----------
        binnings : Iterable[BinningBase or array_like]
        frequencies : Optional[array_like]
        errors2 : Optional[array_like]
        dtype : np.dtype
        keep_missed : bool

        """
        self._binnings = [as_binning(binning) for binning in binnings]

        # Frequencies + appropriate dtypes
        if frequencies is None:
            dtype = kwargs.pop("dtype", np.int64)
            self._frequencies = np.zeros(self.shape, dtype=dtype)
        else:
            dtype = kwargs.pop("dtype", None)
            if dtype is not None:
                frequencies = np.asarray(frequencies, dtype=dtype)
            else:
                frequencies = np.asarray(frequencies)
                if np.issubdtype(frequencies.dtype, np.integer):
                    frequencies = frequencies.astype(np.int64)
                elif np.issubdtype(frequencies.dtype, np.floating):
                    frequencies = frequencies.astype(np.float64)
                else:
                    raise RuntimeError("Frequencies of type {0} not understood"
                                       .format(frequencies.dtype))
            dtype = frequencies.dtype
            if frequencies.shape != self.shape:
                raise RuntimeError("Values must have same dimension as bins.")
            if np.any(frequencies < 0):
                raise RuntimeError("Cannot have negative frequencies.")
            self._frequencies = frequencies
        self._dtype = dtype

        # Errors
        if errors2 is None:
            self._errors2 = self._frequencies.copy()
        else:
            self._errors2 = np.asarray(errors2, dtype=self.dtype)
        if np.any(self._errors2 < 0):
            raise RuntimeError("Cannot have negative squared errors.")
        if self._errors2.shape != self._frequencies.shape:
            raise RuntimeError("Errors must have same dimension as frequencies.")

        self.keep_missed = kwargs.pop("keep_missed", True)
        # Note: missed are dealt differently in 1D/ND cases

        if "axis_names" not in kwargs:
            kwargs["axis_names"] = ["axis{0}".format(i) for i in range(self.ndim)]

        # Meta data
        self._meta_data = kwargs.copy()

    @property
    def meta_data(self):
        """A dictionary of non-numerical information about the histogram.

        It contains several pre-defined ones, but you can add any other.
        These are preserved when saving and also in operations.

        Returns
        -------
        dict
        """
        return self._meta_data

    @property
    def name(self):
        """Name of the histogram (stored in meta-data).

        Returns
        -------
        str
        """
        return self._meta_data.get("name", None)

    @name.setter
    def name(self, value):
        """Name of the histogram.

        In plotting, this will be used as label.
        """
        self._meta_data["name"] = str(value)

    @property
    def title(self):
        """Title of the histogram to be displayed when plotted (stored in meta-data).
        
        If not specified, defaults to `name`.

        Returns
        -------
        str
        """
        return self._meta_data.get("title", self.name)

    @title.setter
    def title(self, value):
        """Title of the histogram.

        In plotting, this will be used as plot title.
        """
        self._meta_data["title"] = str(value)

    @property
    def axis_names(self):
        """Names of axes (stored in meta-data).

        Returns
        -------
        tuple[str]
        """
        default = ["axis{0}".format(i) for i in range(self.ndim)]
        return tuple(self._meta_data.get("axis_names", None) or default)

    @axis_names.setter
    def axis_names(self, value):
        self._meta_data["axis_names"] = tuple(str(name) for name in value)

    def _get_axis(self, name_or_index):
        """Get index of an axis and check its existence

        Parameters
        ----------
        name_or_index : str or int

        Returns
        -------
        int
            zero-based axis index
        """
        # TODO: Add unit test
        if isinstance(name_or_index, int):
            if name_or_index < 0 or name_or_index >= self.ndim:
                raise RuntimeError("No such axis, must be from 0 to {0}".format(self.ndim-1))
            return name_or_index
        elif isinstance(name_or_index, str):
            if name_or_index not in self.axis_names:
                named_axes = [name for name in self.axis_names if name]
                raise RuntimeError("No axis with such name: {0}, available names: {1}. In most places, you can also use numbers."
                                   .format(name_or_index, ", ".join(named_axes)))
            return self.axis_names.index(name_or_index)
        else:
            raise RuntimeError("Argument of type {0} not understood, int or str expected.".format(type(name_or_index)))

    @property
    def shape(self):
        """Shape of histogram's data.

        Returns
        -------
        tuple[int]
            One-element tuple with the number of bins along each axis.
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

    def _get_dtype(self):
        """Data type of the bin contents.

        Returns
        -------
        np.dtype
        """
        return self._dtype

    def set_dtype(self, value, check=True):
        """Change data type of the bin contents.

        Allowed conversions:
        - from integral to float types
        - between the same category of type (float/integer)
        - from float types to integer if weights are trivial

        Parameters
        ----------
        value: np.dtype or something convertible to it.
        check: bool
            If True (default), all values are checked against the limits
        """

        # TODO: Refactor out?
        # TODO? Deal with unsigned types
        value = np.dtype(value)

        if value == self.dtype:
            return    # No change

        if value.kind in "iu":
            type_info = np.iinfo(value)
        elif value.kind == "f":
            type_info = np.finfo(value)
        else:
            raise RuntimeError("Unsupported dtype. Only integer/floating-point types are supported.")

        if np.can_cast(self.dtype, value):
            pass    # Ok
        elif check:
            if np.issubdtype(value, np.integer):
                if self.dtype.kind == "f":
                    for array in (self._frequencies, self._errors2):
                        if np.any(array % 1.0):
                            raise RuntimeError("Data contain non-integer values.")
            for array in (self._frequencies, self._errors2):
                if np.any((array > type_info.max) | (array < type_info.min)):
                    raise RuntimeError("Data contain values outside the specified range.")

        self._dtype = value
        self._frequencies = self._frequencies.astype(value)
        self._errors2 = self._errors2.astype(value)
        self._missed = self._missed.astype(value)

    dtype = property(_get_dtype, set_dtype)

    def _coerce_dtype(self, other_dtype):
        """Possibly change the bin content type to allow correct operations with other operand.

        Parameters
        ----------
        other_dtype : np.dtype or type
        """
        new_dtype = np.find_common_type([self.dtype, np.dtype(other_dtype)], [])
        if new_dtype != self.dtype:
            self.dtype = new_dtype

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
        """Frequencies (values, contents) of the histogram bins.

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

    def normalize(self, inplace=False, percent=False):
        """Normalize the histogram, so that the total weight is equal to 1.

        Parameters
        ----------
        inplace: bool
            If True, updates itself. If False (default), returns copy
        percent: bool
            If True, normalizes to percent instead of 1. Default: False

        Returns
        -------
        HistogramBase : either modified copy or self

        See also
        --------
        densities
        HistogramND.partial_normalize

        """
        if inplace:
            self /= self.total * (.01 if percent else 1)
            return self
        else:
            return self / self.total * (100 if percent else 1)

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
        # TODO: remove in favour of adaptive property
        return all(binning.is_adaptive() for binning in self._binnings)

    def set_adaptive(self, value=True):
        """Change the histogram binning to (non)adaptive.

        This requires binning in all dimensions to allow this.

        Parameters
        ----------
        value : bool
        """
        # TODO: remove in favour of adaptive property
        if not all(b.adaptive_allowed for b in self._binnings):
            raise RuntimeError("All binnings must allow adaptive behaviour.")
        for binning in self._binnings:
            binning.set_adaptive(value)

    @property
    def adaptive(self):
        return self.is_adaptive()

    @adaptive.setter
    def adaptive(self, value):
        self.set_adaptive(value)

    def _change_binning(self, new_binning, bin_map, axis=0):
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

    def merge_bins(self, amount=None, min_frequency=None, axis=None, inplace=False):
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
            histogram.merge_bins(amount, min_frequency=min_frequency, axis=axis, inplace=True)
            return histogram
        elif axis is None:
            for i in range(self.ndim):
                self.merge_bins(amount=amount, min_frequency=min_frequency, axis=i, inplace=True)
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
            self._change_binning(new_binning, bin_map, axis=axis)

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

    def _apply_bin_map(self, old_frequencies, new_frequencies, old_errors2,
                       new_errors2, bin_map, axis=0):
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

    def copy(self, include_frequencies=True):
        """Copy the histogram.

        Parameters
        ----------
        include_frequencies : Optional[bool]
            If false, all frequencies are set to zero.

        Returns
        -------
        copy : HistogramBase
        """
        if include_frequencies:
            frequencies = np.copy(self.frequencies)
            missed = self._missed.copy()
            errors2 = np.copy(self.errors2)
            stats = self._stats or None
        else:
            frequencies = np.zeros_like(self._frequencies)
            errors2 = np.zeros_like(self._errors2)
            missed = np.zeros_like(self._missed)
            stats = None
        a_copy = self.__class__.__new__(self.__class__)
        a_copy._binnings = [binning.copy() for binning in self._binnings]
        a_copy._dtype = self.dtype
        a_copy._frequencies = frequencies
        a_copy._errors2 = errors2
        a_copy._meta_data = self._meta_data.copy()
        a_copy.keep_missed = self.keep_missed
        a_copy._missed = missed
        a_copy._stats = stats
        return a_copy

    def fill(self, value, weight=1, **kwargs):
        """Add a value.

        Abstract method - to be implemented in daughter classes.s

        Parameters
        ----------
        value:
            Value to be added. Can be scalar or array depending on the histogram type.
        weight: Optional
            Weight of the value

        Note
        ----
        May change the dtype if weight is set
        """
        raise NotImplementedError("You have to define the `fill` method in Histogram class.")

    def fill_n(self, values, weights=None, **kwargs):
        """Add more values at once.

        This (default) implementation uses a simple loop to add values using `fill` method.
        Actually, it is not used in neither Histogram1D, nor HistogramND.

        Parameters
        ----------
        values: Iterable
            Values to add
        weights: Optional[Iterable]
            Optional values to assign to each value

        Note
        ----
        This method should be overloaded with a more efficient one.

        May change the dtype if weight is set.
        """
        if weights is not None:
            if weights.shape != values.shape[0]:
                raise RuntimeError("Wrong shape of weights")
        for i, value in enumerate(values):
            if weights is not None:
                self.fill(value, weights[i], **kwargs)
            else:
                self.fill(value, **kwargs)

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

    def to_dict(self):
        """Dictionary with all data in the histogram.

        This is used for export into various formats (e.g. JSON)
        If a descendant class needs to update the dictionary in some way
        (put some more information), override the _update_dict method.

        Returns
        -------
        collections.OrderedDict
        """
        from collections import OrderedDict
        result = OrderedDict()
        result["histogram_type"] = type(self).__name__
        result["binnings"] = [binning.to_dict() for binning in self._binnings]
        result["frequencies"] = self.frequencies.tolist()
        result["dtype"] = str(np.dtype(self.dtype))

        # TODO: Optimize for _errors == _frequencies
        result["errors2"] = self.errors2.tolist()
        result["meta_data"] = self._meta_data
        result["missed"] = self._missed.tolist()
        result["missed_keep"] = self.keep_missed
        self._update_dict(result)
        return result

    def _update_dict(self, a_dict):
        """Update the dictionary for export.

        Override if you want to customize the process.

        Parameters
        ----------
        a_dict : dict
            Dictionary exported by the default implementation of to_dict
        """
        pass

    @classmethod
    def _from_dict_kwargs(cls, a_dict):
        """Modify __init__ arguments from an external dictionary.

        Template method for from dict.
        Override if necessary (like it's done in Histogram1D).

        Parameters
        ----------
        a_dict : dict

        Returns
        -------
        dict
        """
        from .binnings import BinningBase
        kwargs = {
            "binnings": [BinningBase.from_dict(binning_data) for binning_data in a_dict["binnings"]],
            "dtype": np.dtype(a_dict["dtype"]),
            "frequencies": a_dict.get("frequencies"),
            "errors2": a_dict.get("errors2"),
        }
        if "missed" in a_dict:
            kwargs["missed"] = a_dict["missed"]
        kwargs.update(a_dict.get("meta_data", {}))
        if len(kwargs["binnings"]) > 2:
            kwargs["dimension"] = len(kwargs["binnings"])
        return kwargs

    @classmethod
    def from_dict(cls, a_dict):
        """Create an instance from a dictionary.

        If customization is necessary, override the _from_dict_kwargs
        template method, not this one.

        Parameters
        ----------
        a_dict : dict

        Returns
        -------
        HistogramBase
        """
        kwargs = cls._from_dict_kwargs(a_dict)
        return cls(**kwargs)

    def to_json(self, path=None, **kwargs):
        """Convert to JSON representation.

        Parameters
        ----------
        path: Optional[str]
            Where to write the JSON.

        Returns
        -------
        str:
            The JSON representation.
        """
        from .io import save_json
        return save_json(self, path, **kwargs)

    def __repr__(self):
        if self.name:
            result = "{0}('{4}', bins={1}, total={2}, dtype={3})".format(
                self.__class__.__name__, self.shape, self.total, self.dtype, self.name)
        else:
            result = "{0}(bins={1}, total={2}, dtype={3})".format(
                self.__class__.__name__, self.shape, self.total, self.dtype)
        return result

    def __add__(self, other):
        new = self.copy()
        new += other
        new._meta_data = self._merge_meta_data(self, other)
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
                    self._change_binning(new_bins, map1, axis=i)
                    other._change_binning(new_bins, map2, axis=i)
                self._frequencies += other.frequencies
                self._errors2 += other.errors2

            except:
                raise  # RuntimeError("Cannot find common binning for added histograms.")
        else:
            raise RuntimeError("Incompatible binning")

        if self._stats and other._stats:
            for key in self._stats:
                self._stats[key] += other._stats[key]
        return self

    def __sub__(self, other):
        new = self.copy()
        new -= other
        new._meta_data = self._merge_meta_data(self, other)
        return new

    def __isub__(self, other):
        import warnings
        warnings.warn("Subtracting histograms is considered to be a bad idea.")
        return self.__iadd__(other * (-1))

    def __mul__(self, other):
        new = self.copy()
        new *= other
        return new

    def __imul__(self, other):
        if not np.isscalar(other):
            raise RuntimeError("Histograms may be multiplied only by a constant.")
        if np.issubdtype(self.dtype, np.integer) and np.issubdtype(type(other), np.floating):
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

    def __lshift__(self, value):
        """Convenience alias for fill.

        Because of the limit to argument count, weight is not supported.
        """
        self.fill(value)

    @classmethod
    def _merge_meta_data(cls, first, second):
        """Merge meta data of two histograms leaving only the equal values.

        (Used in addition and subtraction)
        """
        keys = set(first._meta_data.keys())
        keys = keys.union(set(second._meta_data.keys()))
        return {key:
                (first._meta_data.get(key, None) if first._meta_data.get(key, None) == second._meta_data.get(key, None) else None)
                for key in keys}

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
