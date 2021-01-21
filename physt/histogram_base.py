"""HistogramBase - base for all histogram classes."""
import abc
import warnings
from typing import (
    Dict,
    List,
    Optional,
    Iterable,
    Mapping,
    Any,
    Tuple,
    TYPE_CHECKING,
    TypeVar,
    cast,
    Union,
)

import numpy as np

from physt.binnings import as_binning, BinningLike, BinningBase
from physt.config import config
from physt.typing_aliases import Axis, ArrayLike, DtypeLike

if TYPE_CHECKING:
    import physt

    HistogramType = TypeVar("HistogramType", bound="HistogramBase")


class HistogramBase(abc.ABC):
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
    _binnings : Schema for binning(s)
    frequencies : np.ndarray
        Bin contents
    errors2 : np.ndarray
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
    (if you set config.free_arithmetics (see below), negative frequencies are also
    allowed.

    Arithmetics
    -----------
    Histograms offer standard arithmetic operators that by default allow only
    meaningful application (i.e. addition / subtraction of two histograms
    with matching or mutually adaptable bin sets, multiplication and division by a constant).

    If you relax the criteria by setting `config.free_aritmetics` or inside
    the config.enable_free_arithmetics() context manager, you are in addition
    allowed to use any array-like with matching shape.

    See Also
    --------
    histogram1d
    histogram_nd
    special

    """

    def __init__(
        self,
        binnings: Iterable[BinningLike],
        frequencies: Optional[ArrayLike] = None,
        errors2: Optional[ArrayLike] = None,
        *,
        axis_names: Optional[Iterable[str]] = None,
        dtype: Optional[DtypeLike] = None,
        keep_missed: bool = True,
        **kwargs,
    ):
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

        new_kwargs = self.default_init_values.copy()
        new_kwargs.update(kwargs)
        kwargs = new_kwargs

        # Frequencies + appropriate dtypes
        if frequencies is None:
            dtype = dtype or np.int64
            self._frequencies = np.zeros(self.shape, dtype=dtype)
        else:
            if dtype is not None:
                frequencies = np.asarray(frequencies, dtype=dtype)
            else:
                frequencies = np.asarray(frequencies)
                if np.issubdtype(frequencies.dtype, np.integer):
                    frequencies = frequencies.astype(np.int64)
                elif np.issubdtype(frequencies.dtype, np.floating):
                    frequencies = frequencies.astype(np.float64)
                else:
                    raise RuntimeError(
                        "Frequencies of type {0} not understood".format(frequencies.dtype)
                    )
            dtype = frequencies.dtype
            self.frequencies = frequencies
        self._dtype, _ = self._eval_dtype(dtype)  # type: ignore

        # Errors
        if errors2 is None:
            errors2 = abs(self._frequencies.copy())
        else:
            errors2 = np.asarray(errors2, dtype=self.dtype)
        self.errors2 = errors2

        self.keep_missed = keep_missed
        # Note: missed are dealt differently in 1D/ND cases

        self._meta_data = kwargs.copy()
        self.axis_names = tuple(axis_names or self.default_axis_names)

    # "Protected" attributes
    _binnings: List[BinningBase]
    _frequencies: np.ndarray
    _errors2: np.ndarray
    _missed: np.ndarray

    @property
    def default_axis_names(self) -> List[str]:
        """Axis names to be used when an instance does not define them."""
        return [f"axis{i}" for i in range(self.ndim)]

    default_init_values: Dict[str, Any] = {}

    @property
    def meta_data(self) -> Dict[str, Any]:
        """A dictionary of non-numerical information about the histogram.

        It contains several pre-defined ones, but you can add any other.
        These are preserved when saving and also in operations.
        """
        return self._meta_data

    @property
    def name(self) -> Optional[str]:
        """Name of the histogram (stored in meta-data)."""
        return self._meta_data.get("name", None)

    @name.setter
    def name(self, value: str):
        """Name of the histogram.

        In plotting, this will be used as label.
        """
        self._meta_data["name"] = str(value)

    @property
    def title(self) -> Optional[str]:
        """Title of the histogram to be displayed when plotted (stored in meta-data).

        If not specified, defaults to `name`.
        """
        return self._meta_data.get("title", self.name)

    @title.setter
    def title(self, value: str):
        """Title of the histogram.

        In plotting, this will be used as plot title.
        """
        self._meta_data["title"] = str(value)

    @property
    def axis_names(self) -> Tuple[str, ...]:
        """Names of axes (stored in meta-data)."""
        default = ["axis{0}".format(i) for i in range(self.ndim)]
        return tuple(self._meta_data.get("axis_names", None) or default)

    @axis_names.setter
    def axis_names(self, value: Iterable[str]):
        self._meta_data["axis_names"] = tuple(str(name) for name in value)

    def _get_axis(self, name_or_index: Axis) -> int:
        """Get a zero-based index of an axis and check its existence."""
        # TODO: Add unit test
        if isinstance(name_or_index, int):
            if name_or_index < 0 or name_or_index >= self.ndim:
                raise ValueError("No such axis, must be from 0 to {0}".format(self.ndim - 1))
            return name_or_index
        if isinstance(name_or_index, str):
            if name_or_index not in self.axis_names:
                named_axes = [name for name in self.axis_names if name]
                raise ValueError(
                    f"No axis with such name: {name_or_index}, available names: "
                    + ", ".join((named_axes))
                    + "In most places, you can also use numbers."
                )
            return self.axis_names.index(name_or_index)
        raise TypeError(
            f"Argument of type {type(name_or_index)} not understood, int or str expected."
        )

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of histogram's data.

        Returns
        -------
        Tuple with the number of bins along each axis.
        """
        return tuple(bins.bin_count for bins in self._binnings)

    @property
    def ndim(self) -> int:
        """Dimensionality of histogram's data.

        i.e. the number of axes along which we bin the values.
        """
        return len(self._binnings)

    @classmethod
    def _eval_dtype(cls, value: DtypeLike) -> Tuple[np.dtype, np.iinfo]:
        """Convert dtype into canonical form, check its applicability and return info.

        Parameters
        ----------
        value: Anything convertible to dtype

        Returns
        -------
        value: Numpy dtype
        type_info: Information about the dtype
        """
        dtype: np.dtype = np.dtype(value)
        if dtype.kind in "iu":
            type_info = np.iinfo(dtype)
        elif dtype.kind == "f":
            type_info = np.finfo(dtype)
        else:
            raise ValueError("Unsupported dtype. Only integer/floating-point types are supported.")
        return dtype, type_info

    @property
    def dtype(self) -> np.dtype:
        """Data type of the bin contents."""
        return self._dtype

    @dtype.setter
    def dtype(self, value: DtypeLike) -> None:
        self.set_dtype(value)

    def set_dtype(self, value: DtypeLike, *, check: bool = True) -> None:
        """Change data type of the bin contents.

        Allowed conversions:
        - from integral to float types
        - between the same category of type (float/integer)
        - from float types to integer if weights are trivial

        Parameters
        ----------
        value: np.dtype or something convertible to it.
        check: If True (default), all values are checked against the limits
        """
        # TODO? Deal with unsigned types
        value, type_info = self._eval_dtype(value)
        if value == self._dtype:
            return

        if self.dtype is None or np.can_cast(self.dtype, value):
            pass  # Ok
        elif check:
            if np.issubdtype(value, np.integer):
                if self.dtype.kind == "f":
                    for array in (self.frequencies, self.errors2):
                        if np.any(array % 1.0):
                            raise ValueError("Data contain non-integer values.")
            for array in (self.frequencies, self.errors2):
                if np.any((array > type_info.max) | (array < type_info.min)):
                    raise ValueError("Data contain values outside the specified range.")

        self._dtype = value
        self._frequencies = self._frequencies.astype(value)
        if self._errors2 is not None:
            self._errors2 = self._errors2.astype(value)
        if self._missed is not None:
            self._missed = self._missed.astype(value)

    def _coerce_dtype(self, other_dtype: DtypeLike) -> None:
        """Possibly change the bin content type to allow correct operations with other operand.

        Parameters
        ----------
        other_dtype : np.dtype or type
        """
        other_dtype, _ = self._eval_dtype(other_dtype)
        if self._dtype is None:
            new_dtype = np.dtype(other_dtype)
        else:
            new_dtype = np.find_common_type([self._dtype, np.dtype(other_dtype)], [])
        if new_dtype != self.dtype:
            self.set_dtype(new_dtype)

    @property
    def bin_count(self) -> int:
        """Total number of bins."""
        return int(np.product(self.shape))

    @property
    def frequencies(self) -> np.ndarray:
        """Frequencies (values, contents) of the histogram bins."""
        return self._frequencies

    @frequencies.setter
    def frequencies(self, values: ArrayLike) -> None:
        frequencies = np.asarray(values)
        if frequencies.shape != self.shape:
            raise ValueError("Values must have same dimension as bins.")
        if np.any(frequencies < 0):
            if config.free_arithmetics:
                warnings.warn("Negative frequencies in the histogram.")
            else:
                raise ValueError("Cannot have negative frequencies.")
        self._frequencies = frequencies

    @property
    def densities(self) -> np.ndarray:
        """Frequencies normalized by bin sizes.

        Useful when bins are not of the same size.
        """
        return self._frequencies / self.bin_sizes

    @property
    @abc.abstractmethod
    def bin_sizes(self) -> np.ndarray:
        raise NotImplementedError

    def normalize(self, inplace: bool = False, percent: bool = False) -> "HistogramBase":
        """Normalize the histogram, so that the total weight is equal to 1.

        Parameters
        ----------
        inplace: If True, updates itself. If False (default), returns copy
        percent: If True, normalizes to percent instead of 1. Default: False

        Returns
        -------
        either modified copy or self

        See also
        --------
        densities
        HistogramND.partial_normalize
        """
        if inplace:
            self /= self.total * (0.01 if percent else 1)
            return self
        else:
            return self / self.total * (100 if percent else 1)

    @property
    def errors2(self) -> np.ndarray:
        """Squares of the bin errors."""
        return self._errors2

    @errors2.setter
    def errors2(self, values: ArrayLike) -> None:
        array: np.ndarray = np.asarray(values)
        if array.shape != self.shape:
            raise ValueError("Square errors must have same dimension as bins.")
        if np.any(array < 0):
            raise ValueError("Cannot have negative square errors.")
        self._errors2 = array

    @property
    def errors(self) -> np.ndarray:
        """Bin errors."""
        return np.sqrt(self.errors2)

    @property
    def total(self) -> float:
        """Total number (sum of weights) of entries excluding underflow and overflow."""
        return self._frequencies.sum()

    @property
    def missed(self) -> float:
        """Total number (weight) of entries that missed the bins."""
        return self._missed.sum()

    def is_adaptive(self) -> bool:
        """Whether the binning can be changed with operations."""
        # TODO: remove in favour of adaptive property
        return all(binning.is_adaptive() for binning in self._binnings)

    def set_adaptive(self, value: bool = True):
        """Change the histogram binning to (non)adaptive.

        This requires binning in all dimensions to allow this.
        """
        # TODO: remove in favour of adaptive property
        if not all(b.adaptive_allowed for b in self._binnings):
            raise RuntimeError("All binnings must allow adaptive behaviour.")
        for binning in self._binnings:
            binning.set_adaptive(value)

    @property
    def adaptive(self) -> bool:
        # TODO: Remove?
        return self.is_adaptive()

    @adaptive.setter
    def adaptive(self, value: bool):
        self.set_adaptive(value)

    def _change_binning(
        self,
        new_binning: BinningBase,
        bin_map: Iterable[Tuple[int, int]],
        axis: Axis = 0,
    ):
        """Set new binnning and update the bin contents according to a map.

        Fills frequencies and errors with 0.
        It's the caller's responsibility to provide correct binning and map.

        Parameters
        ----------
        new_binning: physt.binnings.BinningBase
        bin_map: Tuples containing bin indices (old, new)
        axis: What axis does the binning describe(0..ndim-1)
        """
        axis = self._get_axis(axis)
        self._reshape_data(new_binning.bin_count, bin_map, axis)
        self._binnings[axis] = new_binning

    def merge_bins(
        self: "HistogramType",
        amount: Optional[int] = None,
        *,
        min_frequency: Optional[float] = None,
        axis: Optional[Axis] = None,
        inplace: bool = False,
    ) -> "HistogramType":
        """Reduce the number of bins and add their content:

        Parameters
        ----------
        amount: How many adjacent bins to join together.
        min_frequency: Try to have at least this value in each bin
            (this is not enforce e.g. for minima between high bins)
        axis: On which axis to do this (None => all)
        inplace: Whether to modify this histogram or return a new one
        """
        if not inplace:
            histogram = self.copy()
            histogram.merge_bins(amount, min_frequency=min_frequency, axis=axis, inplace=True)
            return histogram
        elif axis is None:
            for i in range(self.ndim):
                self.merge_bins(amount=amount, min_frequency=min_frequency, axis=i, inplace=True)
        else:
            axis = self._get_axis(axis)
            if amount is not None:
                if not amount == int(amount):
                    raise RuntimeError("Amount must be integer")
                bin_map = [(i, i // amount) for i in range(self.shape[axis])]
            elif min_frequency is not None:
                if self.ndim == 1:
                    check = self.frequencies
                else:
                    # TODO: Check this!
                    from physt.histogram_nd import HistogramND

                    check = cast(HistogramND, self).projection(axis).frequencies
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
        return self

    def _reshape_data(self, new_size: int, bin_map, axis: int = 0):
        """Reshape data to match new binning schema.

        Fills frequencies and errors with 0.

        Parameters
        ----------
        new_size: New size along the axis
        bin_map: Iterable[(old, new)] or int or None
            If None, we can keep the data unchanged.
            If int, it is offset by which to shift the data (can be 0)
            If iterable, pairs specify which old bin should go into which new bin
        axis: On which axis to apply
        """
        if bin_map is None:
            return

        new_shape = list(self.shape)
        new_shape[axis] = new_size
        new_frequencies = np.zeros(new_shape, dtype=self._frequencies.dtype)
        new_errors2 = np.zeros(new_shape, dtype=self._frequencies.dtype)
        self._apply_bin_map(
            old_frequencies=self._frequencies,
            new_frequencies=new_frequencies,
            old_errors2=self._errors2,
            new_errors2=new_errors2,
            bin_map=bin_map,
            axis=axis,
        )
        self._frequencies = new_frequencies
        self._errors2 = new_errors2

    def _apply_bin_map(
        self,
        old_frequencies: np.ndarray,
        new_frequencies: np.ndarray,
        old_errors2: np.ndarray,
        new_errors2: np.ndarray,
        bin_map: Union[Iterable[Tuple[int, int]], int],
        axis: int,
    ):
        """Fill new data arrays using a map.

        Parameters
        ----------
        old_frequencies : Source of frequencies data
        new_frequencies : Target of frequencies data
        old_errors2 : Source of errors data
        new_errors2 : Target of errors data
        bin_map: Iterable[(old, new)] or int or None
            As in _reshape_data
        axis: On which axis to apply

        See also
        --------
        HistogramBase._reshape_data
        """
        if old_frequencies is not None and old_frequencies.shape[axis] > 0:
            if isinstance(bin_map, int):
                new_index: List[Union[int, slice]] = [slice(None) for i in range(self.ndim)]
                new_index[axis] = slice(bin_map, bin_map + old_frequencies.shape[axis])
                new_frequencies[tuple(new_index)] += old_frequencies
                new_errors2[tuple(new_index)] += old_errors2
            else:
                for (old, new) in bin_map:  # Generic enough
                    new_index = [slice(None) for i in range(self.ndim)]
                    new_index[axis] = new
                    old_index: List[Union[int, slice]] = [slice(None) for i in range(self.ndim)]
                    old_index[axis] = old
                    new_frequencies[tuple(new_index)] += old_frequencies[tuple(old_index)]
                    new_errors2[tuple(new_index)] += old_errors2[tuple(old_index)]

    def has_same_bins(self, other: "HistogramBase") -> bool:
        """Whether two histograms share the same binning."""
        if self.shape != other.shape:
            return False
        elif self.ndim == 1:
            return np.allclose(self.bins, other.bins)
        for i in range(self.ndim):
            if not np.allclose(self.bins[i], other.bins[i]):
                return False
        return True

    def copy(self: "HistogramType", *, include_frequencies: bool = True) -> "HistogramType":
        """Copy the histogram.

        Parameters
        ----------
        include_frequencies : If false, all frequencies are set to zero.
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

    @abc.abstractmethod
    def select(self, axis: Axis, index: Union[int, slice], *, force_copy: bool = False) -> Any:
        """Select in an axis.

        Parameters
        ----------
        axis: Axis, in which we select.
        index: Index of bin (as in numpy).
        force_copy: If True, identity slice force a copy to be made.
        """

    @property
    def binnings(self) -> List[BinningBase]:
        """The binnings.

        Note: Please, do not try to update the objects themselves.
        """
        return self._binnings

    @property
    @abc.abstractmethod
    def bins(self):
        ...

    @abc.abstractmethod
    def find_bin(self, value: ArrayLike, axis: Optional[Axis] = None) -> Union[None, int, Tuple[int, ...]]:
        """Index(-ices) of bin corresponding to a value.

        Parameters
        ----------
        value: Value with dimensionality equal to histogram
        axis: If set, find axis along an axis. Otherwise, find bins along all axes.
            None = outside the bins

        Returns
        -------
        If axis is specified (or the histogram is 1D), a number. Otherwise, a tuple. If not available, None.
        """

    @abc.abstractmethod
    def fill(self, value: float, weight: float = 1, **kwargs) -> Union[None, int, Tuple[int, ...]]:
        """Update histogram with a new value.

        It is an in-place operation.

        Parameters
        ----------
        value: Value to be added. Can be scalar or array depending on the histogram type.
        weight: Weight of the value

        Note
        ----
        May change the dtype if weight is set
        """
        # TODO: Perhaps it should just return None?
        ...

    @abc.abstractmethod
    def fill_n(
        self, values: ArrayLike, weights: Optional[ArrayLike] = None, *, dropna: bool = True
    ):
        """Update histogram with more values at once.

        It is an in-place operation.

        Parameters
        ----------
        values: Values to add
        weights: Optional weights to assign to each value
        drop_na: If true (default), all nan's are skipped.

        Note
        ----
        This method should be overloaded with a more efficient one.

        May change the dtype if weight is set.
        """
        ...

    @property
    def plot(self) -> "physt.plotting.PlottingProxy":
        """Proxy to plotting.

        This attribute is a special proxy to plotting. In the most
        simple cases, it can be used as a method. For more sophisticated
        use, see the documentation for physt.plotting package.
        """
        from .plotting import PlottingProxy

        return PlottingProxy(self)

    def to_dict(self) -> Dict[str, Any]:
        """Dictionary with all data in the histogram.

        This is used for export into various formats (e.g. JSON)
        If a descendant class needs to update the dictionary in some way
        (put some more information), override the _update_dict method.
        """
        result: Dict[str, Any] = dict()
        result["histogram_type"] = type(self).__name__
        result["binnings"] = [binning.to_dict() for binning in self._binnings]
        if self.frequencies is not None:
            result["frequencies"] = self.frequencies.tolist()
        else:
            result["frequencies"] = None
        result["dtype"] = str(np.dtype(self.dtype))

        # TODO: Optimize for _errors == _frequencies
        result["errors2"] = self.errors2.tolist()
        result["meta_data"] = self._meta_data
        result["missed"] = self._missed.tolist()
        result["missed_keep"] = self.keep_missed
        self._update_dict(result)
        return result

    def _update_dict(self, a_dict: Dict[str, Any]) -> None:
        """Update the dictionary for export.

        Override if you want to customize the process.

        Parameters
        ----------
        a_dict : Dictionary exported by the default implementation of to_dict
        """
        pass

    @classmethod
    def _kwargs_from_dict(cls, a_dict: Mapping[str, Any]) -> Dict[str, Any]:
        """Modify __init__ arguments from an external dictionary.

        Template method for from dict.
        Override if necessary (like it's done in Histogram1D).
        """
        kwargs = {
            "binnings": [
                BinningBase.from_dict(binning_data) for binning_data in a_dict["binnings"]
            ],
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
    def from_dict(cls, a_dict: Mapping[str, Any]) -> "HistogramBase":
        """Create an instance from a dictionary.

        If customization is necessary, override the _from_dict_kwargs
        template method, not this one.
        """
        kwargs = cls._kwargs_from_dict(a_dict)
        return cls(**kwargs)

    def to_json(self, path: Optional[str] = None, **kwargs) -> str:
        """Convert to JSON representation.

        Parameters
        ----------
        path: Where to write the JSON.

        Returns
        -------
        The JSON representation.
        """
        from .io import save_json

        return save_json(self, path, **kwargs)

    def __repr__(self):
        if self.name:
            result = "{0}('{4}', bins={1}, total={2}, dtype={3})".format(
                self.__class__.__name__, self.shape, self.total, self.dtype, self.name
            )
        else:
            result = "{0}(bins={1}, total={2}, dtype={3})".format(
                self.__class__.__name__, self.shape, self.total, self.dtype
            )
        return result

    def __add__(self, other):
        new = self.copy()
        new += other
        if isinstance(other, HistogramBase):
            new._meta_data = self._merge_meta_data(self, other)
        return new

    def __radd__(self, other):
        if other == 0:  # Enable sum()
            return self
        else:
            return self + other

    def __iadd__(self, other):
        if isinstance(other, HistogramBase):
            if other.ndim != self.ndim:
                raise ValueError("Cannot add histograms with different dimensions.")
            if self.has_same_bins(other):
                # print("Has same!!!!!!!!!!")
                self._coerce_dtype(other.dtype)
                self.frequencies = self.frequencies + other.frequencies
                self.errors2 = self.errors2 + other.errors2
                self._missed += other._missed
            elif self.is_adaptive():
                if other.missed > 0:
                    raise ValueError("Cannot adapt histogram with missed values.")

                other = other.copy()
                other.set_adaptive(True)

                self._coerce_dtype(other.dtype)

                for i in range(self.ndim):
                    new_bins = self._binnings[i].copy()

                    map1, map2 = new_bins.adapt(other._binnings[i])
                    self._change_binning(new_bins, map1, axis=i)
                    other._change_binning(new_bins, map2, axis=i)
                self.frequencies = self.frequencies + other.frequencies
                self.errors2 = self.errors2 + other.errors2
            else:
                raise ValueError("Incompatible binning")

            if self._stats and other._stats:
                for key in self._stats:
                    self._stats[key] += other._stats[key]
        elif config.free_arithmetics:
            array = np.asarray(other)
            self._coerce_dtype(array.dtype)
            self.frequencies = self.frequencies + array
            self.errors2 = self.errors2 + abs(array)
            self._missed = self._missed * np.nan  # TODO: Any reasonable interpretation?
            self._stats = None  # TODO: Any reasonable interpretation?
        else:
            raise TypeError(f"Only histograms can be added together. {type(other)} found instead.")
        return self

    def __sub__(self, other):
        new = self.copy()
        new -= other
        if isinstance(other, HistogramBase):
            new._meta_data = self._merge_meta_data(self, other)
        return new

    def __isub__(self, other):
        warnings.warn("Subtracting histograms is considered to be a bad idea.")
        if isinstance(other, HistogramBase):
            if config.free_arithmetics:
                self += other * (-1)
            else:
                adapted_self = self + 0 * other
                adapted_other = 0 * self + other
                self.frequencies = adapted_self.frequencies - adapted_other.frequencies
                self.errors2 = adapted_self.errors2 + adapted_other.errors2
                self._missed -= other._missed
            self._stats = None
            return self
        array = np.asarray(other)
        return self.__iadd__(array * (-1))

    def __mul__(self, other: Any):
        new = self.copy()
        new *= other
        return new

    def __imul__(self, other: Any):
        if isinstance(other, HistogramBase):
            raise TypeError("Multiplication of two histograms is not supported.")
        elif np.isscalar(other):
            array = np.asarray(other)
            try:
                self._coerce_dtype(array.dtype)
            except ValueError as v:
                raise TypeError(str(v)) from v
            self.frequencies = self.frequencies * other
            self.errors2 = self.errors2 * other ** 2
            self._missed = self._missed * other
            if self._stats:
                self._stats["sum"] *= other
                self._stats["sum2"] *= other ** 2
        elif config.free_arithmetics:  # Treat other as array-like
            array = np.asarray(other)
            self._coerce_dtype(array.dtype)
            self.frequencies = self.frequencies * array
            self.errors2 = self.errors2 * array ** 2
            self._stats = None
            self._missed = self._missed * np.nan
        else:
            raise TypeError("Histograms may be multiplied only by a constant.")
        return self

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        new = self.copy()
        new /= other
        return new

    def __itruediv__(self, other):
        if isinstance(other, HistogramBase):
            raise TypeError("Division of two histograms is not supported.")
        elif np.isscalar(other):
            self._coerce_dtype(np.float64)
            self.frequencies = self.frequencies / other
            self.errors2 = self.errors2 / other ** 2
            self._missed /= other
            if self._stats:
                self._stats["sum"] *= other
                self._stats["sum2"] *= other ** 2
        elif config.free_arithmetics:  # Treat other as array-like
            self._coerce_dtype(np.float64)
            array = np.asarray(other)
            self.frequencies = self.frequencies / array
            self.errors2 = self.errors2 / array ** 2
            self._stats = None
            self._missed /= np.nan
        else:
            raise TypeError("Histograms may be divided only by a constant.")
        return self

    def __lshift__(self, value):
        """Convenience alias for fill.

        Because of the limit to argument count, weight is not supported.
        """
        self.fill(value)

    @classmethod
    def _merge_meta_data(cls, first: "HistogramBase", second: "HistogramBase") -> dict:
        """Merge meta data of two histograms leaving only the equal values.

        (Used in addition and subtraction)
        """
        keys = set(first._meta_data.keys())
        keys = keys.union(set(second._meta_data.keys()))
        return {
            key: (
                first._meta_data.get(key, None)
                if first._meta_data.get(key, None) == second._meta_data.get(key, None)
                else None
            )
            for key in keys
        }

    def __array__(self) -> np.ndarray:
        """Convert to numpy array.

        Returns
        -------
        The array of frequencies

        See also
        --------
        frequencies
        """
        return self.frequencies
