"""Multi-dimensional histograms."""
import warnings
from numbers import Number
from typing import Optional, List, Any, Sequence, Tuple, Union, Iterable, cast, overload

import numpy as np

from physt.histogram_base import HistogramBase, Axis
from physt.binnings import BinningBase, BinningLike
from physt.typing_aliases import ArrayLike, DTypeLike


class HistogramND(HistogramBase):
    """Multi-dimensional histogram data.

    Attributes
    ----------

    """

    def __init__(
        self,
        binnings: Iterable[BinningLike],
        frequencies: Optional[ArrayLike] = None,
        *,
        dimension: Optional[int] = None,
        axis_names: Optional[Iterable[str]] = None,
        missed=0,
        **kwargs,
    ):
        """Constructor

        Parameters
        ----------
        dimension: int

        binnings: The binnings for all axes.
        frequencies: The bin contents.
        errors2: Optional[array_like]
            Quadratic errors of individual bins. If not set, defaults to frequencies.
        keep_missed: bool
        missed: int or float (dtype?)
        name: Optional[str]
        """

        # Bins + checks
        binnings = list(binnings)
        if dimension:
            if len(binnings) != dimension:
                raise ValueError(
                    f"bins must be a sequence of {dimension} schemas, {len(binnings)} found."
                )

        HistogramBase.__init__(self, binnings, frequencies, axis_names=axis_names, **kwargs)

        if len(self.axis_names) != self.ndim:
            raise ValueError(
                f"The length of axis names ({len(self.axis_names)}) must be equal to histogram dimension ({self.ndim})."
            )

        # Missed values
        self._missed = np.array([missed], dtype=self.dtype)

    # Not supported yet
    _stats = None

    @property
    def bins(self) -> List[np.ndarray]:
        """List of bin matrices."""
        return [binning.bins for binning in self._binnings]

    @property
    def edges(self) -> List[np.ndarray]:
        return [binning.numpy_bins for binning in self._binnings]

    @property
    def numpy_bins(self) -> List[np.ndarray]:
        """Numpy-like bins (if available)."""
        warnings.warn(
            "`numpy_bins` is deprecated, use `edges` instead",
            DeprecationWarning,
        )
        return self.edges

    @property
    def numpy_like(self) -> Tuple:
        """Same result as would the numpy.histogram function return."""
        return self.frequencies, self.numpy_bins

    def select(
        self, axis: Axis, index: Union[int, slice], *, force_copy: bool = False
    ) -> HistogramBase:

        # TODO: Implement mask?

        if index == slice(None) and not force_copy:
            return self

        axis_id = self._get_axis(axis)
        array_index: List[Union[int, slice]] = [slice(None, None, None) for i in range(self.ndim)]
        array_index[axis_id] = index

        frequencies = self._frequencies[tuple(array_index)].copy()
        errors2 = self._errors2[tuple(array_index)].copy()

        if isinstance(index, int):
            return self._reduce_dimension(
                [ax for ax in range(self.ndim) if ax != axis_id], frequencies, errors2
            )
        if isinstance(index, slice):
            if index.step is not None and index.step < 0:
                raise IndexError("Cannot change the order of bins")
            copy = self.copy()
            copy._frequencies = frequencies
            copy._errors2 = errors2
            copy._binnings[axis_id] = self._binnings[axis_id][index]
            return copy
        raise TypeError("Invalid index.")

    def __getitem__(
        self, index: Union[int, slice, Iterable[int]]
    ) -> Union["HistogramBase", Tuple[Tuple[Tuple[int, int], ...], float]]:
        """Select subset of histogram.

        Parameters
        ----------
        index: One or more indices to select in subsequent axes.

        Returns
        -------
        Depending on the parameters, a sub-histogram or content of one bin are returned.

        Indexing shares semantics with Numpy arrays, however

        Always returns a new object.
        """
        # TODO: Enable views
        if isinstance(index, (int, slice)):
            return self.select(0, index)
        if isinstance(index, tuple):
            if len(index) > self.ndim:
                raise IndexError(
                    f"Too many indices ({len(index)}) to select from {self.ndim}D histogram"
                )

            # Scalar case => return (bin edges), (frequency)
            if len(index) == self.ndim and all((isinstance(i, int) for i in index)):
                return (
                    tuple(
                        (self.get_bin_left_edges(i)[j], self.get_bin_right_edges(i)[j])
                        for i, j in enumerate(index)
                    ),
                    self._frequencies[index],
                )
            current: Any = self
            for i, subindex in enumerate(index):
                current = current.select(i + current.ndim - self.ndim, subindex, force_copy=False)
            if current is self:
                current = current.copy()
            return current
        raise TypeError("Invalid index.")

    # Missing: cumulative_frequencies - does it make sense?

    @overload
    def get_bin_widths(self, axis: Axis) -> np.ndarray:
        ...

    @overload
    def get_bin_widths(self, axis: None = ...) -> Sequence[np.ndarray]:
        ...

    def get_bin_widths(
        self, axis: Optional[Axis] = None
    ) -> Union[np.ndarray, Sequence[np.ndarray]]:  # TODO: -> Base ?
        if axis is not None:
            axis = self._get_axis(axis)
            return self.get_bin_right_edges(axis) - self.get_bin_left_edges(axis)
        else:
            return np.meshgrid(*[self.get_bin_widths(i) for i in range(self.ndim)], indexing="ij")

    @property
    def bin_sizes(self) -> np.ndarray:
        # TODO: Some kind of caching?
        sizes = self.get_bin_widths(0)
        for i in range(1, self.ndim):
            sizes = np.multiply.outer(sizes, self.get_bin_widths(i))
        return sizes

    @property
    def total_size(self) -> float:
        """The total size of the bin space.

        Note
        ----
        Perhaps not optimized, but should work also with transformed axes
        """
        return float(np.sum(self.bin_sizes))

    @overload
    def get_bin_edges(self, axis: Axis) -> np.ndarray:
        ...

    @overload
    def get_bin_edges(self, axis: None = ...) -> Sequence[np.ndarray]:
        ...

    def get_bin_edges(self, axis: Optional[Axis] = None) -> Union[np.ndarray, Sequence[np.ndarray]]:
        if axis is not None:
            axis = self._get_axis(axis)
            return self.edges[self._get_axis(axis)]
        else:
            edges = [self.get_bin_edges(i) for i in range(self.ndim)]
            return np.meshgrid(*edges, indexing="ij")

    @overload
    def get_bin_left_edges(self, axis: Axis) -> np.ndarray:
        ...

    @overload
    def get_bin_left_edges(self, axis: None = ...) -> Sequence[np.ndarray]:
        ...

    def get_bin_left_edges(
        self, axis: Optional[Axis] = None
    ) -> Union[np.ndarray, Sequence[np.ndarray]]:
        if axis is not None:
            axis = self._get_axis(axis)
            return self.bins[axis][:, 0]
        edges = [self.get_bin_left_edges(i) for i in range(self.ndim)]
        return np.meshgrid(*edges, indexing="ij")

    @overload
    def get_bin_right_edges(self, axis: Axis) -> np.ndarray:
        ...

    @overload
    def get_bin_right_edges(self, axis: None = ...) -> Sequence[np.ndarray]:
        ...

    def get_bin_right_edges(
        self, axis: Optional[Axis] = None
    ) -> Union[np.ndarray, Sequence[np.ndarray]]:
        if axis is not None:
            axis = self._get_axis(axis)
            return self.bins[axis][:, 1]
        edges = [self.get_bin_right_edges(i) for i in range(self.ndim)]
        return np.meshgrid(*edges, indexing="ij")

    @overload
    def get_bin_centers(self, axis: Axis) -> np.ndarray:
        ...

    @overload
    def get_bin_centers(self, axis: None = ...) -> Sequence[np.ndarray]:
        ...

    def get_bin_centers(
        self, axis: Optional[Axis] = None
    ) -> Union[np.ndarray, Sequence[np.ndarray]]:
        if axis is not None:
            axis = self._get_axis(axis)
            return (self.get_bin_right_edges(axis) + self.get_bin_left_edges(axis)) / 2
        return np.meshgrid(*[self.get_bin_centers(i) for i in range(self.ndim)], indexing="ij")

    # @overload
    # def find_bin(self, value: ArrayLike, axis: None) -> Optional[Tuple[int, ...]]: ...

    # @overload
    # def find_bin(self, value: Number, axis: Axis) -> Optional[int]: ...

    def find_bin(
        self, value: ArrayLike, axis: Optional[Axis] = None
    ) -> Union[None, int, Tuple[int, ...]]:
        """Index(-ices) of bin corresponding to a value.

        Parameters
        ----------
        value: Value with dimensionality equal to histogram.
        axis: If set, find axis along an axis. Otherwise, find bins along all axes.
            None = outside the bins

        Returns
        -------
        If axis is specified, a number. Otherwise, a tuple. If not available, None.
        """
        # TODO: Support multiple values?
        if axis is not None:
            if not isinstance(value, Number):
                raise TypeError(f"Number expected: {value!r}")
            value_scalar = cast(float, value)  # TODO: Does that work with scalar?
            axis = self._get_axis(axis)
            ixbin = np.searchsorted(self.get_bin_left_edges(axis), value_scalar, side="right")
            if ixbin == 0:
                return None
            if ixbin == self.shape[axis]:
                if value_scalar <= self.get_bin_right_edges(axis)[-1]:
                    return int(ixbin - 1)
                else:
                    return None
            if value_scalar < self.get_bin_right_edges(axis)[ixbin - 1]:
                return int(ixbin - 1)
            if ixbin == self.shape[axis]:
                return None
            return None

        else:
            if np.isscalar(value):
                raise TypeError(f"Array expected: {value!r}")
            value_array = np.asarray(value)
            if value_array.shape != (self.ndim,):
                raise ValueError(f"Wrong shape: {value_array.shape}, expected: ({self.ndim},)")
            ixbins = cast(
                Tuple[int, ...], tuple(self.find_bin(value_array[i], i) for i in range(self.ndim))
            )
            if None in ixbins:
                return None
            return ixbins

    def fill(self, value: ArrayLike, weight: float = 1, **kwargs):
        self._coerce_dtype(type(weight))
        value_array = np.asarray(value)
        for i, binning in enumerate(self._binnings):
            if binning.is_adaptive():
                bin_map = binning.force_bin_existence(value_array[i])
                self._reshape_data(binning.bin_count, bin_map, i)
        ixbin = self.find_bin(value_array, **kwargs)
        if ixbin is None and self.keep_missed:
            self._missed += weight
        else:
            self._frequencies[ixbin] += weight
            self._errors2[ixbin] += weight ** 2
        return ixbin

    def fill_n(
        self,
        values: ArrayLike,
        weights: Optional[ArrayLike] = None,
        *,
        dropna: bool = True,
        columns: bool = False,
    ):
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
        values_array = np.asarray(values)
        if values_array.ndim != 2:
            raise ValueError(f"Expecting 2D array of values, {values_array.ndim} found.")
        if columns:
            values_array = values_array.T
        if values_array.shape[1] != self.ndim:
            raise ValueError(
                f"Expecting array with {self.ndim} columns, {values_array.shape[1]} found."
            )
        if dropna:
            values_array = values_array[~np.isnan(values_array).any(axis=1)]
        if weights is not None:
            weights = np.asarray(weights)
            # TODO: Check for weights size?
            self._coerce_dtype(weights.dtype)
        for i, binning in enumerate(self._binnings):
            if binning.is_adaptive():
                bin_map = binning.force_bin_existence(values_array[:, i])  # TODO: Add to some test
                self._reshape_data(binning.bin_count, bin_map, i)
        frequencies, errors2, missed = calculate_frequencies(
            values_array, self._binnings, weights=weights
        )
        self._frequencies += frequencies
        self._errors2 += errors2 if errors2 is not None else frequencies
        self._missed[0] += missed

    def _get_projection_axes(self, *axes: Axis) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """Find axis identifiers for projection and all the remaining ones.

        Returns
        -------
        axes: axes to include in the projection
        invert: axes along which to reduce
        """
        axes_: List[int] = [self._get_axis(ax) for ax in axes]
        if not axes_:
            raise ValueError("No axis selected for projection")
        if len(axes_) != len(set(axes_)):
            raise ValueError("Duplicate axes in projection")
        invert = (i for i in range(self.ndim) if not i in axes_)
        return tuple(axes_), tuple(invert)

    def _reduce_dimension(self, axes, frequencies, errors2, **kwargs) -> HistogramBase:
        # TODO: document
        name = kwargs.pop("name", self.name)
        axis_names = [name for i, name in enumerate(self.axis_names) if i in axes]
        bins = [bins for i, bins in enumerate(self._binnings) if i in axes]
        if len(axes) == 1:
            from physt.histogram1d import Histogram1D

            klass = kwargs.get("type", Histogram1D)
            return klass(
                binning=bins[0],
                frequencies=frequencies,
                errors2=errors2,
                axis_name=axis_names[0],
                name=name,
            )
        elif len(axes) == 2:
            klass = kwargs.get("type", Histogram2D)
            return klass(
                binnings=bins,
                frequencies=frequencies,
                errors2=errors2,
                axis_names=axis_names,
                name=name,
            )
        else:
            klass = kwargs.get("type", HistogramND)
            return klass(
                dimension=len(axes),
                binnings=bins,
                frequencies=frequencies,
                errors2=errors2,
                axis_names=axis_names,
                name=name,
            )

    def accumulate(self, axis: Axis) -> HistogramBase:
        """Calculate cumulative frequencies along a certain axis.

        Returns
        -------
        new_hist: Histogram of the same type & size
        """
        # TODO: Merge with Histogram1D.cumulative_frequencies
        # TODO: Deal with errors and totals etc.
        # TODO: inplace
        new_one = self.copy()
        axis_id = self._get_axis(axis)
        new_one._frequencies = np.cumsum(new_one.frequencies, axis_id)
        return new_one

    def projection(self, *axes: Axis, **kwargs) -> HistogramBase:
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
        # TODO: rename to project in 0.5
        axes, invert = self._get_projection_axes(*axes)
        frequencies = self.frequencies.sum(axis=invert)
        errors2 = self.errors2.sum(axis=invert)
        return self._reduce_dimension(axes, frequencies, errors2, **kwargs)

    def __eq__(self, other: Any):
        """Equality comparison"""
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

    @classmethod
    def from_calculate_frequencies(cls, data, binnings, weights=None, *, dtype=None, **kwargs):
        frequencies, errors2, missing = calculate_frequencies(
            data=data, binnings=binnings, weights=weights, dtype=dtype
        )
        return cls(
            binnings=binnings,
            frequencies=frequencies,
            errors2=errors2,
            **kwargs,
        )


class Histogram2D(HistogramND):
    """Specialized 2D variant of the general HistogramND class.

    In contrast to general HistogramND, it is plottable.
    """

    def __init__(self, binnings, frequencies=None, **kwargs):
        kwargs.pop("dimension", None)
        super().__init__(dimension=2, binnings=binnings, frequencies=frequencies, **kwargs)

    @property
    def T(self) -> "Histogram2D":
        """Histogram with swapped axes.

        Returns
        -------
        Histogram2D - a copy with swapped axes
        """
        a_copy = self.copy()
        a_copy._binnings = list(reversed(a_copy._binnings))
        a_copy.axis_names = tuple(reversed(a_copy.axis_names))
        a_copy._frequencies = a_copy._frequencies.T
        if a_copy.errors2 is not None:
            a_copy._errors2 = a_copy._errors2.T
        return a_copy

    def partial_normalize(self, axis: Axis = 0, inplace: bool = False) -> "Histogram2D":
        """Normalize in rows or columns.

        Parameters
        ----------
        axis: int or str
            Along which axis to sum (numpy-sense)
        inplace: bool
            Update the object itself
        """
        # TODO: Is this applicable for HistogramND?
        axis = self._get_axis(axis)
        if not inplace:
            copy = self.copy()
            copy.partial_normalize(axis, inplace=True)
            return copy
        else:
            self._coerce_dtype(float)
            if axis == 0:
                divisor = np.atleast_1d(self._frequencies.sum(axis=0))
            else:
                divisor = np.atleast_2d(self._frequencies.sum(axis=1)[:, np.newaxis])
            divisor[divisor == 0] = 1  # Prevent division errors
            self._frequencies /= divisor
            self._errors2 /= divisor * divisor  # Has its limitations
            return self

    @property
    def numpy_like(self) -> Tuple[np.ndarray, ...]:
        """Same result as would the numpy.histogram function return."""
        return self.frequencies, self.numpy_bins[0], self.numpy_bins[1]


@overload
def calculate_frequencies(
    data: ArrayLike,
    binnings: Iterable[BinningBase],
    weights: Optional[ArrayLike] = None,
    *,
    dtype: Optional[DTypeLike] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    ...


@overload
def calculate_frequencies(
    data: None,
    binnings: Iterable[BinningBase],
    weights: Optional[ArrayLike] = None,
    *,
    dtype: Optional[DTypeLike] = None,
) -> Tuple[None, None, float]:
    ...


def calculate_frequencies(
    data: Optional[ArrayLike],
    binnings: Iterable[BinningBase],
    weights: Optional[ArrayLike] = None,
    *,
    dtype: Optional[DTypeLike] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
    """ "Get frequencies and bin errors from the data (n-dimensional variant).

    Parameters
    ----------
    data : 2D array with ndim columns and row for each entry.
    binnings: Binnings to apply in all axes.
    weights : 1D array of weights to assign to values.
        (If present, must have same length as the number of rows.)
    dtype : Underlying type for the histogram.
        (If weights are specified, default is float. Otherwise int64.)

    Returns
    -------
    frequencies : Frequencies (if data supplied)
    errors2 : Errors squared if different from frequencies
    missing : scalar[dtype]
    """
    if data is None:
        return None, None, 0

    # Prepare numpy array of data
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError(f"calculate_frequencies requires 2D input data, dim={data.ndim} found.")

    # Guess correct dtype and apply to weights
    if weights is None:
        if not dtype:
            dtype = np.int64
    else:
        weights = np.asarray(weights)
        if data is None:
            raise ValueError("Weights specified but data not.")
        if data.shape[0] != weights.shape[0]:
            raise ValueError("Different number of entries in data and weights.")
        if dtype:
            dtype = np.dtype(dtype)
            if dtype.kind in "iu" and weights.dtype.kind == "f":
                raise ValueError("Integer histogram requested but float weights entered.")
        else:
            dtype = weights.dtype

    edges_and_mask = [binning.numpy_bins_with_mask for binning in binnings]
    edges = [em[0] for em in edges_and_mask]
    masks = [em[1] for em in edges_and_mask]

    ixgrid = np.ix_(*masks)  # Indexer to select parts we want

    # TODO: Right edges are not taken into account because they fall into inf bin
    frequencies, _ = np.histogramdd(data, edges, weights=weights)
    frequencies = frequencies.astype(dtype)  # Automatically copy
    frequencies = frequencies[ixgrid]
    if weights is not None:
        missing = weights.sum() - frequencies.sum()
        err_freq, _ = np.histogramdd(data, edges, weights=weights ** 2)
        errors2 = err_freq[ixgrid].astype(dtype)  # Automatically copy
    else:
        missing = data.shape[0] - frequencies.sum()
        errors2 = None

    return frequencies, errors2, missing
