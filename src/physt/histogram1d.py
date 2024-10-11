"""One-dimensional histograms."""
from __future__ import annotations

import dataclasses
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from physt._construction import (
    calculate_1d_frequencies,
    extract_1d_array,
    extract_weights,
)
from physt.histogram_base import HistogramBase
from physt.statistics import INVALID_STATISTICS, Statistics

if TYPE_CHECKING:
    from typing import Any, Dict, Mapping, Optional, Tuple, Type, TypeVar, Union

    from physt.binnings import BinningBase, BinningLike
    from physt.typing_aliases import ArrayLike, Axis, DTypeLike

    Histogram1DType = TypeVar("Histogram1DType", bound="Histogram1D")


# TODO: Fix I/O with binning


class ObjectWithBinning(ABC):
    """Mixin with shared methods for 1D objects that have a binning.

    Note: Used to share behaviour between Histogram1D and HistogramCollection.
    """

    # TODO: Rename to something better

    @property
    @abstractmethod
    def binning(self) -> BinningBase:
        """The binning itself."""

    @property
    def ndim(self) -> int:
        return 1

    @property
    def bins(self) -> np.ndarray:
        """Array of all bin edges.

        Returns
        -------
        Wide-format [[leftedge1, rightedge1], ... [leftedgeN, rightedgeN]]
        """
        # TODO: Read-only copy
        return self.binning.bins  # TODO: or this should be read-only copy?

    @property
    def numpy_bins(self) -> np.ndarray:
        """Bins in the format of numpy."""
        # TODO: If not consecutive, does not make sense
        # TODO: Deprecate
        return self.binning.numpy_bins

    @property
    def edges(self) -> np.ndarray:
        return self.numpy_bins

    @property
    def bin_left_edges(self) -> np.ndarray:
        """Left edges of all bins."""
        return self.bins[..., 0]

    @property
    def bin_right_edges(self) -> np.ndarray:
        """Right edges of all bins."""
        return self.bins[..., 1]

    def get_bin_left_edges(self, i):
        assert i == 0
        return self.bin_left_edges

    def get_bin_right_edges(self, i):
        assert i == 0
        return self.bin_right_edges

    @property
    def min_edge(self) -> float:
        """Left edge of the first bin."""
        return self.bin_left_edges[0]

    @property
    def max_edge(self) -> float:
        """Right edge of the last bin."""
        return self.bin_right_edges[-1]

    @property
    def bin_centers(self) -> np.ndarray:
        """Centers of all bins."""
        return (self.bin_left_edges + self.bin_right_edges) / 2

    @property
    def bin_widths(self) -> np.ndarray:
        """Widths of all bins."""
        return self.bin_right_edges - self.bin_left_edges

    @property
    def total_width(self) -> float:
        """Total width of all bins.

        In inconsecutive histograms, the missing intervals are not counted in.
        """
        return self.bin_widths.sum().item()

    @property
    def bin_sizes(self) -> np.ndarray:
        return self.bin_widths


class Histogram1D(ObjectWithBinning, HistogramBase):
    """One-dimensional histogram data.

    The bins can be of different widths.

    The bins need not be consecutive. However, some functionality may not be available
    for non-consecutive bins (like keeping information about underflow and overflow).

    These are the basic attributes that can be used in the constructor (see there)
    Other attributes are dynamic.
    """

    def __init__(
        self,
        binning: BinningLike,
        frequencies: Optional[ArrayLike] = None,
        errors2: Optional[ArrayLike] = None,
        *,
        keep_missed: bool = True,
        stats: Optional[Statistics] = None,
        overflow: Optional[float] = 0.0,
        underflow: Optional[float] = 0.0,
        inner_missed: Optional[float] = 0.0,
        axis_name: Optional[str] = None,
        **kwargs,
    ):
        """Constructor

        Parameters
        ----------
        binning: The binning
        frequencies: The bin contents.
        keep_missed: Whether to keep track of underflow/overflow when filling with new values.
        underflow: Weight of observations that were smaller than the minimum bin.
        overflow: Weight of observations that were larger than the maximum bin.
        name: Name of the histogram (will be displayed as plot title)
        axis_name: Name of the characteristics that is histogrammed (will be displayed on x axis)
        errors2: Quadratic errors of individual bins. If not set, defaults to frequencies.
        stats: The statistics to use. If not set, defaults INVALID_STATISTICS.
        """
        missed = [
            underflow,
            overflow,
            inner_missed,
        ]
        if axis_name:
            kwargs["axis_names"] = [axis_name]

        HistogramBase.__init__(
            self, [binning], frequencies, errors2, keep_missed=keep_missed, **kwargs
        )

        if frequencies is None:
            self._stats = Statistics()
        else:
            self._stats = stats or INVALID_STATISTICS

        if self.keep_missed:
            self._missed = np.array(missed, dtype=self.dtype)
        else:
            self._missed = np.zeros(3, dtype=self.dtype)

    def copy(self, *, include_frequencies: bool = True) -> "Histogram1D":
        # Overriden to include the statistics as well
        a_copy = super().copy(include_frequencies=include_frequencies)
        if include_frequencies:
            a_copy._stats = dataclasses.replace(self.statistics)
        return a_copy

    @property
    def statistics(self) -> Statistics:
        return self._stats

    @property
    def axis_name(self) -> str:
        return self.axis_names[0]

    @axis_name.setter
    def axis_name(self, value: str):
        self.axis_names = (value,)

    def select(
        self, axis, index, *, force_copy: bool = False
    ) -> Union["Histogram1D", Tuple[np.ndarray, float]]:
        """Alias for [] to be compatible with HistogramND."""
        if axis == 0:
            if index == slice(None) and not force_copy:
                return self
            return self[index]
        else:
            raise ValueError("In Histogram1D.select(), axis must be 0.")

    def __getitem__(
        self, index: Union[int, slice, np.ndarray]
    ) -> Union["Histogram1D", Tuple[np.ndarray, float]]:
        """Select sub-histogram or get one bin.

        Parameters
        ----------
        index : int or slice or bool masked array or array with indices
            In most cases, this has same semantics as for numpy.ndarray.__getitem__


        Returns
        -------
        Histogram1D or tuple
            Depending on the parameters, a sub-histogram or content of one bin are returned.
        """
        underflow = np.nan
        overflow = np.nan
        keep_missed = False
        if isinstance(index, int):
            return self.bins[index], self.frequencies[index]
        if isinstance(index, np.ndarray):
            if index.dtype == bool:
                if index.shape != (self.bin_count,):
                    raise IndexError(
                        "Cannot index with masked array of a wrong dimension"
                    )
        elif isinstance(index, slice):
            keep_missed = self.keep_missed
            # TODO: Fix this
            if index.step:
                raise IndexError("Cannot change the order of bins")
            if index.step == 1 or index.step is None:
                underflow = self.underflow
                overflow = self.overflow
                if index.start:
                    underflow += self.frequencies[0 : index.start].sum()
                if index.stop:
                    overflow += self.frequencies[index.stop :].sum()
        # Masked arrays or item list or ...
        return self.__class__(
            self._binning.as_static(copy=False)[index],
            self.frequencies[index],
            self.errors2[index],
            overflow=overflow,
            keep_missed=keep_missed,
            underflow=underflow,
            dtype=self.dtype,
            name=self.name,
            axis_name=self.axis_name,
        )

    @property
    def _binning(self) -> BinningBase:
        """Adapter property for HistogramBase interface"""
        return self._binnings[0]

    @_binning.setter
    def _binning(self, value: BinningBase):
        self._binnings = [value]

    @property
    def binning(self) -> BinningBase:
        """The binning.

        Note: Please, do not try to update the object itself.
        """
        return self._binning

    @property
    def numpy_like(self) -> Tuple[np.ndarray, np.ndarray]:
        """Same result as would the numpy.histogram function return."""
        return self.frequencies, self.numpy_bins

    @property
    def cumulative_frequencies(self) -> np.ndarray:
        """Cumulative frequencies.

        Note: underflow values are not considered
        """
        return self._frequencies.cumsum()

    @property
    def underflow(self):
        if not self.keep_missed:
            return np.nan
        return self._missed[0]

    @underflow.setter
    def underflow(self, value):
        self._missed[0] = value

    @property
    def overflow(self):
        if not self.keep_missed:
            return np.nan
        return self._missed[1]

    @overflow.setter
    def overflow(self, value):
        self._missed[1] = value

    @property
    def inner_missed(self):
        if not self.keep_missed:
            return np.nan
        return self._missed[2]

    @inner_missed.setter
    def inner_missed(self, value):
        self._missed[2] = value

    def find_bin(self, value: float, axis: Optional[Axis] = None) -> Optional[int]:
        """Index of bin corresponding to a value.

        Returns
        -------
        index of bin to which value belongs
            (-1=underflow, N=overflow, None=not found - inconsecutive)
        """
        if axis is not None:
            self._get_axis(axis)  # Check that it is valid
        if not np.isscalar(value):
            raise ValueError(f"Non-scalar value for 1D histogram: {value}")
        ixbin = np.searchsorted(self.bin_left_edges, value, side="right").item()
        if ixbin == 0:
            return -1
        if ixbin == self.bin_count:
            if value <= self.bin_right_edges[-1]:
                return ixbin - 1
            else:
                return self.bin_count
        if value < self.bin_right_edges[ixbin - 1]:
            return ixbin - 1
        if ixbin == self.bin_count:
            return self.bin_count
        return None

    def fill(self, value: float, weight: float = 1, **kwargs) -> Optional[int]:
        """Update histogram with a new value.

        Parameters
        ----------
        value: Value to be added.
        weight: Weight assigned to the value.

        Returns
        -------
        index of bin which was incremented (-1=underflow, N=overflow, None=not found)

        Note: If a gap in unconsecutive bins is matched, underflow & overflow are not valid anymore.
        Note: Name was selected because of the eponymous method in ROOT
        """
        self._coerce_dtype(type(weight))
        if self._binning.is_adaptive():
            bin_map = self._binning.force_bin_existence(value)
            self._reshape_data(self._binning.bin_count, bin_map)

        ixbin = self.find_bin(value)
        if ixbin is None:
            self.overflow = np.nan
            self.underflow = np.nan
        elif ixbin == -1 and self.keep_missed:
            self.underflow += weight
        elif ixbin == self.bin_count and self.keep_missed:
            self.overflow += weight
        else:
            self._frequencies[ixbin] += weight
            self._errors2[ixbin] += weight**2
            try:
                self._stats = dataclasses.replace(
                    self.statistics,
                    weight=self.statistics.weight + weight,
                    sum=self.statistics.sum + weight * value,
                    sum2=self.statistics.sum2 + weight * value**2,
                    min=min(self.statistics.min, value),
                    max=max(self.statistics.max, value),
                    median=np.nan,
                )
            except OverflowError:
                warnings.warn("Overflow when updating statistics.")
                self._stats = INVALID_STATISTICS

        return ixbin

    def fill_n(
        self,
        values: ArrayLike,
        weights: Optional[ArrayLike] = None,
        *,
        dropna: bool = True,
    ) -> None:
        # TODO: Unify with HistogramBase
        values_array, array_mask = extract_1d_array(values, dropna=dropna)
        if self._binning.is_adaptive():
            map = self._binning.force_bin_existence(values_array)
            self._reshape_data(self._binning.bin_count, map)
        weights_array = extract_weights(weights, array_mask=array_mask)
        if weights_array is not None:
            self._coerce_dtype(weights_array.dtype)
        (frequencies, errors2, underflow, overflow, stats) = calculate_1d_frequencies(
            values_array,
            self._binning,
            dtype=self.dtype,
            weights=weights_array,
            validate_bins=False,
        )
        self._frequencies += frequencies
        self._errors2 += errors2
        # TODO: check that adaptive does not produce under-/over-flows?
        if self.keep_missed:
            self.underflow += underflow
            self.overflow += overflow
        self._stats += stats

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False
        # TODO: Change to something in binning itself
        if not np.allclose(other.bins, self.bins, equal_nan=True):
            return False
        if not np.allclose(other.frequencies, self.frequencies, equal_nan=True):
            return False
        if other.keep_missed != self.keep_missed:
            return False
        if self.keep_missed:
            if not np.allclose(other.overflow, self.overflow, equal_nan=True):
                return False
            if not np.allclose(other.underflow, self.underflow, equal_nan=True):
                return False
            if not np.allclose(other.inner_missed, self.inner_missed, equal_nan=True):
                return False
        if not np.allclose(other.errors2, self.errors2, equal_nan=True):
            return False
        if not other.name == self.name:
            return False
        if not other.axis_name == self.axis_name:
            return False
        return True

    @classmethod
    def _kwargs_from_dict(cls, a_dict: Mapping[str, Any]) -> Dict[str, Any]:
        kwargs = HistogramBase._kwargs_from_dict(a_dict)  # type: ignore
        kwargs["binning"] = kwargs.pop("binnings")[0]
        return kwargs

    @classmethod
    def from_calculate_frequencies(
        cls: Type["Histogram1DType"],
        data: Optional[np.ndarray],
        binning: BinningBase,
        weights: Optional[np.ndarray] = None,
        *,
        validate_bins: bool = True,
        already_sorted: bool = False,
        keep_missed: bool = True,
        dtype: Optional[DTypeLike] = None,
        **kwargs,
    ) -> "Histogram1DType":
        """Construct the histogram from values and bins."""
        # TODO: Remove this method

        if data is None:
            frequencies: Optional[np.ndarray] = None
            errors2: Optional[np.ndarray] = None
            underflow: float = 0.0
            overflow: float = 0.0
            stats: Optional[Statistics] = None
        else:
            frequencies, errors2, underflow, overflow, stats = calculate_1d_frequencies(
                data=data,
                binning=binning,
                weights=weights,
                validate_bins=validate_bins,
                already_sorted=already_sorted,
                dtype=dtype,
            )
            if not keep_missed:
                underflow = 0.0
                overflow = 0.0

        return cls(
            binning=binning,
            frequencies=frequencies,
            errors2=errors2,
            stats=stats,
            underflow=underflow,
            overflow=overflow,
            keep_missed=keep_missed,
            dtype=dtype,
            **kwargs,
        )
