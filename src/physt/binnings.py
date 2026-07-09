"""Different binning algorithms/schemas for the histograms."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable
from contextlib import suppress
from typing import TYPE_CHECKING, ParamSpec, TypeAlias, TypeVar, cast, final, overload

import attrs
import numpy as np
from typing_extensions import Self, override

from physt._bin_utils import (
    find_pretty_width,
    is_bin_subset,
    is_consecutive,
    is_rising,
    make_bin_array,
    to_numpy_bins,
    to_numpy_bins_with_mask,
)
from physt._util import deprecation_alias, find_subclass

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Any, ClassVar

    """Anything that can be converted to a binning."""

    from typing import Literal

    from physt.typing_aliases import ArrayLike, RangeTuple

    BinningType = TypeVar("BinningType", bound="BinningBase")
    BinningLike: TypeAlias = "BinningBase" | ArrayLike


BinMap: TypeAlias = Iterable[tuple[int, int]]
"""Description of the bin remapping - from left to right."""


EdgePair: TypeAlias = np.ndarray[tuple[int], np.dtype[np.float64]]
"""Edges of a bin (as a type)."""


binning_methods: dict[str, Callable] = {}
"""Dictionary of available binnings."""


P = ParamSpec("P")
R = TypeVar("R")


def register_binning(name: str | None = None) -> Callable[[Callable], Callable]:
    """Decorator to register among available binning methods."""

    def decorator(f: Callable[P, R]) -> Callable[P, R]:
        key = name or f.__name__[:-8]
        binning_methods[key] = f
        return f

    return decorator


@attrs.define(frozen=True)
class BinningBase(ABC):
    """Abstract base class for binning schemas.

    Inheriting
    ----------
    - define at least one of the following properties: bins, numpy_bins (cached conversion exists)
    - if you modify bins, put _bins and _numpy_bins into proper state (None may be sufficient)
    - checking of proper bins should be done in __init__
    - if you want to support adaptive histogram, override _force_bin_existence
    - implement _update_dict to contain the binning representation
    - the constructor (and facade methods) must accept any kwargs (and ignores those that are not used).
    """

    adaptive_allowed: ClassVar[bool] = False
    """Whether it is possible to update the bins dynamically."""

    inconsecutive_allowed: ClassVar[bool] = False
    """Whether it is possible to have bins with gaps."""

    adaptive: bool = attrs.field(default=False, kw_only=True)
    """Whether the binning is adaptive (bins are updated dynamically)."""

    @adaptive.validator
    def _validate_adaptive(self, attribute, value) -> bool:
        if value and not self.adaptive_allowed:
            raise ValueError(f"Adaptivity not allowed for {self.__class__.__name__}.")
        return value

    includes_right_edge: bool = attrs.field(default=False, kw_only=True)
    """Whether the right edge of the last bin is included in the binning."""

    def __attrs_post_init__(self):
        if self.includes_right_edge and self.adaptive:
            raise ValueError(
                "Adaptivity does not work together with right-edge inclusion."
            )

    @overload
    def __getitem__(self, index: slice) -> StaticBinning: ...

    @overload
    def __getitem__(self, index: int) -> EdgePair: ...

    def __getitem__(self, index: slice | int) -> StaticBinning | EdgePair:
        if isinstance(index, slice):
            same_right_edge = self.bins[index][-1, 1] == self.bins[-1, 1]
            return StaticBinning(
                bins=self.bins[index],
                includes_right_edge=same_right_edge and self.includes_right_edge,
            )
        return self.bins[index]

    @staticmethod
    def from_dict(a_dict: dict[str, Any]) -> BinningBase:
        binning_type: str = a_dict.pop("binning_type", "StaticBinning")
        klass = find_subclass(BinningBase, binning_type)
        return klass(**a_dict)

    @final
    def to_dict(self) -> dict[str, Any]:
        """Dictionary representation of the binning schema.

        This is a template method with the main attributes, please implement _update_dict
        to add details.
        """
        result: dict[str, Any] = {
            "adaptive": self.adaptive,
            "binning_type": type(self).__name__,
        }
        self._update_dict(result)
        return result

    @abstractmethod
    def _update_dict(self, a_dict: dict[str, Any]) -> None: ...

    def is_regular(self, *, rtol: float = 1.0e-5, atol: float = 1.0e-8) -> bool:
        """Whether all bins have the same width.

        Parameters
        ----------
        rtol, atol : numpy tolerance parameters
        """
        return np.allclose(
            np.diff(self.bins[1] - self.bins[0]), 0.0, rtol=rtol, atol=atol
        )

    def is_consecutive(self, *, rtol: float = 1.0e-5, atol: float = 1.0e-8) -> bool:
        """Whether all bins are in a growing order.

        Parameters
        ----------
        rtol, atol : numpy tolerance parameters
        """
        if self.inconsecutive_allowed:
            return is_consecutive(self.bins, rtol=rtol, atol=atol)
        return True

    @final
    def force_bin_existence(self, values: ArrayLike) -> int | BinMap | None:
        """Change schema so that there is a bin for value.

        It is necessary to implement the _force_bin_existence template method.

        Parameters
        ----------
        values: All values we want bins for.

        Returns
        -------
        bin_map: BinMap or None or int
            None => There was no change in bins
            int => The bins are only shifted (allows mass assignment)
            otherwise => the iterable contains tuples (old bin index, new bin index)
                new bin index can occur multiple times, which corresponds to bin merging
        """
        if not self.is_adaptive():
            raise RuntimeError("Histogram is not adaptive.")
        else:
            return self._force_bin_existence(values)

    def _force_bin_existence(self, values: ArrayLike) -> int | BinMap | None:
        # Implement this if appropriate. It cannot be an abstract method.
        # It does not check whether the binning is adaptive
        raise NotImplementedError()

    @final
    def adapt(self, other: "BinningBase") -> tuple[BinMap | None, BinMap | None]:
        """Adapt this binning so that it contains all bins of another binning.

        Parameters
        ----------
        other: BinningBase

        Returns
        -------
        map1: A remapping from old bins to new bins. If not changed, None.
        map2: A remapping of `other` bins to new bins. If not changed, None.

        Note
        ----
        Implement the `_adapt` template method.
        """
        if not self.is_adaptive():
            raise RuntimeError("Cannot adapt non-adaptive binning.")
        if np.array_equal(self.bins, other.bins):
            # Already adapted
            return None, None
        return self._adapt(other)

    def _adapt(self, other: "BinningBase") -> tuple[BinMap | None, BinMap | None]:
        # Implement this if appropriate. It cannot be an abstract method.
        raise RuntimeError(f"Cannot adapt {self.__class__.__name__}.")

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if type(other) is not type(self):
            return False
        if (bins := self.bins) is not None:
            return np.array_equal(bins, other.bins)
        return False

    @property
    @abstractmethod
    def bin_count(self) -> int:
        """The total number of bins."""

    @property
    @abstractmethod
    def bins(self) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        """Bins in the wider format (as edge pairs)

        Returns
        -------
        bins: np.ndarray
            shape=(bin_count, 2)
        """

    @property
    @abstractmethod
    def numpy_bins(self) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Bins in the numpy format

        This might not be available for inconsecutive binnings.

        Returns
        -------
        edges: np.ndarray
            shape=(bin_count+1,)
        """

    @property
    @abstractmethod
    def numpy_bins_with_mask(self) -> tuple[np.ndarray, np.ndarray]:
        """Bins in the numpy format, including the gaps in inconsecutive binnings.

        Returns
        -------
        edges, mask: np.ndarray

        See Also
        --------
        bin_utils.to_numpy_bins_with_mask
        """

    # TODO: is this used?
    @property
    def first_edge(self) -> float:
        """The left edge of the first bin."""
        return self.bins[0, 0].item()

    # TODO: is this used?
    @property
    def last_edge(self) -> float:
        """The right edge of the last bin."""
        return self.bins[-1, 1].item()

    def as_static(self, *, copy: bool = True) -> "StaticBinning":  # pylint: disable=unused-argument
        """Convert binning to a static form.

        Parameters
        ----------
        copy: bool
            Ensure that we receive another object

        Returns
        -------
        StaticBinning
            A new static binning with a copy of bins.
        """
        if not copy:
            return self
        return StaticBinning(
            bins=self.bins.copy(), includes_right_edge=self.includes_right_edge
        )

    def as_fixed_width(self, *, copy: bool = True) -> "FixedWidthBinning":  # pylint: disable=unused-argument
        """Convert binning to recipe with fixed width (if possible).

        Parameters
        ----------
        copy: If True, ensure that we receive another object.
        """
        if self.bin_count == 0:
            raise ValueError("Cannot guess binning width with zero bins")
        elif self.bin_count == 1 or self.is_consecutive() and self.is_regular():
            return FixedWidthBinning(
                min=self.bins[0, 0],
                bin_count=self.bin_count,
                bin_width=self.bins[1] - self.bins[0],
            )
        else:
            raise ValueError(
                "Cannot create fixed-width binning from differing bin widths."
            )

    def copy(self) -> Self:
        """Create an identical, independent copy."""
        return attrs.evolve(self)

    def apply_bin_map(self, bin_map: BinMap) -> "BinningBase":
        """...

        Parameters
        ----------
        bin_map: Iterator(tuple)
            The bins must be in ascending order
        """
        length = max(item[1] for item in bin_map) + 1
        bins = np.empty((length, 2), dtype=float)
        bins[:] = np.nan
        for old, new in bin_map:
            if np.isnan(bins[new, 0]):
                bins[new, :] = self.bins[old, :]
            else:
                if bins[new, 1] != self.bins[old, 0]:
                    raise ValueError("Merging non-consecutive bins.")
                bins[new, 1] = self.bins[old, 1]
        if np.any(np.isnan(bins)):
            raise ValueError("New binning is not complete.")
        includes_right_edge = (
            self.includes_right_edge and bins[-1, 1] == self.bins[-1, 1]
        )
        binning = StaticBinning(bins, includes_right_edge=includes_right_edge)
        return binning

    def __repr__(self):
        return f"{self.__class__.__name__}({self.numpy_bins!r})"


@attrs.define(frozen=True)
class StaticBinning(BinningBase):
    """Binning defined by an array of bin edge pairs."""

    bins: np.ndarray = attrs.field(converter=make_bin_array)

    inconsecutive_allowed: ClassVar[bool] = True

    @bins.validator
    def _validate_rising_bins(self, attribute, value):
        if not is_rising(value):
            raise ValueError("Bins must be in rising order.")

    @property
    def bin_count(self) -> int:
        return self.bins.shape[0]

    @property
    def numpy_bins(self) -> np.ndarray:
        return to_numpy_bins(self.bins)

    @property
    def numpy_bins_with_mask(self) -> tuple[np.ndarray, np.ndarray]:
        edges, mask = to_numpy_bins_with_mask(self.bins)
        if not self.includes_right_edge:
            edges = np.concatenate([edges, np.asarray([np.inf])])
        return edges, mask

    def as_static(self, copy: bool = True) -> "StaticBinning":
        if copy:
            return self.copy()
        return self

    def copy(self) -> StaticBinning:
        return type(self)(
            bins=self.bins.copy(), includes_right_edge=self.includes_right_edge
        )

    def _update_dict(self, a_dict: dict[str, Any]) -> None:
        a_dict["bins"] = self.bins.tolist()

    def _adapt(self, other: BinningBase) -> tuple[None, BinMap]:
        if is_bin_subset(other.bins, self.bins):
            indices = np.searchsorted(other.bins[:, 0], self.bins[:, 0])
            return None, list(enumerate(indices))
        raise ValueError("Cannot adapt binning with different bins.")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.bins!r})"


@attrs.define(frozen=True, kw_only=True)
class EdgeBasedBinning(BinningBase, ABC):
    """Binning that provides edges and derives bins from it.

    Note: Thus, it cannot be inconsecutive.
    """

    @property
    @override
    def bins(self) -> np.ndarray:
        return make_bin_array(self.numpy_bins)

    @property
    def numpy_bins_with_mask(self) -> tuple[np.ndarray, np.ndarray]:
        if self.includes_right_edge:
            edges = np.concatenate([self.numpy_bins, np.asarray([np.inf])])
            mask = np.arange(self.numpy_bins.shape[0], dtype=int)
        else:
            edges = self.numpy_bins
            mask = np.arange(self.numpy_bins.shape[0] - 1, dtype=int)
        return edges, mask

    @property
    @override
    def bin_count(self) -> int:
        return self.numpy_bins.shape[0] - 1


@attrs.define(frozen=True, kw_only=True)
class NumpyBinning(EdgeBasedBinning):
    """Binning schema working as numpy.histogram."""

    numpy_bins: np.ndarray = attrs.field(converter=np.asarray)

    @numpy_bins.validator
    def _validate_rising_bins(self, attribute, value):
        if not is_rising(value):
            raise ValueError("Bins must be in rising order.")

    def _update_dict(self, a_dict: dict[str, Any]) -> None:
        a_dict["numpy_bins"] = self.numpy_bins.tolist()


@attrs.define(frozen=True, kw_only=True)
class FixedWidthBinning(EdgeBasedBinning):
    """Binning schema with predefined bin width."""

    adaptive_allowed: ClassVar[bool] = True

    bin_count: int = attrs.field(default=0)
    bin_width: float = attrs.field(default=1.0, converter=float)
    times_min: int | None = attrs.field(default=None)
    shift: float = 0.0

    @times_min.validator
    def _validate_times_min(self, attribute, value):
        if value is None and self.bin_count > 0:
            raise ValueError("times_min must be defined when bin_count > 0.")
        return value

    @bin_count.validator
    def _validate_bin_count(self, attribute, value):
        if value < 0:
            raise ValueError("bin_count must be >= 0.")
        return value

    @bin_width.validator
    def _validate_bin_width(self, attribute, value):
        if value <= 0:
            raise ValueError("bin_width must be > 0.")
        return value

    @classmethod
    def create_from_min_max(
        cls,
        *,
        min_: float,
        max_: float,
        bin_width: float,
        includes_right_edge: bool = False,
        align: bool = True,
    ) -> "FixedWidthBinning":
        times_min = int(np.floor(min_ / bin_width))
        shift = 0 if align else min_ - times_min * bin_width
        bin_count = max(1, int(np.floor(max_ - (times_min * bin_width + shift))))
        if includes_right_edge and shift + bin_count * bin_width == max_:
            bin_count += 1
        return cls(
            bin_width=bin_width,
            bin_count=bin_count,
            times_min=times_min,
            shift=shift,
            includes_right_edge=includes_right_edge,
        )

    @override
    def __repr__(self):
        result = (
            f"{self.__class__.__name__}(bin_width={self.bin_width}, "
            f"bin_count={self.bin_count}, min={self.first_edge}"
        )
        if self.is_adaptive():
            result += ", adaptive=True"
        return result + ")"

    def is_regular(self, **kwargs) -> bool:
        return True

    def _force_bin_existence_single(
        self, value: float, *, includes_right_edge: bool | None = None
    ) -> tuple["FixedWidthBinning", int | None]:
        if includes_right_edge is None:
            includes_right_edge = self.includes_right_edge

        if self._bin_count == 0:
            self._times_min = int(np.floor((value - self._shift) / self.bin_width))
            self._bin_count = 1
            # No bins yet, no remapping needed
            return 0
        else:
            add_left = add_right = 0
            if value < self.numpy_bins[0]:
                add_left = int(np.ceil((self.numpy_bins[0] - value) / self.bin_width))
                self._times_min -= add_left  # type: ignore  # We know it is not None
                self._bin_count += add_left
            elif value >= self.numpy_bins[-1]:
                add_right = (value - self.numpy_bins[-1]) / self.bin_width
                add_right = int(np.ceil(add_right))
                self._bin_count += add_right
                if self.last_edge == value and not includes_right_edge:
                    add_right += 1
                    self._bin_count += 1
            if add_left or add_right:
                return add_left
            else:
                return None

    def _force_bin_existence(
        self, values: ArrayLike, *, includes_right_edge=None
    ) -> int | BinMap | None:
        if np.isscalar(values):
            return self._force_bin_existence_single(
                cast(float, values), includes_right_edge=includes_right_edge
            )
        else:
            arr = np.asarray(values)
            min_, max_ = arr.min(), arr.max()
            result = self._force_bin_existence_single(min_)
            result2 = self._force_bin_existence_single(
                max_, includes_right_edge=includes_right_edge
            )
            if result is None:
                return result2
            else:
                return result

    @property
    def first_edge(self) -> float:
        return self._bin_width * self._validate_times_min() + self._shift

    @property
    def last_edge(self) -> float:
        return (
            self._validate_times_min() + self._bin_count
        ) * self._bin_width + self._shift

    @property
    # TODO: Cache this
    def numpy_bins(self) -> np.ndarray:
        if not self.bin_count:
            return np.zeros((0, 2), dtype=float)
        return (
            self._validate_times_min() + np.arange(self.bin_count + 1, dtype=int)
        ) * self.bin_width + self.shift

    def _validate_times_min(self) -> int:
        """Check the binning is well-defined and return the times min."""
        if self.times_min is None:
            raise ValueError(
                "No bins and not enough information to provide first edge."
            )
        return self.times_min

    def _force_new_min_max(self, new_min, new_max) -> BinMap | None:
        bin_map = None
        add_right = add_left = 0
        times_min = self._validate_times_min()
        if new_min < times_min:  # type: ignore
            add_left = times_min - new_min
        if new_max - times_min > self._bin_count:  # type: ignore
            add_right = new_max - times_min - self._bin_count
        if add_left or add_right:
            bin_map = ((i, i + add_left) for i in range(self._bin_count))
            self._set_min_and_count(
                times_min - add_left, self._bin_count + add_left + add_right
            )
        return bin_map

    def _set_min_and_count(self, times_min: int | None, bin_count: int) -> None:
        self._bin_count = bin_count
        self._times_min = times_min

    def _adapt(self, other: BinningBase) -> tuple[BinMap | None, BinMap | None]:
        other = other.as_fixed_width()
        if self.bin_width != other.bin_width:
            raise ValueError(
                "Cannot adapt fixed-width histograms with different widths"
            )
        if self._shift != other._shift:
            raise ValueError(
                f"Cannot adapt shifted fixed-width histograms: {self._shift} vs {other._shift}"
            )
        # Following operations modify schemas
        other = other.copy()
        if other.bin_count == 0:
            return None, ()
        if self.bin_count == 0:
            self._set_min_and_count(other._times_min, other.bin_count)
            return (), None

        times_min = self._validate_times_min()
        other_times_min = other._validate_times_min()

        new_min: float = min(times_min, other_times_min)
        new_max: float = max(
            times_min + self._bin_count, other_times_min + other._bin_count
        )

        bin_map1 = self._force_new_min_max(new_min, new_max)
        bin_map2 = other._force_new_min_max(new_min, new_max)
        return bin_map1, bin_map2

    def as_fixed_width(self, *, copy: bool = True) -> "FixedWidthBinning":
        if copy:
            return self.copy()
        return self

    def _update_dict(self, a_dict: dict[str, Any]) -> None:
        # TODO: Fix to be instantiable from JSON
        a_dict["bin_count"] = self.bin_count
        a_dict["bin_width"] = self.bin_width
        a_dict["bin_shift"] = self.shift
        a_dict["bin_times_min"] = self.times_min


@attrs.define(frozen=True, kw_only=True)
class ExponentialBinning(EdgeBasedBinning):
    """Binning schema with exponentially distributed bins."""

    adaptive_allowed: ClassVar[bool] = False

    bin_count: int = attrs.field()
    log_min: float
    log_width: float

    @bin_count.validator
    def _validate_bin_count(self, attribute, value):
        if value <= 0:
            raise ValueError("bin_count must be positive")

    def is_regular(self, **kwargs) -> bool:
        return False

    @property
    def numpy_bins(self) -> np.ndarray:
        if self.bin_count == 0:
            return np.ndarray((0,), dtype=float)
        log_bins = self.log_min + np.arange(self.bin_count + 1) * self.log_width
        return 10.0**log_bins

    def _update_dict(self, a_dict: dict[str, Any]) -> None:
        a_dict["log_min"] = self.log_min
        a_dict["log_width"] = self.log_width
        a_dict["bin_count"] = self.bin_count


@register_binning()
def numpy_binning(
    data: np.ndarray | None = None,
    bin_count: int = 10,
    range: RangeTuple | None = None,
) -> NumpyBinning:
    """Construct binning schema compatible with numpy.histogram together with int argument

    Parameters
    ----------
    data: This is optional if both bins and range are set
    bin_count: int
    range: (min, max)
    includes_right_edge: default: True

    See Also
    --------
    numpy.histogram
    static_binning
    """
    if not isinstance(bin_count, int):
        raise TypeError("bin_count must be a number.")
    if range:
        edges = np.linspace(range[0], range[1], bin_count + 1)
    else:
        if data is None:
            raise ValueError("Either `range` or `data` must be set.")
        if data.size < 2:
            raise ValueError(
                f"At least 2 values required to infer bins, {data.size} given."
            )
        start = data.min()
        stop = data.max()
        if start == stop:
            raise ValueError(
                f"At least 2 different values required to infer bins, all are equal to {start}."
            )
        if not np.isfinite(stop - start):
            raise ValueError(f"Range too large to find bins: {start} to {stop}.")
        edges = np.linspace(start, stop, bin_count + 1)
        if (np.diff(edges) == 0).any():
            # Artificially widen the range so that the bins are distinct
            warnings.warn(
                f"Range too narrow to split into {bin_count} bins: {start} to {stop}.",
                RuntimeWarning,
            )
            edges_ = list(np.unique(edges))
            from builtins import range as range_

            for _ in range_(bin_count - len(edges_) + 1):
                edges_.append(np.nextafter(edges_[-1], np.inf))
            edges = np.array(edges_)

    return NumpyBinning(numpy_bins=edges)


@register_binning()
def pretty_binning(
    data: np.ndarray | None,
    bin_count: int | None = None,
    *,
    kind: Literal["time"] | None = None,
    range: RangeTuple | None = None,
    min_bin_width: float | None = None,
    max_bin_width: float | None = None,
    adaptive: bool = False,
    includes_right_edge: bool = False,
) -> FixedWidthBinning:
    """Construct fixed-width ninning schema with bins automatically optimized to human-friendly widths.

    Typical widths are: 1.0, 25,0, 0.02, 500, 2.5e-7, ...

    Parameters
    ----------
    bin_count: Starting number of bins (the result will be close)
    kind: Optional value "time" works in h,m,s scale instead of seconds
    range: Tuple of (min, max)
    min_bin_width: If present, the bin cannot be narrower than this.
    max_bin_width: If present, the bin cannot be wider than this.
    """
    # TODO: remove colliding kwargs
    if range is None:
        if data is None:
            raise ValueError("Cannot guess optimum bin width without data.")
        min_ = data.min().item()
        max_ = data.max().item()
    else:
        min_, max_ = range
    if bin_count is None:
        if data is None:
            raise ValueError("Cannot guess optimum bin count without data.")
        bin_count = ideal_bin_count(data)

    raw_width = (max_ - min_) / bin_count
    bin_width = find_pretty_width(raw_width, kind=kind)

    if min_bin_width:
        bin_width = max(bin_width, min_bin_width)
    if max_bin_width:
        bin_width = min(bin_width, max_bin_width)

    return fixed_width_binning(
        bin_width=bin_width,
        data=data,
        range=range,
        align=True,
        adaptive=adaptive,
        includes_right_edge=includes_right_edge,
    )


human_binning = deprecation_alias(pretty_binning, "human_binning")
register_binning(name="human")(human_binning)


@register_binning()
def quantile_binning(
    data: np.ndarray | None,
    *,
    bin_count: int | None = None,
    q: Sequence[float] | None = None,
    qrange: RangeTuple | None = None,
    includes_right_edge: bool = True,
) -> StaticBinning:
    """Binning schema based on quantile ranges.

    This binning finds equally spaced quantiles. This should lead to
    all bins having roughly the same frequencies.

    Note: weights are not (yet) take into account for calculating
    quantiles.

    Parameters
    ----------
    bin_count: Number of bins
    q: Sequence of quantiles to be used as edges (a la numpy)
    qrange: Two floats as minimum and maximum quantile (default: 0.0, 1.0)
    """
    if data is None:
        raise ValueError("Cannot construct quantile binning without data.")
    if (bin_count is not None and q is not None) or (bin_count is None and q is None):
        raise ValueError("Exactly one of `bin_count` and `q` must be set.")
    if bin_count:
        if qrange is None:
            qrange = (0.0, 1.0)
        percentiles = np.linspace(qrange[0] * 100, qrange[1] * 100, bin_count + 1)
    elif qrange is not None:
        raise ValueError("Cannot set both `q` and `qrange`")
    else:
        percentiles = np.asarray(q) * 100.0
    bins = np.percentile(data, percentiles)
    return static_binning(
        bins=make_bin_array(bins), includes_right_edge=includes_right_edge
    )


@register_binning()
def static_binning(
    data: np.ndarray | None = None,
    *,
    bins: ArrayLike,
    includes_right_edge: bool = False,
) -> StaticBinning:
    """Construct static binning with whatever bins.

    Any data passed in will be ignored.
    """
    # TODO: Fail with no bins!
    return StaticBinning(
        bins=make_bin_array(bins), includes_right_edge=includes_right_edge
    )


@register_binning()
def integer_binning(
    data: np.ndarray | None = None,
    *,
    adaptive: bool = False,
    range: tuple[int, int] | None = None,
    bin_width: int = 1,
) -> FixedWidthBinning:
    """Construct fixed-width binning schema with bins centered around integers.

    Parameters
    ----------
    range:  min (included) and max integer (excluded) bin
    bin_width: group "bin_width" integers into one bin (not recommended)
    """
    kwargs: dict[str, Any] = {}
    if range:
        kwargs["range"] = tuple(r - 0.5 for r in range)
    else:
        kwargs["bin_shift"] = 0.5
    return fixed_width_binning(
        data=data, bin_width=bin_width, align=False, adaptive=adaptive, **kwargs
    )


@register_binning()
def fixed_width_binning(
    data: np.ndarray | None = None,
    bin_width: float = 1.0,
    *,
    min_: float | None = None,
    bin_count: int | None = None,
    adaptive: bool = False,
    range: RangeTuple | None = None,
    align: bool | None = None,
    includes_right_edge: bool = False,
) -> FixedWidthBinning:
    """Construct fixed-width binning schema.

    Parameters
    ----------
    bin_width: float
    range: (min, max)
    align: Must be multiple of bin_width
    """
    if data is not None or range:
        # First try to create from limits
        min_ = None
        max_ = None

        if bin_count is not None:
            raise ValueError("Cannot set both `bin_count` and `data`/`range`.")

        if data is not None:
            arr = np.asarray(data)
            min_ = np.min(arr)
            max_ = np.max(arr)
            if align is None:
                align = True
        if range:
            # This takes precedence over data
            if align:
                raise ValueError("Cannot set both `align` and `range`.")
            if align is None:
                # We should respect the lower bound of the range
                align = False
            if adaptive:
                raise ValueError("Cannot set `adaptive` when `range` is set.")
            align = False
            min_ = range[0]
            max_ = range[1]

        return FixedWidthBinning.create_from_min_max(
            min_=min_,
            max_=max_,
            bin_width=bin_width,
            align=align,
            includes_right_edge=includes_right_edge,
        )

    return FixedWidthBinning(
        bin_width=bin_width,
        includes_right_edge=includes_right_edge,
        adaptive=adaptive,
    )


@register_binning()
def exponential_binning(
    data: np.ndarray | None = None,
    bin_count: int | None = None,
    *,
    range: RangeTuple | None = None,
    **kwargs,
) -> ExponentialBinning:
    """Construct exponential binning schema.

    Parameters
    ----------
    bin_count: Number of bins
    range: (min, max)

    See also
    --------
    numpy.logspace - note that our range semantics is different
    """
    if bin_count is None:
        if data is None:
            raise ValueError("Cannot find optimum bin count without data.")
        bin_count = ideal_bin_count(data)

    if range:
        range = (np.log10(range[0]), np.log10(range[1]))
    else:
        if data is None:
            raise ValueError("Cannot guess the range without data.")
        range = (np.log10(data.min()), np.log10(data.max()))
    log_width = (range[1] - range[0]) / bin_count
    return ExponentialBinning(
        log_min=range[0], log_width=log_width, bin_count=bin_count, **kwargs
    )


with suppress(ImportError):
    # If possible, import astropy's binning methods
    # See: http://docs.astropy.org/en/stable/visualization/histogram.html

    from astropy.stats.histogram import histogram as _astropy_histogram  # noqa: F401

    warnings.filterwarnings("ignore", module="astropy\\..*")

    @register_binning(name="blocks")
    def bayesian_blocks_binning(
        data: np.ndarray, *, range: RangeTuple | None = None, **kwargs
    ) -> StaticBinning:
        """Binning schema based on Bayesian blocks (from astropy).

        Computationally expensive for large data sets.

        See also
        --------
        astropy.stats.histogram.bayesian_blocks
        astropy.stats.histogram.histogram
        """
        # TODO: This is not in astropy.histogram.__all__!
        from astropy.stats.histogram import bayesian_blocks

        if range is not None:
            data = data[(data >= range[0]) & (data <= range[1])]
        edges = bayesian_blocks(data)
        return StaticBinning(edges, **kwargs)

    @register_binning()
    def knuth_binning(
        data: np.ndarray, *, range: RangeTuple | None = None, **kwargs
    ) -> StaticBinning:
        """Binning schema based on Knuth's rule (from astropy).

        Computationally expensive for large data sets.

        See also
        --------
        astropy.stats.histogram.knuth_bin_width
        astropy.stats.histogram.histogram
        """
        # TODO: Could we possibly use it with FixedWidthBinning?
        from astropy.stats.histogram import knuth_bin_width

        if range is not None:
            data = data[(data >= range[0]) & (data <= range[1])]
        _, edges = knuth_bin_width(data, return_bins=True)
        return StaticBinning(edges, **kwargs)

    @register_binning()
    def scott_binning(
        data: np.ndarray, *, range: RangeTuple | None = None, **kwargs
    ) -> StaticBinning:
        """Binning schema based on Scott's rule (from astropy).

        See also
        --------
        astropy.stats.histogram.scott_bin_width
        astropy.stats.histogram.histogram
        """
        from astropy.stats.histogram import scott_bin_width

        if range is not None:
            data = data[(data >= range[0]) & (data <= range[1])]
        _, edges = scott_bin_width(data, return_bins=True)
        return StaticBinning(edges, **kwargs)

    @register_binning()
    def freedman_binning(
        data: np.ndarray, *, range: RangeTuple | None = None, **kwargs
    ) -> StaticBinning:
        """Binning schema based on Freedman-Diaconis rule (from astropy).

        Parameters
        ----------
        data: arraylike
        range: Optional[tuple]

        See also
        --------
        astropy.stats.histogram.freedman_bin_width
        astropy.stats.histogram.histogram
        """
        # TODO: Could we possibly use it with FixedWidthBinning?
        from astropy.stats.histogram import freedman_bin_width

        if range is not None:
            data = data[(data >= range[0]) & (data <= range[1])]
        _, edges = freedman_bin_width(data, return_bins=True)
        return StaticBinning(edges, **kwargs)


def ideal_bin_count(data: np.ndarray, method: str = "default") -> int:
    """A theoretically ideal bin count.

    Parameters
    ----------
    data: Data to work on. Most methods don't use this.
    method: str
        Name of the method to apply, available values:
          - default (~sturges)
          - sqrt
          - sturges
          - doane
          - rice
        See https://en.wikipedia.org/wiki/Histogram for the description
    """
    value_count = data.size
    if value_count < 1:
        return 1
    if method == "default":
        if value_count <= 32:
            return 7
        else:
            return ideal_bin_count(data, "sturges")
    if method == "sqrt":
        return int(np.ceil(np.sqrt(value_count)))
    if method == "sturges":
        return int(np.ceil(np.log2(value_count)) + 1)
    if method == "doane":
        if value_count < 3:
            return 1

        sigma = np.sqrt(6 * (value_count - 2) / (value_count + 1) * (value_count + 3))
        return int(
            np.ceil(1 + np.log2(value_count) + np.log2(1 + np.abs(_skew(data)) / sigma))
        )
    if method == "rice":
        return int(np.ceil(2 * np.power(value_count, 1 / 3)))
    raise ValueError(f"Unknown bin count method: {method}")


bincount_methods = ["default", "sturges", "rice", "sqrt", "doane"]


def as_binning(obj: BinningLike, copy: bool = False) -> BinningBase:
    """Ensure that an object is a binning

    Parameters
    ---------
    obj : BinningBase or array_like
        Can be a binning, numpy-like bins or full physt bins
    copy : If true, ensure that the returned object is independent
    """
    if isinstance(obj, BinningBase):
        if copy:
            return obj.copy()
        else:
            return obj
    else:
        bins = make_bin_array(obj)
        return StaticBinning(bins)


def _skew(data: np.ndarray) -> float:
    """Compute skewness, i.e. the third standardised moment.

    See also: `scipy.stats.skew`

    https://en.wikipedia.org/wiki/Skewness#Definition
    """
    data = data.flatten()
    mean = np.mean(data)
    sigma = np.std(data)
    sum_res = np.sum((data - mean) ** 3)
    return (sum_res / sigma**3 / len(data)).item()
