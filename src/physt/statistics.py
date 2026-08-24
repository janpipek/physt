"""Support for summary statistics kept in the histogram instances."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, cast

import attrs
import numpy as np

if TYPE_CHECKING:
    from typing import Any


# Define equality comparer for our Statistics class
_nan_equal = partial(np.array_equal, equal_nan=True)


@attrs.define(frozen=True)
class Statistics:
    """Container of statistics accumulative data."""

    sum: float = attrs.field(default=0.0, eq=attrs.cmp_using(_nan_equal))
    """Weighted sum of all values entered into histogram."""

    sum2: float = attrs.field(default=0.0, eq=attrs.cmp_using(_nan_equal))
    """Weighted sum of squares of the values used to construct the histogram."""

    min: float = attrs.field(default=np.inf, eq=attrs.cmp_using(_nan_equal))
    """Minimum value used to construct the histogram."""

    max: float = attrs.field(default=-np.inf, eq=attrs.cmp_using(_nan_equal))
    """Maximum value used to construct the histogram."""

    weight: float = attrs.field(default=0.0, eq=attrs.cmp_using(_nan_equal))
    """The total weight of values used to construct the histogram."""

    median: float = attrs.field(default=np.nan, eq=attrs.cmp_using(_nan_equal))
    """The median of the values used to construct the histogram.

    Note that any addition/subtraction or filling will destroy the
    value (unlike some other summary statistics.)
    """

    @property
    def mean(self) -> float:
        """Statistical mean of all values entered into histogram (weighted)."""
        try:
            return float(self.sum / self.weight)
        except ZeroDivisionError:
            return np.nan

    @property
    def std(self) -> float:
        """Standard deviation of all values entered into histogram."""
        return float(np.sqrt(self.variance))

    @property
    def variance(self) -> float:
        """Statistical variance of all values entered into histogram.

        This number is precise, because we keep the necessary data
        separate from bin contents.
        """
        if self.weight > 0:
            return float((self.sum2 - self.sum**2 / self.weight) / self.weight)
        return np.nan

    def __add__(self, other: Any) -> Statistics:
        if not isinstance(other, Statistics):
            return INVALID_STATISTICS
        return Statistics(
            sum=self.sum + other.sum,
            sum2=self.sum2 + other.sum2,
            min=min(self.min, other.min),
            max=max(self.max, other.max),
            weight=self.weight + other.weight,
            median=np.nan,
        )

    def __mul__(self, other: Any) -> Statistics:
        if not np.isscalar(other):
            return INVALID_STATISTICS
        other_scalar = cast(float, other)
        return attrs.evolve(
            self,
            sum=self.sum * other_scalar,
            sum2=self.sum2 * other_scalar**2,
            weight=self.weight * other_scalar,
        )

    def __rich_repr__(self):
        # Output interesting attributes instead of internal representation
        yield "mean", self.mean
        yield "std", self.std
        yield "min", self.min
        yield "max", self.max
        yield "total", self.weight

    __rich_repr__.angular = True  # type: ignore[attr-defined]

    def __str__(self):
        rich_str = ", ".join(f"{key}={value}" for key, value in self.__rich_repr__())
        return f"Statistics({rich_str})"


INVALID_STATISTICS: Statistics = Statistics(
    sum=np.nan, sum2=np.nan, min=np.nan, max=np.nan, weight=np.nan
)
"""Invalid statistics object used as placeholder when not enough information is available."""
