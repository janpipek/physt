"""Support for summary statistics kept in the histogram instances."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    from typing import Any


@dataclass(frozen=True)
class Statistics:
    """Container of statistics accumulative data."""

    sum: float = 0.0
    """Weighted sum of all values entered into histogram."""

    sum2: float = 0.0
    """Weighted sum of squares of the values used to construct the histogram."""

    min: float = np.inf
    """Minimum value used to construct the histogram."""

    max: float = -np.inf
    """Maximum value used to construct the histogram."""

    weight: float = 0.0
    """The total weight of values used to construct the histogram."""

    median: float = np.nan
    """The median of the values used to construct the histogram.

    Note that any addition/subtraction or filling will destroy the
    value (unlike some other summary statistics.)
    """

    @property
    def mean(self) -> float:
        """Statistical mean of all values entered into histogram (weighted)."""
        try:
            return self.sum / self.weight
        except ZeroDivisionError:
            return np.nan

    @property
    def std(self) -> float:
        """Standard deviation of all values entered into histogram."""
        return np.sqrt(self.variance)

    @property
    def variance(self) -> float:
        """Statistical variance of all values entered into histogram.

        This number is precise, because we keep the necessary data
        separate from bin contents.
        """
        if self.weight > 0:
            return (self.sum2 - self.sum**2 / self.weight) / self.weight
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
        return dataclasses.replace(
            self,
            sum=self.sum * other_scalar,
            sum2=self.sum2 * other_scalar**2,
            weight=self.weight * other_scalar,
        )

    def __eq__(self, other):
        if not isinstance(other, Statistics):
            return False
        return (
            np.array_equal(self.sum, other.sum, equal_nan=True)
            and np.array_equal(self.sum2, other.sum2, equal_nan=True)
            and np.array_equal(self.min, other.min, equal_nan=True)
            and np.array_equal(self.max, other.max, equal_nan=True)
            and np.array_equal(self.weight, other.weight, equal_nan=True)
            and np.array_equal(self.median, other.median, equal_nan=True)
        )


INVALID_STATISTICS: Statistics = Statistics(
    sum=np.nan, sum2=np.nan, min=np.nan, max=np.nan, weight=np.nan
)
"""Invalid statistics object used as placeholder when not enough information is available."""
