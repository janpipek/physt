from dataclasses import dataclass
import dataclasses
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Statistics:
    """Container of statistics accumulative data."""

    sum: float = 0.0
    sum2: float = 0.0
    min: float = np.inf
    max: float = -np.inf
    weight: float = 0.0

    def mean(self) -> float:
        """Statistical mean of all values entered into histogram (weighted)."""
        try:
            return self.sum / self.weight
        except ZeroDivisionError:
            return np.nan

    def std(self) -> float:  # , ddof=0):
        """Standard deviation of all values entered into histogram."""
        # TODO: Add DOF
        return np.sqrt(self.variance())

    def variance(self) -> float:  # , ddof: int = 0) -> float:
        """Statistical variance of all values entered into histogram.

        This number is precise, because we keep the necessary data
        separate from bin contents.
        """
        # TODO: Add DOF
        # http://stats.stackexchange.com/questions/6534/how-do-i-calculate-a-weighted-standard-deviation-in-excel
        if self.weight > 0:
            return (self.sum2 - self.sum ** 2 / self.weight) / self.weight
        return np.nan

    def __add__(self, other: Any) -> "Statistics":
        if not isinstance(other, Statistics):
            return INVALID_STATISTICS
        return Statistics(
            sum=self.sum + other.sum,
            sum2=self.sum2 + other.sum2,
            min=min(self.min, other.min),
            max=max(self.max, other.max),
            weight=self.weight + other.weight,
        )

    def __mul__(self, other: Any) -> "Statistics":
        if not np.isscalar(other):
            return INVALID_STATISTICS
        other_scalar = float(other)
        return dataclasses.replace(
            self,
            sum=self.sum * other_scalar,
            sum2=self.sum2 * other_scalar ** 2,
            weight=self.weight * other_scalar,
        )


INVALID_STATISTICS: Statistics = Statistics(
    sum=np.nan, sum2=np.nan, min=np.nan, max=np.nan, weight=np.nan
)
