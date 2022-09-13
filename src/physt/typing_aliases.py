"""Definitions for type hints."""
from typing import Tuple, Union

from numpy.typing import ArrayLike, DTypeLike

RangeTuple = Tuple[float, float]

Axis = Union[int, str]
"""Identifier for axis - either the numerical order or the name."""

__all__ = ["RangeTuple", "Axis", "ArrayLike", "DTypeLike"]
