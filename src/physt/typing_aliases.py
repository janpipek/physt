"""Definitions for type hints."""

from numpy.typing import ArrayLike, DTypeLike

RangeTuple = tuple[float, float]

Axis = int | str
"""Identifier for axis - either the numerical order or the name."""

__all__ = ["RangeTuple", "Axis", "ArrayLike", "DTypeLike"]
