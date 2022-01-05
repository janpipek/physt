"""Definitions for type hints."""
from typing import Iterable, Tuple, Type, Union

from numpy import ndarray, dtype
from numpy.typing import ArrayLike, DTypeLike

RangeTuple = Tuple[float, float]
Axis = Union[int, str]

__all__ = ["RangeTuple", "Axis", "ArrayLike", "DTypeLike"]
