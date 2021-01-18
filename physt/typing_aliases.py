"""Definitions for type hints."""
from typing import Tuple, Union, Iterable

from numpy import ndarray, dtype

RangeTuple = Tuple[float, float]
Axis = Union[int, str]

# TODO: Update with numpy 1.20
ArrayLike = Union[ndarray, Iterable, int, float]
DtypeLike = Union[type, dtype, str]
