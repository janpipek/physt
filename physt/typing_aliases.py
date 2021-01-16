"""Definitions for type hints."""
from numpy import ndarray
from typing import NewType, Tuple, Union, Iterable

RangeTuple = Tuple[float, float]
Axis = Union[int, str]

# TODO: Update with numpy 1.20
ArrayLike = Union[ndarray, Iterable, int, float]
DtypeLike = NewType("DtypeLike", object)
