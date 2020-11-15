"""Definitions for type hints."""
from typing import NewType, Tuple, Union

RangeTuple = Tuple[float, float]
Axis = Union[int, str]

# TODO: Update with numpy 1.20
ArrayLike = NewType("ArrayLike", object)
DtypeLike = NewType("DtypeLike", object)
