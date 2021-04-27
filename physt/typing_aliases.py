"""Definitions for type hints."""
from typing import Iterable, Tuple, Type, Union

from numpy import ndarray, dtype

RangeTuple = Tuple[float, float]
Axis = Union[int, str]

ArrayLike = Union[ndarray, Iterable, int, float]
DtypeLike = Union[type, dtype, str]

# ArrayLike: Type
# DtypeLike: Type

# if np.__version__ >= "1.20":
#     ArrayLike = typing.ArrayLike
#     DTypeLike = typing.DTypeLike
# else:
#     ArrayLike = Union[ndarray, Iterable, int, float]
#     DTypeLike = Union[type, dtype, str]

__all__ = ["RangeTuple", "Axis", "ArrayLike", "DtypeLike"]
