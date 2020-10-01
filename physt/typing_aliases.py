"""Definitions for type hints."""
from typing import Dict, Tuple, Union, NewType, Any

RangeTuple = Tuple[float, float]
Axis = Union[int, str]

# Aliases that are hard to define
DtypeLike = NewType("DtypeLike", Any)
ArrayLike = NewType("ArrayLike", Any)
BinningLike = NewType("BinningLike", Any)

MetaData = Dict[str, Any]
