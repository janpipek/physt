"""Various utility functions to support physt implementation.

These functions are mostly general Python functions, not specific
for numerical computing, histogramming, etc.
"""
from __future__ import annotations

import warnings
from functools import singledispatch, wraps
from typing import TYPE_CHECKING, Iterable, List, Optional

import numpy as np

if TYPE_CHECKING:
    from typing import Any, Callable, Dict, Tuple


def all_subclasses(cls: type) -> Tuple[type, ...]:
    """All subclasses of a class.

    From: http://stackoverflow.com/a/17246726/2692780
    """
    subclasses = []
    for subclass in cls.__subclasses__():
        subclasses.append(subclass)
        subclasses.extend(all_subclasses(subclass))
    return tuple(subclasses)


def find_subclass(base: type, name: str) -> type:
    """Find a named subclass of a base class.

    Uses only the class name without namespace.
    """
    class_candidates = [klass for klass in all_subclasses(base) if klass.__name__ == name]
    if not class_candidates:
        raise TypeError(f"No '{base.__name__}' subclass of '{name}'.")
    if len(class_candidates) > 1:
        raise TypeError(f"Multiple '{base.__name__}' subclasses of '{name}'.")
    return class_candidates[0]


def pop_many(a_dict: Dict[str, Any], *args: str, **kwargs) -> Dict[str, Any]:
    """Pop multiple items from a dictionary.

    Parameters
    ----------
    a_dict : Dictionary from which the items will popped
    args: Keys which will be popped (and not included if not present)
    kwargs: Keys + default value pairs (if key not found, this default is included)

    Returns
    -------
    A dictionary of collected items.
    """
    result = {}
    for arg in args:
        if arg in a_dict:
            result[arg] = a_dict.pop(arg)
    for key, value in kwargs.items():
        result[key] = a_dict.pop(key, value)
    return result


def deprecation_alias(f: Callable, deprecated_name: str) -> Callable:
    """Provide a deprecated copy of a function.

    Parameters
    ----------
    f : The correct function
    deprecated_name : The name the function will be given

    Examples
    --------
    >>> def new(x): return 1
    >>> old = deprecation_alias(new, "old")
    """

    @wraps(f)
    def inner(*args, **kwargs):
        warnings.warn(
            f"{deprecated_name} is deprecated, use {f.__name__} instead",
            FutureWarning,
        )
        return f(*args, **kwargs)

    return inner


# TODO: The following functions enable to structure h1,h2 ... as a set of individual
#   separately tested steps. Think about where it belongs.


@singledispatch
def extract_1d_array(data: Any, *, dropna: bool = True) -> Optional[np.ndarray]:
    array: np.ndarray = np.asarray(data)
    if dropna:
        array_mask = ~np.isnan(array)
        array = array[array_mask]
    else:
        array_mask = None
    return array, array_mask


@extract_1d_array.register
def _(data: None, *, dropna=True):
    return None, None


@singledispatch
def extract_nd_array(
    data: Any, *, dim: Optional[int], dropna: bool = True
) -> Tuple[int, Optional[np.ndarray], Optional[np.ndarray]]:
    array: np.ndarray = np.asarray(data)
    if array.ndim != 2:
        raise ValueError(f"Array must have shape (n, d), {array.shape} encountered.")
    if dim is not None and dim != array.shape[1]:
        raise ValueError(f"Dimension mismatch: {dim} != {array.shape[1]}")
    _, dim = array.shape
    # TODO: This might not work with weights!
    if dropna:
        array_mask = ~np.isnan(array).any(axis=1)
        array = array[array_mask]
    else:
        array_mask = None
    return dim, array, array_mask


@extract_nd_array.register
def _(data: None, *, dim: int, dropna: bool = True):
    if dim is None:
        raise ValueError("You have to specify either data or its dimension.")
    return dim, None, None


@singledispatch
def extract_axis_name(data: Any, *, axis_name: Optional[str] = None) -> Optional[str]:
    """For input data, find the axis name."""
    if not axis_name:
        if hasattr(data, "name"):
            return str(data.name)  # type: ignore
        elif (
            hasattr(data, "fields")
            and len(data.fields) == 1  # type: ignore
            and isinstance(data.fields[0], str)  # type: ignore
        ):
            # Case of dask fields (examples)
            return str(data.fields[0])  # type: ignore
    return axis_name


@singledispatch
def extract_axis_names(
    data: Any, *, axis_names: Optional[Iterable[str]] = None
) -> Optional[Tuple[str, ...]]:
    """For input data, find the names for axes."""
    if axis_names is not None:
        return tuple(axis_names)
    if hasattr(data, "columns"):
        return tuple(str(c) for c in data.columns)  # type: ignore
    return None


@singledispatch
def extract_weights(weights: Any, array_mask: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    if weights is None:
        return None
    weights_array = np.asarray(weights)
    if array_mask is not None:
        weights_array = weights_array[array_mask]
    return weights_array
