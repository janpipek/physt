"""Support for pola.rs library.

pola.rs Series and DataFrames can be passed to h1, ..., h
in the same way as their pandas equivalents.

Examples:
    >>> import polars, physt
    >>> series = polars.Series("x", range(100))
    >>> physt.h1(series)
    Histogram1D(bins=(10,), total=100, dtype=int64)
"""

from typing import Iterable, NoReturn, Optional, Tuple

import numpy as np
import pandas as pd
import polars

from physt._construction import (
    extract_1d_array,
    extract_axis_name,
    extract_axis_names,
    extract_nd_array,
    extract_weights,
)

NUMERIC_POLARS_DTYPES = [
    polars.Int8,
    polars.Int16,
    polars.Int32,
    polars.Int64,
    polars.UInt8,
    polars.UInt16,
    polars.UInt32,
    polars.UInt64,
    polars.Float32,
    polars.Float64,
]


@extract_axis_name.register
def _(data: polars.Series, *, axis_name: Optional[str] = None) -> Optional[str]:
    if axis_name is not None:
        return axis_name
    return data.name


@extract_axis_name.register
def _(data: polars.DataFrame, **kwargs) -> NoReturn:
    raise ValueError("Cannot extract axis name from a polars DataFrame.")


@extract_1d_array.register
def _(
    data: polars.Series, *, dropna: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if data.dtype not in NUMERIC_POLARS_DTYPES:
        raise ValueError(
            f"Cannot extract float array from type {data.dtype}, must be int-like or float-like"
        )
    return extract_1d_array(data.to_numpy(zero_copy_only=True), dropna=dropna)  # type: ignore


@extract_1d_array.register
def _(data: polars.DataFrame, **kwargs) -> NoReturn:
    raise ValueError(
        "Cannot extract 1D array suitable for histogramming from a polars dataframe. "
        "Either select a Series or extract multidimensional data."
    )


@extract_nd_array.register
def _(data: polars.Series, **kwargs) -> NoReturn:
    raise ValueError(
        "Cannot extract multidimensional array suitable for histogramming from a polars series. "
        "Either select a DataFrame or extract 1D data."
    )


@extract_nd_array.register
def _(
    data: polars.DataFrame, *, dim: Optional[int] = None, dropna: bool = True
) -> Tuple[int, np.ndarray, Optional[np.ndarray]]:
    if data.shape[1] == 0:
        raise ValueError("Must have at least one column.")
    pandas_df = pd.DataFrame(
        {key: extract_1d_array(data[key], dropna=False)[0] for key in data.columns}
    )
    return extract_nd_array(pandas_df, dim=dim, dropna=dropna)  # type: ignore


@extract_axis_names.register
def _(
    data: polars.DataFrame, *, axis_names: Optional[Iterable[str]] = None
) -> Optional[Tuple[str, ...]]:
    if axis_names is not None:
        result = tuple(axis_names)
        if (given_length := len(result)) != (expected_length := data.shape[1]):
            raise ValueError(
                f"Explicit {axis_names=} has invalid length {given_length}, {expected_length} expected."
            )
        return result
    return tuple(data.columns)


@extract_axis_names.register
def _(data: polars.Series, **kwargs) -> NoReturn:
    raise ValueError("Cannot extract axis names from a single polars Series.")


@extract_weights.register
def _(data: polars.Series, array_mask: Optional[np.ndarray] = None) -> np.ndarray:
    array, _ = extract_1d_array(data, dropna=False)
    return extract_weights(array, array_mask=array_mask)  # type: ignore


@extract_weights.register
def _(data: polars.DataFrame, **kwargs) -> NoReturn:
    raise ValueError("Cannot extract weights from a polars DataFrame.")
