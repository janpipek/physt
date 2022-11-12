from typing import Iterable, NoReturn, Optional, Tuple

import numpy as np
import polars

from physt._construction import (
    extract_1d_array,
    extract_axis_name,
    extract_axis_names,
    extract_nd_array,
)


@extract_axis_name.register
def _(data: polars.Series, *, axis_name: str) -> Optional[str]:
    if axis_name is not None:
        return axis_name
    return data.name


@extract_axis_name.register
def _(data: polars.DataFrame, **kwargs) -> NoReturn:
    raise ValueError("Cannot extract axis name from a polars DataFrame.")


@extract_1d_array.register
def _(data: polars.Series, *, dropna: bool = True) -> tuple[np.ndarray, Optional[np.ndarray]]:
    if polars.datatypes.dtype_to_py_type(data.dtype) not in [int, float]:
        raise ValueError(
            f"Cannot extract float array from type {data.dtype}, must be int-like or float-like"
        )
    return extract_1d_array(data.view(), dropna=dropna)


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
) -> tuple[int, np.ndarray, Optional[np.ndarray]]:
    pandas_df = data.to_pandas().astype(float)
    return extract_nd_array(pandas_df, dim=dim, dropna=dropna)


@extract_axis_names.register
def _(
    data: polars.DataFrame, *, axis_names: Optional[Iterable[str]] = None
) -> Optional[tuple[str, ...]]:
    if axis_names is not None:
        result = tuple(axis_names)
        if (given_length := len(result)) != (expected_length := data.shape[1]):
            raise ValueError(
                f"Explicit {axis_names=} has invalid length {given_length}, {expected_length} expected."
            )
    return tuple(data.columns)


@extract_axis_names.register
def _(data: polars.Series, **kwargs) -> NoReturn:
    raise ValueError("Cannot extract axis names from a single polars Series.")
