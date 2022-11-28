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

NUMERIC_POLARS_DTYPES = tuple(
    dtype
    for dtype, py_type in polars.datatypes.DataTypeMappings.DTYPE_TO_PY_TYPE.items()
    if py_type in (int, float)
)


@extract_axis_name.register
def _(data: polars.Series, *, axis_name: Optional[str] = None) -> Optional[str]:
    if axis_name is not None:
        return axis_name
    return data.name


@extract_axis_name.register
def _(data: polars.DataFrame, **kwargs) -> NoReturn:
    raise ValueError("Cannot extract axis name from a polars DataFrame.")


@extract_1d_array.register
def _(data: polars.Series, *, dropna: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if data.dtype not in NUMERIC_POLARS_DTYPES:
        raise ValueError(
            f"Cannot extract float array from type {data.dtype}, must be int-like or float-like"
        )
    return extract_1d_array(data.view(), dropna=dropna)  # type: ignore


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
    return extract_weights(array, array_mask=array_mask)


@extract_weights.register
def _(data: polars.DataFrame, **kwargs) -> NoReturn:
    raise ValueError("Cannot extract weights from a polars DataFrame.")
