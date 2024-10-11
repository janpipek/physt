"""Support for polars library.

polars Series and DataFrames can be passed to h1, ..., h
in the same way as their pandas equivalents.

Note that by default, we drop NAs, but not nulls.
Histogramming a column with nulls will result in an error.

Examples:
    >>> import polars, physt
    >>> series = polars.Series("x", range(100))
    >>> physt.h1(series)
    Histogram1D(bins=(10,), total=100, dtype=int64)
"""

# TODO: Support structures with numerical items

from typing import Any, Iterable, NoReturn, Optional, Tuple, Union

import numpy as np
import pandas as pd
import polars

import physt
from physt._construction import (
    extract_1d_array,
    extract_axis_name,
    extract_axis_names,
    extract_nd_array,
    extract_weights,
)
from physt.types import Histogram1D, HistogramND

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
    if data.is_null().any():
        raise ValueError("Cannot create histogram from series with nulls")
    return extract_1d_array(data.to_numpy(allow_copy=True), dropna=dropna)  # type: ignore


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
    # TODO: This is not very optimized
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


@polars.api.register_series_namespace("physt")
class PhystSeries:
    def __init__(self, series: polars.Series):
        # TODO: Check numeric dtypes!
        self._series = series

    def h1(self, bins: Any = None, **kwargs) -> Histogram1D:
        """Create a 1D histogram from the Series.

        :param bins: Binning specification. If None, default bins are used.
        """
        return physt.h1(self._series, bins=bins, **kwargs)


@polars.api.register_dataframe_namespace("physt")
class PhystFrame:
    def __init__(self, df: polars.DataFrame):
        self._df = df

    def h(
        self,
        *selectors: Any,
        bins: Any = None,
        **kwargs,
    ) -> Union[Histogram1D, HistogramND]:
        """Create a histogram from the DataFrame.

        :param selectors: Any selectors. If none, all numeric columns are used.
        :param bins: Binning specification. If None, default bins are used.
        """

        if not selectors:
            selectors = (polars.selectors.numeric(),)
        data = self._df.select(*selectors)
        columns = data.columns

        if len(columns) < 1:
            raise KeyError("No columns selected for histogramming.")

        if isinstance(columns, str):
            columns = [columns]
        else:
            columns = list(columns)

        if len(columns) == 1:
            return physt.h1(data, bins=bins, **kwargs)
        if len(columns) == 2:
            return physt.h2(data[columns[0]], data[columns[1]], bins=bins, **kwargs)
        # TODO: Check numeric dtypes ?
        return physt.h(data, bins=bins, **kwargs)
