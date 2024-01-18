"""Pandas integration.

- conversion between histograms and Series/DataFrames
- .physt accessor for pandas objects
"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, NoReturn, Optional, Tuple, cast

import numpy as np
import pandas
import pandas as pd
from pandas.api.types import is_numeric_dtype

from physt._construction import calculate_1d_bins, extract_1d_array, extract_nd_array
from physt._facade import h, h1
from physt.binnings import BinningBase, static_binning
from physt.types import Histogram1D, Histogram2D, HistogramND

if TYPE_CHECKING:
    from typing import Any, Union

    from physt.typing_aliases import ArrayLike


@extract_1d_array.register
def _(
    series: pandas.Series, *, dropna: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if not pd.api.types.is_numeric_dtype(series):
        raise ValueError(
            f"Cannot extract suitable array from non-numeric dtype: {series.dtype}"
        )
    series = series.astype(float)
    # if isinstance(series.dtype, BaseMaskedDtype):
    #     array = cast(BaseMaskedArray, series.array)
    #     if not dropna and any(array.mask):
    #         raise ValueError("Cannot histogram series with NA's. Set `dropna` to True to override.")
    #     array_mask = ~array._mask
    #     array = array._data[~array._mask]
    if dropna:
        array_mask = series.notna().values
        array = series.dropna().values
    else:
        array_mask = None
        array = series.values
    return array, array_mask


@extract_1d_array.register
def _(dataframe: pd.DataFrame, **kwargs) -> NoReturn:
    # TODO: What about dataframes with just one column?
    raise ValueError(
        "Cannot extract 1D array suitable for histogramming from a dataframe. "
        "Either select a Series or extract multidimensional data."
    )


@extract_nd_array.register
def _(series: pd.Series, **kwargs) -> NoReturn:
    raise ValueError(
        "Cannot extract multidimensional array suitable for histogramming from a series. "
        "Either select a DataFrame or extract 1D data."
    )


@extract_nd_array.register
def _(
    data_frame: pd.DataFrame, *, dim: Optional[int] = None, dropna: bool = True
) -> Tuple[int, np.ndarray, Optional[np.ndarray]]:
    if non_numeric_columns := [
        name for name, series in data_frame.items() if not is_numeric_dtype(series)
    ]:
        raise ValueError(f"Cannot histogram non-numeric columns: {non_numeric_columns}")
    if dim and dim != data_frame.shape[1]:
        raise ValueError(f"Invalid dim {data_frame.shape[1]}, {dim} expected.")
    if dropna:
        array_mask = data_frame.isna().any().values
        data_frame = data_frame.dropna()
    else:
        array_mask = None
    array = data_frame.astype(float).values
    return data_frame.shape[1], array, array_mask


@pandas.api.extensions.register_series_accessor("physt")
class PhystSeriesAccessor:
    """Histogramming methods for pandas Series.

    It exists only for numeric series.
    """

    def __init__(self, series: pandas.Series):
        if not is_numeric_dtype(series):
            raise AttributeError(
                f"Series must be of a numeric type, not {series.dtype}"
            )
        self._series = series

    def h1(self, bins=None, **kwargs) -> Histogram1D:
        """Create a histogram from the series."""
        return h1(self._series, bins=bins, **kwargs)

    histogram = h1

    def cut(self, bins=None, **kwargs) -> pd.Series:
        """Bin values using physt binning (eq. to pd.cut)."""
        warnings.warn(
            "This method is experimental, only partially implemented and may removed."
        )
        binning = calculate_1d_bins(
            extract_1d_array(self._series, dropna=True)[0], bins, **kwargs
        )
        return pd.cut(self._series, binning.numpy_bins)


@pandas.api.extensions.register_dataframe_accessor("physt")
class PhystDataFrameAccessor:
    """Histogramming methods for pandas DataFrames."""

    def __init__(self, df: pandas.DataFrame):
        self._df = df

    def h1(
        self,
        column: Any = None,
        bins=None,
        *,
        weights: Union[ArrayLike, str, None] = None,
        **kwargs,
    ) -> Histogram1D:
        """Create 1D histogram from a column.

        Parameters
        ----------
        column: Name of the column to apply on (not required for 1-column data frames)
        bins: Universal `bins` argument
        weights: Name of the column to use for weight or some arraylike object

        See Also
        --------
        physt.h1
        """
        if column is None:
            if self._df.shape[1] != 1:
                raise ValueError("Argument `column` must be set.")
            column = self._df.columns[0]
        try:
            data = self._df[column]
        except KeyError as exc:
            raise KeyError(f"Column '{column}' not found.") from exc
        if not isinstance(data, pd.Series):
            raise ValueError(f"Argument `column` must select a single series: {column}")
        if isinstance(weights, str) and weights in self._df.columns:
            # TODO: This might be wrong if NAs are in play
            weights = self._df[weights]
        if not is_numeric_dtype(data):
            raise ValueError(f"Column '{column}' is not numeric.")
        return data.physt.h1(bins=bins, weights=weights, **kwargs)

    def h2(
        self, column1: Any = None, column2: Any = None, bins=None, **kwargs
    ) -> Histogram2D:
        """Create 2D histogram from two columns.

        Parameters
        ----------
        column1: Name of the first column (not required for 2-column data frames)
        column2: Name of the second column (not required for 2-column data frames)
        bins: Universal `bins` argument
        dropna: Ignore NA values

        See Also
        --------
        physt.h2
        """
        if self._df.shape[1] < 2:
            raise ValueError("At least two columns required for 2D histograms.")
        if column1 is None and column2 is None and self._df.shape[1] == 2:
            column1, column2 = self._df.columns
        elif column1 is None or column2 is None:
            raise ValueError("Arguments `column1` and `column2` must be set.")
        return cast(
            Histogram2D, self.histogram([column1, column2], bins=bins, **kwargs)
        )

    def histogram(self, columns: Any = None, bins: Any = None, **kwargs) -> HistogramND:
        """Create a histogram.

        Parameters
        ----------
        columns: The column(s) to apply on. Uses all columns if not set. It can be
            a `str` for one column, `tuple` for a multi-level index, `list` for
            more columns, everything that pandas item selection supports.
        bins: Argument to be passed to find the proper binnings.

        Returns
        -------
        A histogram with dimensionality depending on the final set of columns.

        See Also
        --------
        physt.h
        """
        if columns is None:
            columns = self._df.columns
        try:
            data = self._df[columns]
        except KeyError as exc:
            raise KeyError(
                f"At least one of the columns '{columns}' could not be found."
            ) from exc
        if isinstance(data, pd.Series) or data.shape[1] == 1:
            return data.physt.h1(bins, **kwargs)
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"Argument `columns` does not select a DataFrame: '{columns}'"
            )
        if not data.shape[1]:
            raise ValueError("Cannot make histogram from DataFrame with no columns.")
        for column in data.columns:
            if not is_numeric_dtype(data[column]):
                raise ValueError(f"Column '{column}' is not numeric")
        # TODO: Enable weights to be a name of the column
        # TODO: Unify for masked arrays
        return h(data=data.astype(float), bins=bins, **kwargs)


def binning_to_index(
    binning: BinningBase, name: Optional[str] = None
) -> pandas.IntervalIndex:
    """Convert physt binning to a pandas interval index."""
    # TODO: Check closedness
    return pandas.IntervalIndex.from_arrays(
        left=binning.bins[:, 0], right=binning.bins[:, 1], closed="left", name=name
    )


def index_to_binning(index: pandas.IntervalIndex) -> BinningBase:
    """Convert an interval index into physt binning."""
    if not isinstance(index, pandas.IntervalIndex):
        raise TypeError(f"IntervalIndex required, '{type(index)}' passed.")
    if not index.closed_left:
        raise ValueError("Only `closed_left` indices supported.")
    if index.is_overlapping:
        raise ValueError("Intervals cannot overlap.")
    bins = np.hstack(
        [index.left.values[:, np.newaxis], index.right.values[:, np.newaxis]]
    )
    return static_binning(bins=bins)


def _h1_to_dataframe(h1: Histogram1D) -> pandas.DataFrame:
    """Convert histogram to pandas DataFrame."""
    return pandas.DataFrame(
        {"frequency": h1.frequencies, "error": h1.errors},
        index=binning_to_index(h1.binning, name=h1.name),
    )


def _h1_to_series(h1: Histogram1D) -> pandas.Series:
    """Convert histogram to pandas Series."""
    return pandas.Series(
        h1.frequencies,
        name="frequency",
        index=binning_to_index(h1.binning, name=h1.name),
    )


setattr(Histogram1D, "to_dataframe", _h1_to_dataframe)
setattr(Histogram1D, "to_series", _h1_to_series)


# TODO: Implement multidimensional binning to index
# TODO: Implement multidimensional histogram to series/dataframe
# TODO: Implement histogram collection to series/dataframe
# TODO: Implement histogram collection from dataframe / groupby ?
# TODO: Implement multidimensional index to binning
