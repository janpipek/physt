"""Pandas integration.

- conversion between histograms and Series/DataFrames
- .physt accessor for pandas objects
"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas
import pandas as pd
from pandas.core.arrays.masked import BaseMaskedDtype
from pandas.api.types import is_numeric_dtype

from physt.binnings import BinningBase, calculate_bins, static_binning
from physt.facade import h, h1, h2
from physt.types import HistogramBase, Histogram2D, Histogram1D


if TYPE_CHECKING:
    from typing import Any, Optional, Union
    from physt.typing_aliases import ArrayLike


def _extract_values(series: pandas.Series, dropna: bool = True) -> np.ndarray:
    if isinstance(series.dtype, BaseMaskedDtype):
        array = series.array
        if not dropna and any(array._mask):
            raise ValueError("Cannot histogram series with NA's. Set `dropna` to True to override.")
        return array._data[~array._mask]
    return series.values


@pandas.api.extensions.register_series_accessor("physt")
class PhystSeriesAccessor:
    """Histogramming methods for pandas Series.

    It exists only for numeric series.
    """

    def __init__(self, series: pandas.Series):
        if not is_numeric_dtype(series):
            raise AttributeError(f"Series must be of a numeric type, not {series.dtype}")
        self._series = series

    def h1(self, bins=None, *, dropna: bool = True, **kwargs) -> Histogram1D:
        values = _extract_values(self._series, dropna=dropna)
        return h1(data=values, name=self._series.name, bins=bins, dropna=False, **kwargs)

    histogram = h1

    def cut(self, bins=None, *, dropna: bool = True, **kwargs) -> pd.Series:
        warnings.warn("This method is experimental, only partially implemented and may removed.")
        binning = calculate_bins(_extract_values(self._series, dropna=dropna), bins)
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
            weights = self._df[weights]
        if not is_numeric_dtype(data):
            raise ValueError(f"Column '{column}' is not numeric.")
        if "axis_name" not in kwargs:
            kwargs["axis_name"] = column
        return data.physt.h1(bins=bins, weights=weights, **kwargs)

    def h2(
        self, column1: Any = None, column2: Any = None, bins=None, *, dropna: bool = True, **kwargs
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
        try:
            data = self._df[[column1, column2]]
        # TODO: Enable weights to be a name of the column
        except KeyError as exc:
            raise KeyError(f"Column(s) '{column1}' and/or '{column2}' could not be found.") from exc
        if not is_numeric_dtype(data[column1]):
            raise ValueError(f"Column '{column1}' is not numeric.")
        if not is_numeric_dtype(data[column2]):
            raise ValueError(f"Column '{column2}' is not numeric.")
        if dropna:
            data = data.dropna()
        if "axis_names" not in kwargs:
            kwargs["axis_names"] = (column1, column2)
        return h2(
            data1=_extract_values(data[column1], dropna=False),
            data2=_extract_values(data[column2], dropna=False),
            bins=bins,
            dropna=False,  # Already done
            **kwargs,
        )

    def histogram(
        self, columns: Any = None, bins: Any = None, *, dropna: bool = True, **kwargs
    ) -> HistogramBase:
        """Create a histogram.

        Parameters
        ----------
        columns: The column(s) to apply on. Uses all columns if not set. It can be
            a `str` for one column, `tuple` for a multi-level index, `list` for
            more columns, everything that pandas item selection supports.

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
            raise KeyError(f"At least one of the columns '{columns}' could not be found.") from exc
        if isinstance(data, pd.Series) or data.shape[1] == 1:
            return data.physt.h1(bins, dropna=dropna, **kwargs)
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"Argument `columns` does not select a DataFrame: '{columns}'")
        if dropna:
            data = data.dropna()
        if not data.shape[1]:
            raise ValueError("Cannot make histogram from DataFrame with no columns.")
        for column in data.columns:
            if not is_numeric_dtype(data[column]):
                raise ValueError(f"Column '{column}' is not numeric")
        # TODO: Enable weights to be a name of the column
        if "axis_names" not in kwargs:
            kwargs["axis_names"] = data.columns.tolist()
        return h(data=data.astype(float).values, bins=bins, **kwargs)


def binning_to_index(binning: BinningBase, name: Optional[str] = None) -> pandas.IntervalIndex:
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
    bins = np.hstack([index.left.values[:, np.newaxis], index.right.values[:, np.newaxis]])
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
