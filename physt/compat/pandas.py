from typing import Any, List, Optional

import pandas
from pandas.core.arrays.masked import BaseMaskedDtype

from physt.binnings import BinningBase
from physt.facade import h, h1, h2
from physt.histogram1d import Histogram1D
from physt.histogram_nd import Histogram2D, HistogramND


@pandas.api.extensions.register_series_accessor("physt")
class PhystSeriesAccessor:
    """Histogramming methods for pandas Series."""
    def __init__(self, series: pandas.Series):
        self._series = series

    def h1(self, bins=None, *, dropna: bool = True, **kwargs) -> Histogram1D:
        if isinstance(self._series.dtype, BaseMaskedDtype):
            array = self._series.array
            if not dropna and any(array._mask):
                raise ValueError("Cannot histogram series with NA's. Set `dropna` to True to override.")
            values = array._data[~array._mask]
        else:
            values = self._series.values
        return h1(data=values, name=self._series.name, bins=bins, dropna=False, **kwargs)


@pandas.api.extensions.register_dataframe_accessor("physt")
class PhystDataFrameAccessor:
    """Histogramming methods for pandas DataFrames."""

    def __init__(self, df: pandas.DataFrame):
        self._df = df

    def h1(self, col: Any = None, bins=None, **kwargs) -> Histogram1D:
        if col is None:
            if self._df.shape[1] != 1:
                raise ValueError("Argument `col` must be set.")
            col = self._df.columns[0]
        return self[col].physt.h1(bins=bins, **kwargs)

    def h2(self, col1: Any = None, col2: Any = None, bins=None, **kwargs) -> Histogram2D:
        if col1 is None and col2 is None and self._df.shape[1] == 2:
            col1, col2 = self._df.columns
        elif col1 is None or col2 is None:
            raise ValueError("Arguments `col1` and `col2` must be set.")
        return h2(data1=self._df[col1], data2=self._df[col2], bins=bins, **kwargs)

    def h(self, cols: List[Any], bins=None, **kwargs) -> HistogramND:
        return h(data=self._df[cols], bins=bins, **kwargs)


def binning_to_index(binning: BinningBase, name: Optional[str] = None) -> pandas.IntervalIndex:
    return pandas.IntervalIndex.from_arrays(
        left=binning.bins[:,0], right=binning.bins[:,1], closed="left", name=name
    )


def h1_to_dataframe(h1: Histogram1D) -> pandas.DataFrame:
    """Convert histogram to pandas DataFrame."""
    return pandas.DataFrame(
        { "frequencies": h1.frequencies, "errors": h1.errors},
        index=binning_to_index(h1.binning, name=h1.name),
    )


def h1_to_series(h1: Histogram1D) -> pandas.Series:
    """Convert histogram to pandas Series."""
    return pandas.Series(
        h1.frequencies,
        index=binning_to_index(h1.binning, name=h1.name),
    )


setattr(Histogram1D, "to_dataframe", h1_to_dataframe)
setattr(Histogram1D, "to_series", h1_to_series)