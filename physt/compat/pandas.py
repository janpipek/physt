from typing import Any, List, Optional

import numpy as np
import pandas
from pandas.core.arrays.masked import BaseMaskedDtype
from pandas.api.types import is_numeric_dtype

from physt.binnings import BinningBase
from physt.facade import h, h1, h2
from physt.histogram1d import Histogram1D
from physt.histogram_nd import Histogram2D, HistogramND


def _extract_values(series: pandas.Series, dropna: bool = True) -> np.ndarray:
    if isinstance(series.dtype, BaseMaskedDtype):
        array = series.array
        if not dropna and any(array._mask):
            raise ValueError(
                "Cannot histogram series with NA's. Set `dropna` to True to override."
            )
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
        return h1(
            data=values, name=self._series.name, bins=bins, dropna=False, **kwargs
        )


@pandas.api.extensions.register_dataframe_accessor("physt")
class PhystDataFrameAccessor:
    """Histogramming methods for pandas DataFrames."""

    def __init__(self, df: pandas.DataFrame):
        self._df = df

    def h1(self, col: Any = None, bins=None, **kwargs) -> Histogram1D:
        """Create 1D histogram from a column."""
        if col is None:
            if self._df.shape[1] != 1:
                raise ValueError("Argument `col` must be set.")
            col = self._df.columns[0]
        return self._df[col].physt.h1(bins=bins, **kwargs)

    def h2(
        self,
        col1: Any = None,
        col2: Any = None,
        bins=None,
        *,
        dropna: bool = True,
        **kwargs
    ) -> Histogram2D:
        """Create 2D histogram from two columns.

        Parameters
        ----------
        col1: Name of the first column
        col2: Name of the second column
        bins: Universal bins
        dropna:
        """
        if col1 is None and col2 is None and self._df.shape[1] == 2:
            col1, col2 = self._df.columns
        elif col1 is None or col2 is None:
            raise ValueError("Arguments `col1` and `col2` must be set.")
        data = self._df[[col1, col2]]
        if dropna:
            data = data.dropna()
        return h2(
            data1=_extract_values(data[col1], dropna=False),
            data2=_extract_values(data[col2], dropna=False),
            bins=bins,
            dropna=False,  # Already done
            **kwargs
        )

    def h(self, cols: List[Any], bins=None, *, dropna=True, **kwargs) -> HistogramND:
        data = self._df[cols]
        if dropna:
            data = data.dropna()
        return h(data=data.astype(float).values, bins=bins, **kwargs)


def binning_to_index(
    binning: BinningBase, name: Optional[str] = None
) -> pandas.IntervalIndex:
    """Convert binning to a pandas interval index."""
    # TODO: Check closedness
    return pandas.IntervalIndex.from_arrays(
        left=binning.bins[:, 0], right=binning.bins[:, 1], closed="left", name=name
    )


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