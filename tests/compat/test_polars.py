from typing import Tuple

import hypothesis.strategies as st
import numpy as np
import polars
import pytest
from hypothesis import assume, given
from hypothesis.extra.numpy import array_shapes, arrays
from numpy.testing import assert_array_equal
from polars.testing.parametric import dataframes, series

from physt import h, h1
from physt._construction import (
    extract_1d_array,
    extract_axis_name,
    extract_axis_names,
    extract_nd_array,
    extract_weights,
)
from physt.compat.polars import NUMERIC_POLARS_DTYPES
from physt.types import Histogram1D, HistogramND


class TestH1:
    @given(data=series(allowed_dtypes=NUMERIC_POLARS_DTYPES, allow_infinities=False))
    def test_with_series(self, data):
        assume(polars.n_unique(data) >= 2)
        result = h1(data)
        assert isinstance(result, Histogram1D)


class TestH:
    @given(data=dataframes(allowed_dtypes=NUMERIC_POLARS_DTYPES, allow_infinities=False))
    def test_with_dataframe(self, data):
        assume(all(polars.n_unique(data[col]) >= 2 for col in data.columns))
        result = h(data)
        assert isinstance(result, HistogramND)


class TestExtraNDArray:
    @given(data=dataframes(allowed_dtypes=NUMERIC_POLARS_DTYPES))
    def test_same_result_as_with_arrays(self, data):
        extract_nd_array(data)

    def test_fails_with_wrong_types(self):
        pass

    def test_fails_with_series(self):
        series = polars.Series(values=[1, 2, 3, 4, 5])
        with pytest.raises(
            ValueError,
            match="Cannot extract multidimensional array suitable for histogramming from a polars series",
        ):
            extract_nd_array(series)


class TestExtract1DArray:
    @given(
        values=st.lists(
            st.integers(min_value=-1_000_000_000, max_value=1_000_000_000) | st.floats()
        ),
        dropna=st.booleans(),
    )
    def test_with_series(self, dropna, values):
        pl_input = polars.Series(values=values)
        nd_input = np.array(values)

        pl_array, pl_mask = extract_1d_array(pl_input, dropna=dropna)
        nd_array, nd_mask = extract_1d_array(nd_input, dropna=dropna)

        assert_array_equal(pl_array, nd_array)
        if dropna:
            assert_array_equal(pl_mask, nd_mask)
        else:
            assert pl_mask is None

    @given(data=series(excluded_dtypes=NUMERIC_POLARS_DTYPES))
    def test_fails_with_wrong_type(self, data):
        # See https://pola-rs.github.io/polars/py-polars/html/reference/datatypes.html
        series = polars.Series(data)
        with pytest.raises(ValueError, match="Cannot extract float array from type"):
            extract_1d_array(series)

    def test_fails_with_dataframes(self):
        import polars

        df = polars.DataFrame()
        # TODO: Or should it be a type error?
        with pytest.raises(
            ValueError,
            match="Cannot extract 1D array suitable for histogramming from a polars dataframe",
        ):
            extract_1d_array(df)


class TestExtractAxisName:
    @given(data=series())
    def test_uses_polars_names(self, data: polars.Series):
        assert data.name == extract_axis_name(data)

    @given(data=dataframes())
    def test_fails_with_dataframe(self, data):
        with pytest.raises(ValueError, match="Cannot extract axis name from a polars DataFrame."):
            extract_axis_name(data)


class TestExtractAxisNames:
    @given(data=series())
    def test_fails_with_series(self, data):
        with pytest.raises(
            ValueError, match="Cannot extract axis names from a single polars Series"
        ):
            extract_axis_names(data)

    @given(data=dataframes())
    def test_uses_polars_names(self, data: polars.DataFrame):
        # TODO: Test with explicit axis_names
        assert tuple(data.columns) == extract_axis_names(data)


@st.composite
def series_and_mask(
    draw, *, min_length: int = 0, max_length: int = 10, allowed_dtypes=NUMERIC_POLARS_DTYPES
) -> Tuple[polars.Series, np.ndarray]:
    length = draw(st.integers(min_value=min_length, max_value=max_length))
    mask = draw(arrays(shape=(length,), dtype=bool))
    s = draw(series(size=length, allowed_dtypes=allowed_dtypes))
    return s, mask


class TestExtractWeights:
    @given(data=series(allowed_dtypes=NUMERIC_POLARS_DTYPES))
    def test_selects_full_series_without_mask(self, data: polars.Series):
        result = extract_weights(data, array_mask=None)
        assert_array_equal(result, data.to_numpy())

    @given(data_and_mask=series_and_mask())
    def test_selects_with_mask(self, data_and_mask):
        data, array_mask = data_and_mask
        result = extract_weights(data, array_mask=array_mask)
        assert isinstance(result, np.ndarray)

    @given(
        data=series(allowed_dtypes=NUMERIC_POLARS_DTYPES),
        array_mask=arrays(dtype=bool, shape=array_shapes(min_dims=1, max_dims=1)),
    )
    def test_fails_with_shape_mismatch(self, data: polars.Series, array_mask: np.ndarray):
        assume(array_mask.shape != data.shape)
        with pytest.raises(ValueError, match="Weights array shape"):
            extract_weights(data, array_mask=array_mask)

    @given(data=series(excluded_dtypes=NUMERIC_POLARS_DTYPES))
    def test_fails_with_wrong_type(self, data):
        with pytest.raises(ValueError, match="Cannot extract float array from type"):
            extract_weights(data)

    @given(
        data=dataframes(),
        array_mask=st.none() | arrays(dtype=bool, shape=array_shapes(max_dims=1)),
    )
    def test_fails_with_dataframe(self, data: polars.DataFrame, array_mask):
        with pytest.raises(ValueError, match="Cannot extract weights from a polars DataFrame"):
            extract_weights(data, array_mask=array_mask)
