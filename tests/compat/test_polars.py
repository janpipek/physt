from datetime import date, datetime

import hypothesis.strategies as st
import numpy as np
import polars
import polars.testing.parametric
import pytest
from hypothesis import given
from hypothesis.extra.numpy import array_shapes, arrays
from numpy.testing import assert_array_equal

from physt._construction import (
    extract_1d_array,
    extract_axis_name,
    extract_axis_names,
    extract_nd_array,
    extract_weights,
)

NUMERIC_POLARS_DTYPES = [
    dtype
    for dtype, py_type in polars.datatypes.DataTypeMappings.DTYPE_TO_PY_TYPE.items()
    if py_type in (int, float)
]


class TestExtraNDArray:
    @given(data=polars.testing.parametric.dataframes(allowed_dtypes=NUMERIC_POLARS_DTYPES))
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

    @pytest.mark.parametrize(
        "data",
        [
            pytest.param(["abc", "def"], id="Utf8"),
            pytest.param([datetime(2020, 1, 1)], id="Datetime"),
            pytest.param([date(2020, 1, 1)], id="Date"),
            pytest.param([[1, 2], [1, 3]], id="List"),
        ],
    )
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
    @given(data=polars.testing.parametric.series())
    def test_uses_polars_names(self, data: polars.Series):
        assert data.name == extract_axis_name(data)

    @given(data=polars.testing.parametric.dataframes())
    def test_fails_with_dataframe(self, data):
        with pytest.raises(ValueError, match="Cannot extract axis name from a polars DataFrame."):
            extract_axis_name(data)


class TestExtractAxisName:
    @given(data=polars.testing.parametric.series())
    def test_fails_with_series(self, data):
        with pytest.raises(
            ValueError, match="Cannot extract axis names from a single polars Series"
        ):
            extract_axis_names(data)

    @given(data=polars.testing.parametric.dataframes())
    def test_uses_polars_names(self, data: polars.DataFrame):
        # TODO: Test with explicit axis_names
        assert tuple(data.columns) == extract_axis_names(data)


class TestExtractWeights:
    @given(data=polars.testing.parametric.series(allowed_dtypes=NUMERIC_POLARS_DTYPES))
    def test_selects_full_series_without_mask(self, data: polars.Series):
        result = extract_weights(data, array_mask=None)
        assert_array_equal(result, data.to_numpy())

    @given(
        data=polars.testing.parametric.dataframes(),
        array_mask=st.none() | arrays(dtype=bool, shape=array_shapes(max_dims=1)),
    )
    def test_fails_with_dataframe(self, data: polars.DataFrame, array_mask):
        with pytest.raises(ValueError, match="Cannot extract weights from a polars DataFrame"):
            extract_weights(data, array_mask=array_mask)
