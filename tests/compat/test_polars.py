import sys
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

if sys.version_info[:2] <= (3, 8):
    # We will not fix this as EOL for Python 3.8 is 2024-10-14
    pytest.skip(
        "To run all tests, we would need to explicitly require the zoneinfo module",
        allow_module_level=True,
    )


@pytest.fixture
def series_of_int() -> polars.Series:
    return polars.Series("series_of_int", [0, 1, 2, 3, 4, 5])


@pytest.fixture
def series_with_null() -> polars.Series:
    return polars.Series("series_with_null", [0, 1, None, 3, 4, 5])


@pytest.fixture
def simple_data_frame() -> polars.DataFrame:
    return polars.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": [5, 6, 7, 8],
        }
    )


@pytest.fixture
def dataframe_with_null(series_with_null, series_of_int) -> polars.DataFrame:
    return polars.DataFrame([series_with_null, series_of_int])


class TestH1:
    # Just check that the whole construction works.
    # More detailed tests for individual steps below.

    @given(
        data=series(
            allowed_dtypes=NUMERIC_POLARS_DTYPES,
            allow_infinities=False,
            allow_null=False,
            min_size=2,
        )
    )
    def test_with_series(self, data):
        assume(np.inf > (data.max() - data.min()) > 0)
        result = h1(data)
        assert isinstance(result, Histogram1D)

    def test_with_null_series(self, series_with_null):
        with pytest.raises(
            ValueError, match="Cannot create histogram from series with nulls"
        ):
            h1(series_with_null)


class TestH:
    # Just check that the whole construction works.
    # More detailed tests for individual steps below.

    @given(
        data=dataframes(
            allowed_dtypes=NUMERIC_POLARS_DTYPES,
            allow_infinities=False,
            allow_null=False,
            allow_nan=False,
            min_cols=1,
            max_cols=4,
            min_size=2,
            max_size=6,
        )
    )
    def test_with_dataframe(self, data):
        # Any better way of getting rid of NAs?
        data = data.fill_nan(0)
        assume(
            all(
                (np.inf > (data[col].max() - data[col].min()) > 0)
                for col in data.columns
            )
        )
        result = h(data)
        assert isinstance(result, HistogramND)

    def test_with_null_dataframe(self, dataframe_with_null):
        with pytest.raises(ValueError, match="Cannot create histogram .* with nulls"):
            h(dataframe_with_null)


class TestExtraNDArray:
    @given(
        data=dataframes(allowed_dtypes=NUMERIC_POLARS_DTYPES, allow_null=False),
        dropna=st.booleans(),
    )
    def test_same_result_as_with_arrays(self, data: polars.DataFrame, dropna: bool):
        # With equality, we do not have to look at specific cases
        _, result, _ = extract_nd_array(data, dropna=dropna)
        array = data.to_numpy()
        _, array_result, _ = extract_nd_array(array, dropna=dropna)
        assert_array_equal(result, array_result)

    @pytest.mark.parametrize("dropna", [False, True])
    def test_with_empty_data_frame(self, dropna: bool):
        empty_df = polars.DataFrame()
        with pytest.raises(ValueError, match="Must have at least one column"):
            extract_nd_array(empty_df, dropna=dropna)

    @given(data=dataframes(excluded_dtypes=NUMERIC_POLARS_DTYPES), dropna=st.booleans())
    def test_fails_with_wrong_types(self, data: polars.DataFrame, dropna: bool):
        with pytest.raises(ValueError, match=""):
            extract_nd_array(data, dropna=dropna)

    @given(data=series())
    def test_fails_with_series(self, data):
        # series = polars.Series(values=[1, 2, 3, 4, 5])
        with pytest.raises(
            ValueError,
            match="Cannot extract multidimensional array suitable for histogramming from a polars series",
        ):
            extract_nd_array(data)


class TestExtract1DArray:
    @given(
        values=st.lists(
            st.integers(min_value=-1_000_000_000, max_value=1_000_000_000)
            | st.floats(),
        ),
        dropna=st.booleans(),
    )
    def test_with_series(self, dropna: bool, values):
        if not values:
            # Null type was a bit of problem
            pl_input = polars.Series([], dtype=polars.Float64)
        else:
            pl_input = polars.Series(values=values, dtype=polars.Float64, strict=False)
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
        with pytest.raises(
            ValueError, match="Cannot extract axis name from a polars DataFrame."
        ):
            extract_axis_name(data)

    @given(data=series(), explicit_name=st.text())
    def test_uses_explicit_value(self, data, explicit_name):
        assert explicit_name == extract_axis_name(data, axis_name=explicit_name)


@st.composite
def dataframes_and_axis_names(
    draw, *, min_length: int = 1, max_length: int = 6, equal_length: bool = True
):
    df_length = draw(st.integers(min_value=min_length, max_value=max_length))
    if equal_length:
        names_length = df_length
    else:
        if min_length == max_length:
            raise ValueError("Cannot create examples.")
        while (
            names_length := draw(st.integers(min_value=0, max_value=max_length))
        ) == df_length:
            pass
    return (
        draw(dataframes(cols=df_length)),
        draw(st.lists(st.text(), min_size=names_length, max_size=names_length)),
    )


class TestExtractAxisNames:
    @given(data=series())
    def test_fails_with_series(self, data):
        with pytest.raises(
            ValueError, match="Cannot extract axis names from a single polars Series"
        ):
            extract_axis_names(data)

    @given(data=dataframes())
    def test_uses_polars_names(self, data: polars.DataFrame):
        assert tuple(data.columns) == extract_axis_names(data)

    @given(data_and_names=dataframes_and_axis_names(equal_length=True))
    def test_explicit_with_correct_length(self, data_and_names):
        data, names = data_and_names
        result = extract_axis_names(data, axis_names=names)
        assert result == tuple(names)

    @given(data_and_names=dataframes_and_axis_names(equal_length=False))
    def test_explicit_with_incorrect_length(self, data_and_names):
        data, names = data_and_names
        with pytest.raises(ValueError, match="Explicit axis_names.*invalid length"):
            extract_axis_names(data, axis_names=names)


@st.composite
def series_and_mask(
    draw,
    *,
    min_length: int = 0,
    max_length: int = 10,
    allowed_dtypes=NUMERIC_POLARS_DTYPES,
) -> Tuple[polars.Series, np.ndarray]:
    length = draw(st.integers(min_value=min_length, max_value=max_length))
    mask = draw(arrays(shape=(length,), dtype=bool))
    s = draw(series(size=length, allowed_dtypes=allowed_dtypes, allow_null=False))
    return s, mask


class TestExtractWeights:
    @given(data=series(allowed_dtypes=NUMERIC_POLARS_DTYPES, allow_null=False))
    def test_selects_full_series_without_mask(self, data: polars.Series):
        result = extract_weights(data, array_mask=None)
        assert_array_equal(result, data.to_numpy())

    @given(data_and_mask=series_and_mask())
    def test_selects_with_mask(self, data_and_mask):
        data, array_mask = data_and_mask
        result = extract_weights(data, array_mask=array_mask)
        assert isinstance(result, np.ndarray)

    @given(
        data=series(allowed_dtypes=NUMERIC_POLARS_DTYPES, allow_null=False),
        array_mask=arrays(dtype=bool, shape=array_shapes(min_dims=1, max_dims=1)),
    )
    def test_fails_with_shape_mismatch(
        self, data: polars.Series, array_mask: np.ndarray
    ):
        assume(array_mask.shape != data.shape)
        with pytest.raises(ValueError, match="Weights array shape"):
            extract_weights(data, array_mask=array_mask)

    @given(data=series(excluded_dtypes=NUMERIC_POLARS_DTYPES, allow_null=False))
    def test_fails_with_wrong_type(self, data):
        with pytest.raises(ValueError, match="Cannot extract float array from type"):
            extract_weights(data)

    @given(
        data=dataframes(),
        array_mask=st.none() | arrays(dtype=bool, shape=array_shapes(max_dims=1)),
    )
    def test_fails_with_dataframe(self, data: polars.DataFrame, array_mask):
        with pytest.raises(
            ValueError, match="Cannot extract weights from a polars DataFrame"
        ):
            extract_weights(data, array_mask=array_mask)


class TestPhystSeriesAccessors:
    def test_exists_compatible_dtype(self, series_of_int) -> None:
        assert hasattr(series_of_int, "physt")
        assert hasattr(series_of_int.physt, "h1")
        assert not hasattr(series_of_int.physt, "h2")


class TestPhystDataFrameAccessors:
    def test_exists(self, simple_data_frame) -> None:
        assert hasattr(simple_data_frame, "physt")
        assert hasattr(simple_data_frame.physt, "h")
