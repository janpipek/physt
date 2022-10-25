import hypothesis.strategies as st
import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis.extra.numpy import array_shapes, arrays, floating_dtypes, integer_dtypes
from hypothesis.extra.pandas import series

from physt._construction import extract_1d_array, extract_axis_name


class TestExtract1DArray:
    @pytest.mark.parametrize("dropna", [False, True])
    def test_none(self, dropna):
        result = extract_1d_array(None, dropna=dropna)
        assert result == (None, None)

    class TestArrays:
        @given(data=arrays(dtype=floating_dtypes() | integer_dtypes(), shape=array_shapes()))
        def test_removes_nans_if_requested(self, data):
            array, array_mask = extract_1d_array(data, dropna=True)
            assert not any(np.isnan(array))

        @given(
            data=arrays(dtype=floating_dtypes() | integer_dtypes(), shape=array_shapes()),
            dropna=st.booleans(),
        )
        def test_output_is_always_1d(self, data, dropna):
            array, array_mask = extract_1d_array(data, dropna=True)
            assert array.size <= data.size
            assert array.ndim == 1

    class TestPandasSeries:
        @given(data=series(dtype=float), dropna=st.booleans())
        def test_uses_values_of_the_series(self, data, dropna):
            result = extract_1d_array(data, dropna=dropna)
            # TODO: Finish

        @pytest.mark.skip(reason="Not supported by hypothesis yet.")
        @given(data=series(dtype="Int64"))
        def test_extracts_values(self, data):
            pass

        @pytest.mark.parametrize("dtype", ["string", "object", "datetime64[ns]"])
        @pytest.mark.parametrize("dropna", [False, True])
        def test_wrong_dtype(self, dtype, dropna):
            # TODO: Add more types
            data = pd.Series([], dtype=dtype)
            extract_1d_array(data, dropna=dropna)

    class TestIterables:
        @given(data=st.iterables(st.floats() | st.integers()))
        def test_extracts_arrays(self, data):
            array, array_mask = extract_1d_array(data)
            assert isinstance(array, np.ndarray)

    # TODO: Test lists
    # TODO:


class TestExtractNDArray:
    pass


class TestExtractAndConcatArrays:
    pass


class TestExtractAxisName:
    @given(
        data=arrays(dtype=floating_dtypes() | integer_dtypes(), shape=array_shapes())
        | st.iterables(st.floats() | st.integers())
    )
    def test_no_name_for_arrays_and_lists(self, data):
        assert extract_axis_name(data) is None

    @given(
        data=series(dtype=float),
    )
    def test_uses_pandas_series_name(self, data):
        assert data.name == extract_axis_name(data)


class TestExtractAxisNames:
    pass


class TestExtractWeigths:
    pass
