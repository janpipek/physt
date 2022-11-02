import random
from itertools import tee
from typing import Tuple

import hypothesis.strategies as st
import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given
from hypothesis.extra.numpy import array_shapes, arrays, floating_dtypes, integer_dtypes
from hypothesis.extra.pandas import data_frames, series

from physt._construction import (
    extract_1d_array,
    extract_and_concat_arrays,
    extract_axis_name,
    extract_nd_array,
)


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
        def test_keeps_non_nan(self, data, dropna):
            array, array_mask = extract_1d_array(data, dropna=dropna)
            # TODO: Finish

        @given(
            data=arrays(dtype=floating_dtypes() | integer_dtypes(), shape=array_shapes()),
            dropna=st.booleans(),
        )
        def test_output_is_always_1d(self, data, dropna):
            array, array_mask = extract_1d_array(data, dropna=dropna)
            assert array.size <= data.size
            assert array.ndim == 1

    class TestPandas:
        # Note: this is implemented in physt.compat.pandas

        @given(data=series(dtype=float), dropna=st.booleans())
        def test_uses_values_of_the_series(self, data, dropna):
            result = extract_1d_array(data, dropna=dropna)
            # TODO: Finish

        @pytest.mark.skip(reason="Not supported by hypothesis yet.")
        @given(data=series(dtype="Int64"))
        def test_extracts_values_from_series(self, data):
            # TODO: Finish
            pass

        @pytest.mark.parametrize("dtype", ["string", "object", "datetime64[ns]"])
        @pytest.mark.parametrize("dropna", [False, True])
        def test_series_with_wrong_dtype(self, dtype, dropna):
            # TODO: Add more types
            data = pd.Series([], dtype=dtype)
            with pytest.raises(
                ValueError, match="Cannot extract suitable array from non-numeric dtype"
            ):
                extract_1d_array(data, dropna=dropna)

        @pytest.mark.parametrize(
            "data",
            [
                pd.DataFrame(),
                pd.DataFrame({"x": [1, 2, 3]}),
                pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}),
            ],
        )
        def test_dataframes(self, data):
            with pytest.raises(ValueError):
                extract_1d_array(data)

    class TestXarray:
        # TODO: pip install hypothesis-gufunc[xarray] ?
        pass

    class TestIterables:
        @given(data=st.iterables(st.floats() | st.integers()), dropna=st.booleans())
        def test_extracts_arrays(self, data, dropna):
            iter1, iter2 = tee(data)
            data_list = list(iter2)

            array, array_mask = extract_1d_array(iter2, dropna=dropna)
            assert isinstance(array, np.ndarray)
            if dropna:
                assert array.size == len(data_list)
            else:
                assert array.size <= len(data_list)

    @pytest.mark.parametrize("data", ["a_string", 42])
    def test_invalid_scalar_objects(self, data):
        with pytest.raises(ValueError, match="Cannot extract array data from scalar"):
            extract_1d_array(data)

    @pytest.mark.parametrize(
        "data",
        [
            {"a_string": [17]},
        ],
    )
    def test_invalid_container_objects(self, data):
        with pytest.raises(ValueError, match="Cannot extract array data from"):
            extract_1d_array(data)


class TestExtractNDArray:
    @pytest.mark.parametrize("dropna", [False, True])
    @given(
        data=arrays(
            dtype=integer_dtypes() | floating_dtypes(), shape=array_shapes(min_dims=1, max_dims=1)
        )
    )
    def test_low_dimension(self, data, dropna):
        with pytest.raises(ValueError, match=r"Data must have a 2D shape"):
            extract_nd_array(data, dropna=dropna)

    @pytest.mark.parametrize("dropna", [False, True])
    @given(data=arrays(dtype=integer_dtypes() | floating_dtypes(), shape=array_shapes(min_dims=3)))
    def test_high_dimension(self, data, dropna):
        with pytest.raises(ValueError, match=r"Data must have a 2D shape"):
            extract_nd_array(data, dropna=dropna)

    @pytest.mark.parametrize("dropna", [False, True])
    @given(
        dim1=st.integers(min_value=2, max_value=10),
        dim2=st.integers(min_value=2, max_value=10),
    )
    def test_jagged_lists(self, dim1, dim2, dropna):
        assume(dim1 != dim2)
        data = [[random.random() for i in range(dim1)], [random.random() for i in range(dim2)]]
        with pytest.raises(ValueError, match="Data must have a regular 2D shape"):
            extract_nd_array(data, dropna=dropna)

    def test_list_of_lists(self):
        pass


class TestExtractAndConcatArrays:
    @staticmethod
    @st.composite
    def two_array_of_the_same_shape(draw, shape, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        shape_ = draw(shape)
        return (draw(arrays(shape=shape_, **kwargs)), draw(arrays(shape=shape_, **kwargs)))

    @given(
        data1=arrays(
            dtype=floating_dtypes() | integer_dtypes(), shape=array_shapes(min_dims=1, max_dims=1)
        ),
        data2=arrays(
            dtype=floating_dtypes() | integer_dtypes(), shape=array_shapes(min_dims=1, max_dims=1)
        ),
    )
    def test_fails_with_non_matching_size(self, data1: np.ndarray, data2: np.ndarray):
        assume(data1.shape != data2.shape)
        with pytest.raises(ValueError, match="Array shapes do not match"):
            extract_and_concat_arrays(data1, data2)

    @given(
        data=two_array_of_the_same_shape(
            shape=array_shapes(min_dims=2), dtype=floating_dtypes() | integer_dtypes()
        )
    )
    def test_works_with_nd_array(self, data):
        array, mask = extract_and_concat_arrays(data[0], data[1], dropna=False)
        assert array.shape == (data[0].size, 2)

    # TODO: Test correct values
    # TODO: Test dropna


class TestExtractAxisName:
    @given(
        data=arrays(dtype=floating_dtypes() | integer_dtypes(), shape=array_shapes())
        | st.iterables(st.floats() | st.integers())
    )
    def test_no_name_for_arrays_and_lists(self, data):
        assert extract_axis_name(data) is None

    @given(
        data=series(dtype=float, name=st.text() | st.none()),
    )
    def test_uses_pandas_series_name(self, data):
        assert data.name == extract_axis_name(data)


class TestExtractAxisNames:
    pass


class TestExtractWeights:
    pass
