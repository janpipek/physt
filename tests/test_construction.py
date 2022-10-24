import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given
from hypothesis.extra.numpy import array_shapes, arrays, floating_dtypes, integer_dtypes

from physt._construction import extract_1d_array


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
