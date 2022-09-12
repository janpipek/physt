from typing import Tuple

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given
from hypothesis.extra.numpy import (
    array_shapes,
    arrays,
    floating_dtypes,
    from_dtype,
    integer_dtypes,
)

from physt.facade import h1, h2
from physt.types import Histogram1D, Histogram2D


class TestH1:
    class TestNoArgs:
        @given(
            arrays(
                dtype=floating_dtypes() | integer_dtypes(), shape=array_shapes(), unique=True
            ).filter(lambda arr: np.isfinite(arr).all() and arr.size > 2)
        )
        def test_array_at_least_two_different_values(self, arr):
            # Reasonable defaults for at least two different values
            histogram = h1(arr)
            assert isinstance(histogram, Histogram1D)
            assert histogram.bin_right_edges[-1] >= arr.max()
            assert histogram.bin_left_edges[0] >= arr.min()
            assert histogram.total == arr.size

        def test_empty_ndarray(self, empty_ndarray):
            with pytest.raises(ValueError, match="At least 2 values required to infer bins"):
                h1(empty_ndarray)


@st.composite
def two_1d_arrays_of_the_same_length(
    draw, *, min_side=None, max_side=None, dtype=float, **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    array_shape = draw(array_shapes(min_side=min_side, max_side=max_side, min_dims=1, max_dims=1))
    arr_strategy = arrays(shape=array_shape, dtype=dtype, **kwargs)
    return draw(arr_strategy), draw(arr_strategy)


@st.composite
def valid_h2_inputs(draw, *, min_side=2, **kwargs):
    dtype = draw(floating_dtypes() | integer_dtypes())
    return draw(
        two_1d_arrays_of_the_same_length(
            dtype=dtype,
            min_side=min_side,
            elements=from_dtype(dtype, allow_infinity=False),
            **kwargs,
        ).filter(lambda arrs: (arrs[0].max() > arrs[0].min()) and (arrs[1].max() > arrs[1].min()))
    )


class TestH2:
    class TestNoArgs:
        @given(valid_h2_inputs())
        def test_array_at_least_two_different_values(self, arrays):
            histogram = h2(arrays[0], arrays[1])
            assert isinstance(histogram, Histogram2D)

        def test_empty_ndarray(self, empty_ndarray):
            with pytest.raises(ValueError, match="At least 2 values required to infer bins"):
                h2(empty_ndarray, empty_ndarray)
