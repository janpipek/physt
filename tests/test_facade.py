from typing import Tuple

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis.extra.numpy import (
    array_shapes,
    arrays,
    floating_dtypes,
    from_dtype,
    integer_dtypes,
)

from physt._facade import h1, h2
from physt.types import Histogram1D, Histogram2D


class TestH1:
    class TestNoArgs:
        @given(
            array=st.one_of(
                arrays(
                    # TODO: Add more floating types?
                    dtype=float,
                    shape=array_shapes(min_side=2),
                    unique=True,
                    elements=st.floats(
                        allow_nan=False, allow_infinity=False, allow_subnormal=False
                    ),
                ),
                arrays(
                    # TODO: Add more integer types?
                    dtype=int,
                    shape=array_shapes(min_side=2),
                    unique=True,
                ),
            )
        )
        def test_array_at_least_two_different_values(self, array):
            # Reasonable defaults for at least two different values
            # Avoid too narrow ranges in float precision
            array_range = array.max() - array.min()
            assume(np.inf > array_range > np.spacing(array.min()) * 20)

            histogram = h1(array)
            assert isinstance(histogram, Histogram1D)
            assert histogram.bin_right_edges[-1] >= array.max()
            assert histogram.bin_left_edges[0] >= array.min()
            assert histogram.total == array.size

        def test_empty_ndarray(self, empty_ndarray):
            with pytest.raises(
                ValueError, match="At least 2 values required to infer bins"
            ):
                h1(empty_ndarray)

        def test_infinitesimal_range(self):
            array = np.array([1, np.nextafter(1, 2)])
            h1(array)
            # TODO: Test that it is actually ok


@st.composite
def two_1d_arrays_of_the_same_length(
    draw, *, min_side=None, max_side=None, dtype=float, **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    array_shape = draw(
        array_shapes(min_side=min_side, max_side=max_side, min_dims=1, max_dims=1)
    )
    array_strategy = arrays(shape=array_shape, dtype=dtype, **kwargs)
    return draw(array_strategy), draw(array_strategy)


@st.composite
def valid_h2_inputs(draw, *, min_side=2, **kwargs):
    dtype = draw(floating_dtypes() | integer_dtypes())
    return draw(
        two_1d_arrays_of_the_same_length(
            dtype=dtype,
            min_side=min_side,
            elements=from_dtype(dtype, allow_infinity=False),
            **kwargs,
        ).filter(
            lambda arrays: (arrays[0].max() > arrays[0].min())
            and (arrays[1].max() > arrays[1].min())
        )
    )


class TestH2:
    class TestNoArgs:
        @given(arrays=valid_h2_inputs())
        def test_array_at_least_two_different_values(self, arrays):
            array1_range = arrays[0].max() - arrays[0].min()
            array2_range = arrays[1].max() - arrays[1].min()

            # Ensure that we can safely create the bins from a very
            # narrow float range
            min_diff1 = np.spacing(arrays[0].min()) * 20
            min_diff2 = np.spacing(arrays[1].min()) * 20

            assume(np.isfinite(array1_range) and array1_range > min_diff1)
            assume(np.isfinite(array2_range) and array2_range > min_diff2)
            histogram = h2(arrays[0], arrays[1])
            assert isinstance(histogram, Histogram2D)

        def test_infinite_range(self):
            # TODO: Move to numpy bins testing
            array = np.array([-1e308, 1e308])
            with pytest.raises(ValueError, match="Range too large to find bins"):
                h2(array, array)

        def test_empty_ndarray(self, empty_ndarray):
            with pytest.raises(
                ValueError, match="At least 2 values required to infer bins"
            ):
                h2(empty_ndarray, empty_ndarray)
