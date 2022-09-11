import numpy as np
from hypothesis import given
from hypothesis.extra.numpy import array_shapes, arrays, floating_dtypes, integer_dtypes

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
