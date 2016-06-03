import sys
import os
sys.path = [os.path.join(os.path.dirname(__file__), "..")] + sys.path
from physt.histogram1d import AdaptiveHistogram1D
import numpy as np
import pytest


class TestAdaptive(object):
    def test_create_empty(self):
        h = AdaptiveHistogram1D(10)
        assert h.bin_count == 0
        assert h.total == 0
        assert h.bin_width == 10
        assert np.isnan(h.mean())
        assert np.isnan(h.std())
        assert h.overflow == 0
        assert h.underflow == 0

    def test_fill_empty(self):
        h = AdaptiveHistogram1D(10)
        h.fill(24)
        assert h.bin_count == 1
        assert np.array_equal(h.bin_left_edges, [20])
        assert np.array_equal(h.bin_right_edges, [30])
        assert np.array_equal(h.frequencies, [1])

    def test_fill_non_empty(self):
        h = AdaptiveHistogram1D(10)
        h.fill(4)
        h.fill(-14)
        assert h.bin_count == 3
        assert h.total == 2
        assert np.array_equal(h.bin_left_edges, [-20, -10, 0])
        assert np.array_equal(h.bin_right_edges, [-10, 0, 10])
        assert np.array_equal(h.frequencies, [1, 0, 1])       

        h.fill(-14)
        assert h.bin_count == 3
        assert h.total == 3

        h.fill(14)
        assert h.bin_count == 4
        assert h.total == 4
        assert np.array_equal(h.frequencies, [2, 0, 1, 1])


if __name__ == "__main__":
    pytest.main(__file__)