import sys
import os
sys.path = [os.path.join(os.path.dirname(__file__), "..")] + sys.path
# from physt.histogram1d import Histogram1D
from physt import histogram
import numpy as np
import pytest


class TestNumpyBins(object):
    def test_nbin(self):
        arr = np.random.rand(100)
        hist = histogram(arr, bins=15)
        assert hist.bin_count == 15
        assert np.isclose(hist.bin_right_edges[-1], arr.max())
        assert np.isclose(hist.bin_left_edges[0], arr.min())

    def test_edges(self):
        arr = np.arange(0, 1, 0.01)
        hist = histogram(arr, np.arange(0.1, 0.8001, 0.1))
        assert np.allclose(hist.numpy_bins, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        assert hist.underflow == 10
        assert hist.overflow == 19

    def test_range(self):
        arr = np.arange(0, 1.00, 0.01)
        hist = histogram(arr, 10, range=(0.5, 1.0))
        assert hist.bin_count == 10
        assert hist.bin_left_edges[0] == 0.5
        assert hist.bin_right_edges[-1] == 1.0
        assert hist.overflow == 0
        assert hist.underflow == 50
        assert hist.total == 50

        hist = histogram(arr, bins=10, range=(0.5, 1.0), keep_missed=False)
        assert hist.total == 50
        assert np.isnan(hist.underflow)
        assert np.isnan(hist.overflow)


if __name__ == "__main__":
    pytest.main(__file__)
