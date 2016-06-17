import sys
import os
sys.path = [os.path.join(os.path.dirname(__file__), "..")] + sys.path
import physt
import numpy as np
import pytest


class TestHistogramND(object):
    def test_creation(self):
        data1 = np.random.rand(100)
        data2 = np.random.rand(100)
        data3 = np.random.rand(100)
        data = np.array([data1, data2, data3]).T
        h = physt.histogramdd(data)
        assert h.ndim == 3

    def test_copy(self):
        data1 = np.random.rand(100)
        data2 = np.random.rand(100)
        data3 = np.random.rand(100)
        data = np.array([data1, data2, data3]).T
        h = physt.histogramdd(data)
        print(h._binnings)
        h2 = h.copy()
        assert h == h2
        assert np.array_equal(h.bins, h2.bins)


class TestProjections(object):
    def test_4_to_3(self):
        assert False

    def test_4_to_2(self):
        assert False

    def test_3_to_2(self):
        assert False

    def test_2_to_1(self):
        assert False


if __name__ == "__main__":
    pytest.main(__file__)
