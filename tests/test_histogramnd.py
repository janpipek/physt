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


if __name__ == "__main__":
    pytest.main(__file__)
