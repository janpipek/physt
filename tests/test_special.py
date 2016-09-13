import sys
import os
sys.path = [os.path.join(os.path.dirname(__file__), "..")] + sys.path
import physt
import numpy as np
import pytest
from physt import special


class TestPolar(object):
    def test_simple_create(self):
        data = np.array([[0.01, 0.01], [0.01, 0.99], [-1, .01], [-1, -.01]])
        x = data[:,0]
        y = data[:,1]
        h = special.polar_histogram(x, y, radial_bins=2, phi_bins=4)
        assert h.bin_count == 8
        assert np.array_equal([[1, 0, 0, 0], [1, 1, 1, 0]], h.frequencies)


    # def test_fill_n(self):


if __name__ == "__main__":
    pytest.main(__file__)
