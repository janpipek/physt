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

    def test_transform(self):
        t = special.PolarHistogram.transform([1, 0])
        assert np.array_equal(t, [1, 0])

        t = special.PolarHistogram.transform([0, 2])
        assert np.allclose(t, [2, np.pi / 2])

        data = np.asarray([[1, 0], [0, 2]])
        t = special.PolarHistogram.transform(data)
        assert np.allclose(t, [[1, 0], [2, np.pi / 2]])


class TestSpherical(object):
    def test_transform(self):
        t = special.SphericalHistogram.transform([0, 0, 1])
        assert np.array_equal(t,  [1, 0, 0])

        t = special.SphericalHistogram.transform([2, 2, 0])
        assert np.array_equal(t,  [np.sqrt(8), np.pi / 2, np.pi / 4])

        data = np.asarray([[3, 0, 0], [0, 0, 0], [0, .5, -.5]])
        expected = np.asarray([[3, np.pi / 2, 0], [0, 0, 0], [np.sqrt(.5), .75 * np.pi, np.pi / 2]])
        assert np.allclose(expected, special.SphericalHistogram.transform(data))


class TestCylindrical(object):
    def test_transform(self):
        t = special.CylindricalHistogram.transform([0, 0, 1])
        assert np.array_equal(t,  [0, 0, 1])

        t = special.CylindricalHistogram.transform([2, 2, 2])
        assert np.array_equal(t,  [np.sqrt(8), np.pi / 4, 2])

        data = np.asarray([[3, 0, 0], [0, 0, 0], [0, .5, -.5]])
        expected = np.asarray([[3, 0, 0], [0, 0, 0], [.5, np.pi / 2, -.5]])
        assert np.allclose(expected, special.CylindricalHistogram.transform(data))


    # def test_fill_n(self):


if __name__ == "__main__":
    pytest.main(__file__)
