import sys
import os
sys.path = [os.path.join(os.path.dirname(__file__), "..")] + sys.path
import physt
from physt import histogram_nd
import numpy as np
import pytest


vals = [
    [0.1, 2],
    [-0.1, 0.7],
    [0.2, 1.5],
    [0.2, -1.5],
    [0.2, 1.47],
    [1.2, 1.23],
    [0.7, 0.5]
]


class TestCalculateFrequencies(object):
    def test_simple(self):
        bins = [[0, 1, 2], [0, 1, 2]]
        frequencies, errors2, missing = histogram_nd.calculate_frequencies(vals, ndim=2, bins=bins)
        assert np.array_equal([[1, 3], [0, 1]], frequencies)
        assert missing == 2
        assert np.array_equal(errors2, frequencies)

    def test_gap(self):
        bins = [
            [[-1, 0], [1, 2]],
            [[-2, -1], [1, 2]]
        ]
        frequencies, errors2, missing = histogram_nd.calculate_frequencies(vals, ndim=2, bins=bins)
        assert np.array_equal([[0, 0], [0, 1]], frequencies)
        assert missing == 6
        assert np.array_equal(errors2, frequencies)

    def test_errors(self):
        bins = [
            [[-1, 0], [1, 2]],
            [[-2, -1], [1, 2]]
        ]
        weights = [2, 1, 1, 1, 1, 2, 1]
        frequencies, errors2, missing = histogram_nd.calculate_frequencies(vals, ndim=2, bins=bins, weights=weights)
        assert np.array_equal([[0, 0], [0, 2]], frequencies)
        assert missing == 7
        assert np.array_equal([[0, 0], [0, 4]], errors2)


class TestHistogram2D(object):
    def test_simple_random(self):
        x = np.random.normal(100, 1, 1000)
        y = np.random.normal(10, 10, 1000)
        h2 = physt.h2(x, y, [8, 4], name="Some histogram", axis_names=["x", "y"])
        assert h2.frequencies.sum() == 1000
        assert h2.shape == (8, 4)
        assert h2.name == "Some histogram"
        assert h2.axis_names == ["x", "y"]

    def test_dropna(self):
        vals2 = np.array(vals)
        vals2[0, 1] = np.nan
        with pytest.raises(RuntimeError):
            hist = physt.h2(vals2[:,0], vals2[:,1])
        hist = physt.h2(vals2[:, 0], vals2[:, 1], dropna=True)
        assert hist.frequencies.sum() == 6


# Calculated:
freqs = np.array([[ 1.,  0.],
    [ 1.,  0.],
    [ 1.,  1.],
    [ 1.,  0.],
    [ 1.,  0.]])


class TestArithmetics(object):
    def test_multiply_by_constant(self):
        xx = np.array([0.5, 1.5, 2.5, 2.2, 3.3, 4.2])
        yy = np.array([1.5, 1.5, 1.5, 2.2, 1.3, 1.2])
        h = physt.h2(xx, yy, "fixed_width", 1)

        assert np.array_equal(h.frequencies, freqs)
        i = h * 2
        assert np.array_equal(i.frequencies, freqs * 2)
        assert np.array_equal(i.errors2, freqs * 4)        

    def test_multiply_by_other(self):
        xx = np.array([0.5, 1.5, 2.5, 2.2, 3.3, 4.2])
        yy = np.array([1.5, 1.5, 1.5, 2.2, 1.3, 1.2])        
        h = physt.h2(xx, yy, "fixed_width", 1)
        with pytest.raises(RuntimeError):
            h * h

    def test_divide_by_other(self):
        xx = np.array([0.5, 1.5, 2.5, 2.2, 3.3, 4.2])
        yy = np.array([1.5, 1.5, 1.5, 2.2, 1.3, 1.2])        
        h = physt.h2(xx, yy, "fixed_width", 1)
        with pytest.raises(RuntimeError):
            h * h

    def test_divide_by_constant(self):
        xx = np.array([0.5, 1.5, 2.5, 2.2, 3.3, 4.2])
        yy = np.array([1.5, 1.5, 1.5, 2.2, 1.3, 1.2])        
        h = physt.h2(xx, yy, "fixed_width", 1)
        i = h / 2
        assert np.array_equal(i.frequencies, freqs / 2)
        assert np.array_equal(i.errors2, freqs / 4)                   

    def test_addition_by_constant(self):
        xx = np.array([0.5, 1.5, 2.5, 2.2, 3.3, 4.2])
        yy = np.array([1.5, 1.5, 1.5, 2.2, 1.3, 1.2])        
        h = physt.h2(xx, yy, "fixed_width", 1)
        with pytest.raises(RuntimeError):
            h + 4        

    def test_addition_with_another(self):
        xx = np.array([0.5, 1.5, 2.5, 2.2, 3.3, 4.2])
        yy = np.array([1.5, 1.5, 1.5, 2.2, 1.3, 1.2])        
        h = physt.h2(xx, yy, "fixed_width", 1)
        i = h + h
        assert np.array_equal(i.frequencies, freqs * 2)
        assert np.array_equal(i.errors2, freqs * 2)   

    def test_subtraction_with_another(self):
        xx = np.array([0.5, 1.5, 2.5, 2.2, 3.3, 4.2])
        yy = np.array([1.5, 1.5, 1.5, 2.2, 1.3, 1.2])        
        h = physt.h2(xx, yy, "fixed_width", 1)
        i = h * 2 - h
        assert np.array_equal(i.frequencies, freqs)
        assert np.array_equal(i.errors2, 5 * freqs)          

    def test_subtraction_by_constant(self):
        xx = np.array([0.5, 1.5, 2.5, 2.2, 3.3, 4.2])
        yy = np.array([1.5, 1.5, 1.5, 2.2, 1.3, 1.2])        
        h = physt.h2(xx, yy, "fixed_width", 1)
        with pytest.raises(RuntimeError):
            h - 4               


if __name__ == "__main__":
    pytest.main(__file__)
