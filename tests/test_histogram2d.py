from __future__ import division
import sys
import os
sys.path = [os.path.join(os.path.dirname(__file__), "..")] + sys.path
import physt
from physt import histogram_nd, h2, binnings
from physt.histogram_nd import Histogram2D
import numpy as np
import pytest


vals = [
    [0.1, 2.0],
    [-0.1, 0.7],
    [0.2, 1.5],
    [0.2, -1.5],
    [0.2, 1.47],
    [1.2, 1.23],
    [0.7, 0.5]
]

np.random.seed(42)


class TestCalculateFrequencies(object):
    def test_simple(self):
        bins = [[0, 1, 2], [0, 1, 2]]
        schemas = [binnings.static_binning(None, np.asarray(bs)) for bs in bins]
        frequencies, errors2, missing = histogram_nd.calculate_frequencies(vals, ndim=2, binnings=schemas)
        assert np.array_equal([[1, 3], [0, 1]], frequencies)
        assert missing == 2
        assert np.array_equal(errors2, frequencies)

    def test_gap(self):
        bins = [
            [[-1, 0], [1, 2]],
            [[-2, -1], [1, 2]]
        ]
        schemas = [binnings.static_binning(None, np.asarray(bs)) for bs in bins]
        frequencies, errors2, missing = histogram_nd.calculate_frequencies(vals, ndim=2, binnings=schemas)
        assert np.array_equal([[0, 0], [0, 1]], frequencies)
        assert missing == 6
        assert np.array_equal(errors2, frequencies)

    def test_errors(self):
        bins = [
            [[-1, 0], [1, 2]],
            [[-2, -1], [1, 2]]
        ]
        weights = [2, 1, 1, 1, 1, 2, 1]
        schemas = [binnings.static_binning(None, np.asarray(bs)) for bs in bins]
        frequencies, errors2, missing = histogram_nd.calculate_frequencies(vals, ndim=2, binnings=schemas, weights=weights)
        # frequencies, errors2, missing = histogram_nd.calculate_frequencies(vals, ndim=2, bins=bins, weights=weights)
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
        assert h2.axis_names == ("x", "y")

    def test_dropna(self):
        vals2 = np.array(vals)
        vals2[0, 1] = np.nan
        with pytest.raises(RuntimeError):
            hist = physt.h2(vals2[:,0], vals2[:,1], dropna=False)
        hist = physt.h2(vals2[:, 0], vals2[:, 1])
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

        i = h * 0.5
        assert np.array_equal(i.frequencies, freqs * 0.5)
        assert np.array_equal(i.errors2, freqs * 0.25)

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

    def test_addition_with_adaptive(self):
        ha = h2([1], [11], "fixed_width", 10, adaptive=True)
        hb = h2([10], [5], "fixed_width", 10, adaptive=True)
        hha = ha + hb
        assert hha == hb + ha
        assert hha.shape == (2, 2)
        assert hha.total == 2
        assert np.array_equal(hha.frequencies, [[0, 1], [1, 0]])

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


class TestDtype(object):
    def test_simple(self):
        from physt import examples
        assert examples.normal_h2().dtype == np.dtype(np.int64)


class TestMerging(object):
    def test_2(self):
        data1 = np.random.rand(100)
        data2 = np.random.rand(100)
        hh = h2(data1, data2, 120)
        hha = h2(data1, data2, 60)
        hhb = hh.merge_bins(2, inplace=False)
        assert hha == hhb


class TestPartialNormalizing(object):
    def test_wrong_arguments(self):
        freqs = [
            [1, 0],
            [1, 2]
        ]
        h = Histogram2D(binnings=(range(3), range(3)), frequencies=freqs)
        with pytest.raises(RuntimeError):
            h0 = h.partial_normalize(2)
        with pytest.raises(RuntimeError):
            h0 = h.partial_normalize(-2)

    def test_axis_names(self):
        freqs = [
            [1, 0],
            [1, 2]
        ]
        h = Histogram2D(binnings=(range(3), range(3)), frequencies=freqs, axis_names=["first_axis", "second_axis"])
        h1 = h.partial_normalize("second_axis")
        assert np.allclose(h1.frequencies, [[1, 0], [.333333333333, .6666666666]])
        with pytest.raises(RuntimeError):
            h0 = h.partial_normalize("third_axis")

    def test_inplace(self):
        freqs = [
            [1, 0],
            [1, 2]
        ]
        h = Histogram2D(binnings=(range(3), range(3)), frequencies=freqs)
        h1 = h.partial_normalize(0, inplace=False)
        assert np.allclose(h.frequencies, freqs)
        assert not np.allclose(h1.frequencies, h.frequencies)
        h.partial_normalize(0, inplace=True)
        assert h1 == h

    def test_values(self):
        freqs = [
            [1, 0],
            [1, 2]
        ]
        h = Histogram2D(binnings=(range(3), range(3)), frequencies=freqs)
        h0 = h.partial_normalize(0)
        h1 = h.partial_normalize(1)

        assert np.allclose(h0.frequencies, [[.5, 0], [.5, 1.0]])
        assert np.allclose(h1.frequencies, [[1, 0], [.333333333333, .6666666666]])

    def test_with_zeros(self):
        freqs = [
            [0, 0],
            [0, 2]
        ]
        h = Histogram2D(binnings=(range(3), range(3)), frequencies=freqs)
        h1 = h.partial_normalize(1)
        assert np.allclose(h1.frequencies, [[0, 0], [0, 1.0]])



if __name__ == "__main__":
    pytest.main(__file__)
