import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
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
        h2 = physt.histogram2d(x, y, [8, 4], name="Some histogram", axis_names=["x", "y"])
        assert h2.frequencies.sum() == 1000
        assert h2.shape == (8, 4)
        assert h2.name == "Some histogram"
        assert h2.axis_names == ["x", "y"]

    def test_dropna(self):
        vals2 = np.array(vals)
        vals2[0, 1] = np.nan
        with pytest.raises(RuntimeError):
            hist = physt.histogram2d(vals2[:,0], vals2[:,1])
        hist = physt.histogram2d(vals2[:, 0], vals2[:, 1], dropna=True)
        assert hist.frequencies.sum() == 6


