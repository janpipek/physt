import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from physt import histogram_nd
import numpy as np
import pytest


vals = [
    [0.1, 1.4],
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
        frequencies, missing, errors2 = histogram_nd.calculate_frequencies(vals, ndim=2, bins=bins)
        assert np.array_equal([[1, 3], [0, 1]], frequencies)
        assert missing == 2
        assert np.array_equal(errors2, frequencies)

    def test_gap(self):
        bins = [
            [[-1, 0], [1, 2]],
            [[-2, -1], [1, 2]]
        ]
        frequencies, missing, errors2 = histogram_nd.calculate_frequencies(vals, ndim=2, bins=bins)
        assert np.array_equal([[0, 0], [0, 1]], frequencies)
        assert missing == 6
        assert np.array_equal(errors2, frequencies)

    def test_errors(self):
        bins = [
            [[-1, 0], [1, 2]],
            [[-2, -1], [1, 2]]
        ]
        weights = [2, 1, 1, 1, 1, 2, 1]
        frequencies, missing, errors2 = histogram_nd.calculate_frequencies(vals, ndim=2, bins=bins, weights=weights)
        assert np.array_equal([[0, 0], [0, 2]], frequencies)
        assert missing == 7
        assert np.array_equal([[0, 0], [0, 4]], errors2)