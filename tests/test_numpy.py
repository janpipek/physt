import pytest

import numpy as np

import physt
from physt.compat.numpy import BIN_COUNT_ALGORITHMS
from physt.compat.numpy import histogram as _histogram
from physt.compat.numpy import histogram2d as _histogram2d
from physt.compat.numpy import histogramdd as _histogramnd

X = np.random.normal(0, 1, 100)
Y = np.random.normal(0, 1, 100)

class TestHistogramEqualness:
    def _test_with_args(self, array, *args, **kwargs):
        values, edges = np.histogram(array, *args, **kwargs)
        histogram = _histogram(array, *args, **kwargs)

        assert np.array_equal(values, histogram.values)
        assert np.array_equal(edges, histogram.schema.edges)
        assert np.array_equal(None, histogram.schema.mask)

    def test_no_args(self):
        self._test_with_args(X)

    def test_with_range(self):
        array = np.arange(0, 10, 100)
        self._test_with_args(array, range=(2, 3))

    def test_with_bin_number(self):
        self._test_with_args(X, 47)

    def test_with_bin_strings(self):
        for algo in BIN_COUNT_ALGORITHMS:
            self._test_with_args(X, algo)

    def test_with_fixed_bins(self):
        edges = [-0.1, 0, 0.2, 0.4, 0.7]
        self._test_with_args(X, edges)

    def test_with_weights(self):
        weights = np.random.lognormal(0, 1, 100)
        self._test_with_args(X, weights=weights)


class TestHistogram2dEqualness:
    def _test_with_args(self, x, y, *args, **kwargs):
        values, edges_x, edges_y = np.histogram2d(x, y, *args, **kwargs)
        histogram = _histogram2d(x, y, *args, **kwargs)

        assert np.array_equal(values, histogram.values)
        assert np.array_equal(edges_x, histogram.schema[0].edges)
        assert np.array_equal(edges_y, histogram.schema[1].edges)
        assert np.array_equal(None, histogram.schema[0].mask)
        assert np.array_equal(None, histogram.schema[1].mask)

    def test_no_args(self):
        self._test_with_args(X, Y)


class TestHistogram3dEqualness:
    pass


if __name__ == "__main__":
    pytest.main(__file__)
