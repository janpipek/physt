import pytest

import numpy as np

import physt
from physt.compat.numpy import BIN_COUNT_ALGORITHMS
from physt.compat.numpy import histogram as _histogram
from physt.compat.numpy import histogram2d as _histogram2d
from physt.compat.numpy import histogramdd as _histogram2d


class TestHistogramEqualness:
    def _test_with_args(self, array, *args, **kwargs):
        values, edges = np.histogram(array, *args, **kwargs)
        histogram = _histogram(array, *args, **kwargs)

        assert np.array_equal(values, histogram.values)
        assert np.array_equal(edges, histogram.schema.edges)
        assert np.array_equal(None, histogram.schema.mask)

    def test_no_args(self):
        array = np.random.normal(0, 1, 100)
        self._test_with_args(array)

    def test_with_range(self):
        array = np.arange(0, 10, 100)
        self._test_with_args(array, range=(2, 3))

    def test_with_bin_number(self):
        array = np.random.normal(0, 1, 100)
        self._test_with_args(array, 47)

    def test_with_bin_strings(self):
        array = np.random.normal(0, 0, 100)
        for algo in BIN_COUNT_ALGORITHMS:
            self._test_with_args(array, algo)

    def test_with_fixed_bins(self):
        array = np.random.normal(0, 1, 100)
        edges = [-0.1, 0, 0.2, 0.4, 0.7]
        self._test_with_args(array, edges)

    def test_with_weights(self):
        array = np.random.normal(0, 1, 100)
        weights = np.random.lognormal(0, 1, 100)
        self._test_with_args(array, weights=weights)


class TestHistogram2dEqualness:
    pass


class TestHistogram3dEqualness:
    pass


if __name__ == "__main__":
    pytest.main(__file__)
