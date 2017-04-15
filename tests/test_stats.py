from __future__ import division
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pytest
import physt
from physt.histogram1d import calculate_frequencies

values = [1, 2, 3, 4]
weights = [1, 1, 1, 2]


class TestStatistics(object):
    def test_stats_filled_in(self):
        h = physt.h1(values)
        assert h._stats["sum"] == 10
        assert h._stats["sum2"] == 30

    def test_mean_no_weights(self):
        h = physt.h1(values)
        assert h.mean() == 2.5

    def test_std_no_weights(self):
        h = physt.h1(values)
        assert np.allclose(h.std(), np.sqrt(5/4))

    def test_mean_weights(self):
        hw = physt.h1(values, weights=weights)
        assert hw.mean() == 2.8

    def test_std_weights(self):
        hw = physt.h1(values, weights=weights)
        assert np.allclose(hw.std(), np.sqrt(6.8 / 5))


if __name__ == "__main__":
    pytest.main(__file__)
