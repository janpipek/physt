import numpy as np
import pytest

import physt

values = [1, 2, 3, 4]
weights = [1, 1, 1, 2]


class TestStatistics:
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
