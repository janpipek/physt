import numpy as np
import pytest

import physt
from physt.facade import h1
from physt.types import Histogram1D
from physt.typing_aliases import ArrayLike


@pytest.fixture
def input_values() -> ArrayLike:
    return [1, 2, 3, 4]


@pytest.fixture
def input_weights() -> ArrayLike:
    return [1, 1, 1, 2]


@pytest.fixture
def simple_h1_without_stats(simple_h1: Histogram1D) -> Histogram1D:
    h1_copy = simple_h1.copy()
    h1_copy._stats = None
    return h1_copy


@pytest.mark.parametrize("range_", [None, pytest.param((1.5, 2.5), id="limiting")])
@pytest.mark.parametrize("use_weights", [False, True])
class TestStatisticsComputation:
    @pytest.fixture
    def histogram(
        self, input_values: ArrayLike, input_weights: ArrayLike, range_, use_weights: bool
    ) -> Histogram1D:
        if use_weights:
            histogram = h1(input_values, weights=input_weights, range=range_)
        else:
            histogram = h1(input_values, range=range_)
        return histogram

    def test_min(self, histogram):
        assert histogram._stats["min"] == 1

    def test_max(self, histogram):
        assert histogram._stats["max"] == 4

    def test_sum(self, histogram, use_weights):
        if use_weights:
            assert histogram._stats["sum"] == 14
        else:
            assert histogram._stats["sum"] == 10

    def test_sum2(self, histogram, use_weights):
        if use_weights:
            assert histogram._stats["sum2"] == 46
        else:
            assert histogram._stats["sum2"] == 30

    def test_weight(self, histogram, use_weights):
        if use_weights:
            assert histogram._stats["weight"] == 5
        else:
            assert histogram._stats["weight"] == 4

    def test_mean(self, histogram, use_weights):
        if use_weights:
            assert histogram.mean() == 2.8
        else:
            assert histogram.mean() == 2.5

    def test_std(self, histogram, use_weights):
        if use_weights:
            assert np.allclose(histogram.std(), np.sqrt(6.8 / 5))
        else:
            assert np.allclose(histogram.std(), np.sqrt(5 / 4))


class TestEmptyHistogram:
    def test_zero_statistics(self, empty_h1):
        assert empty_h1._stats == Histogram1D.EMPTY_STATS
