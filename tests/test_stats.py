import numpy as np
import pytest

from physt._facade import h1
from physt.histogram1d import Statistics
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
        self,
        input_values: ArrayLike,
        input_weights: ArrayLike,
        range_,
        use_weights: bool,
    ) -> Histogram1D:
        if use_weights:
            histogram = h1(input_values, weights=input_weights, range=range_)
        else:
            histogram = h1(input_values, range=range_)
        return histogram

    def test_min(self, histogram):
        assert histogram.statistics.min == 1

    def test_max(self, histogram):
        assert histogram.statistics.max == 4

    def test_sum(self, histogram, use_weights):
        if use_weights:
            assert histogram.statistics.sum == 14
        else:
            assert histogram.statistics.sum == 10

    def test_sum2(self, histogram, use_weights):
        if use_weights:
            assert histogram.statistics.sum2 == 46
        else:
            assert histogram.statistics.sum2 == 30

    def test_weight(self, histogram, use_weights):
        if use_weights:
            assert histogram.statistics.weight == 5
        else:
            assert histogram.statistics.weight == 4

    def test_median(self, histogram, use_weights):
        if use_weights:
            assert np.isnan(histogram.statistics.median)
        else:
            assert histogram.statistics.median == 2.5

    def test_mean(self, histogram, use_weights):
        if use_weights:
            assert histogram.statistics.mean() == 2.8
        else:
            assert histogram.statistics.mean() == 2.5

    def test_std(self, histogram, use_weights):
        if use_weights:
            assert np.allclose(histogram.statistics.std(), np.sqrt(6.8 / 5))
        else:
            assert np.allclose(histogram.statistics.std(), np.sqrt(5 / 4))


class TestEmptyHistogram:
    def test_zero_statistics(self, empty_h1):
        assert empty_h1.statistics == Statistics()
