from hypothesis import given

from physt.testing.strategies import histograms_1d, histograms_nd
from physt.types import Histogram1D, HistogramND


class TestHistograms1D:
    @given(example=histograms_1d())
    def test_it_works(self, example):
        assert isinstance(example, Histogram1D)


class TestHistogramsND:
    @given(example=histograms_nd())
    def test_it_works(self, example):
        assert isinstance(example, HistogramND)
