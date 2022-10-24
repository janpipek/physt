from physt.testing.strategies import histograms_1d, histograms_nd
from physt.types import Histogram1D, HistogramND


class TestHistograms1D:
    def test_it_works(self):
        assert isinstance(histograms_1d().example(), Histogram1D)


class TestHistogramsND:
    def test_it_works(self):
        assert isinstance(histograms_nd().example(), HistogramND)
