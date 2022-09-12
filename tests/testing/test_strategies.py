from physt.testing.strategies import histograms_1d
from physt.types import Histogram1D


class TestHistograms1D:
    def test_it_works(self):
        assert isinstance(histograms_1d().example(), Histogram1D)
