import numpy as np
import pytest
from hypothesis import given

from physt import examples, io
from physt.testing.strategies import histograms_1d, histograms_nd
from physt.types import Histogram1D, HistogramBase, HistogramCollection


class TestIO:
    def test_json_write_string(self):
        bins = [1.2, 1.4, 1.5, 1.7, 1.8]
        values = [4, 0, 3, 7.2]
        example = Histogram1D(bins, values, overflow=1, underflow=2)
        io.save_json(example)
        # TODO: Check output?

    def test_json_write_2d(self):
        from physt import h2

        values = np.random.rand(500, 2)
        h = h2(values[:, 0], values[:, 1], 3)
        h.to_json()
        # TODO: Check output?

    @staticmethod
    def _assert_reversibility(h: HistogramBase):
        json = h.to_json()
        read = io.parse_json(json)
        assert h == read

    @pytest.mark.skipif("munros" not in dir(examples), reason="Pandas required.")
    def test_io_equality_on_examples(self):
        h = examples.munros()
        self._assert_reversibility(h)

    @given(histograms_1d())
    def test_reversibility_1d(self, h):
        self._assert_reversibility(h)

    @given(histograms_nd())
    def test_reversibility_nd(self, h):
        self._assert_reversibility(h)


class TestCollectionIO:
    def test_json_write_collection(self):
        bins = [1.2, 1.4, 1.5, 1.7, 1.8]
        values1 = [4, 0, 3, 7.2]
        values2 = [14, 10, 13, 17.2]
        col = HistogramCollection(
            Histogram1D(bins, values1), Histogram1D(bins, values2)
        )
        col.add(Histogram1D(bins, values2))
        json = col.to_json()
        read = io.parse_json(json)
        assert read == col
