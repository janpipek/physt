import pytest
import numpy as np

import physt
from physt import io
from physt.histogram1d import Histogram1D
from physt.histogram_collection import HistogramCollection


class TestIO:
    def test_json_write_string(self):
        bins = [1.2, 1.4, 1.5, 1.7, 1.8 ]
        values = [4, 0, 3, 7.2]
        example = Histogram1D(bins, values, overflow=1, underflow=2)
        output = io.save_json(example)
        #print(output)
        #assert False

    def test_json_write_2d(self):
        from physt import h2
        values = np.random.rand(500, 2)
        h = h2(values[:,0], values[:,1], 3)
        #print(h.to_json())
        #assert False

    def test_io_equality_on_examples(self):
        from physt.examples import munros
        # for example in ALL_EXAMPLES:
        h = munros()
        json = h.to_json()
        read = io.parse_json(json)
        assert h == read

    def test_simple(self):
        h = physt.h2(None, None, "integer", adaptive=True)
        h << (0, 1)
        json = h.to_json()
        read = io.parse_json(json)
        assert h == read


class TestCollectionIO:
    def test_json_write_collection(self):
        bins = [1.2, 1.4, 1.5, 1.7, 1.8 ]
        values1 = [4, 0, 3, 7.2]
        values2 = [14, 10, 13, 17.2]
        col = HistogramCollection(Histogram1D(bins, values1), Histogram1D(bins, values2))
        col.add(Histogram1D(bins, values2))
        json = col.to_json()
        read = io.parse_json(json)
        assert read == col
