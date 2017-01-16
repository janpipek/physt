import sys
import os
sys.path = [os.path.join(os.path.dirname(__file__), "..")] + sys.path
from physt import bin_utils, io
from physt.histogram1d import Histogram1D
import numpy as np
import pytest


class TestIO(object):
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
        from physt.example import ALL_EXAMPLES

        for example in ALL_EXAMPLES:
            h = example()
            json = h.to_json()
            read = io.parse_json(json)
            assert h == read


if __name__ == "__main__":
    pytest.main(__file__)
