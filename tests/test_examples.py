import sys

import pytest
import numpy as np

from physt import examples
from physt.histogram1d import Histogram1D
from physt.histogram_nd import HistogramND, Histogram2D

try:
    import pandas as pd
except ImportError:
    pass

try:
    import seaborn as sns
except ImportError:
    pass


class TestExamples:
    def test_normal(self):
        assert isinstance(examples.normal_h1(), Histogram1D)
        assert isinstance(examples.normal_h2(), Histogram2D)
        assert isinstance(examples.normal_h3(), HistogramND)

    @pytest.mark.skipif('pandas' not in sys.modules, reason="requires the pandas library")
    def test_munros(self):
        h1 = examples.munros()
        assert h1.total == 282
        assert h1.name == "munros"
        assert h1.axis_names == ("lat", "long")
        assert np.allclose(56.166666667, h1.get_bin_left_edges(0)[0])
        assert np.allclose(58.333333333, h1.get_bin_left_edges(0)[-1])
