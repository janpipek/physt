import sys
import os

import pytest
import numpy as np
try:
    import pandas as pd
except ImportError:
    pass

try:
    import seaborn as sns
except ImportError:
    pass


sys.path = [os.path.join(os.path.dirname(__file__), "..")] + sys.path

from physt import examples
from physt.histogram1d import Histogram1D
from physt.histogram_nd import HistogramND, Histogram2D


class TestExamples:
    def test_normal(self):
        assert isinstance(examples.normal_h1(), Histogram1D)
        assert isinstance(examples.normal_h2(), Histogram2D)
        assert isinstance(examples.normal_h3(), HistogramND)

    @pytest.mark.skipif('seaborn' not in sys.modules, reason="requires the seaborn library")
    def test_iris(self):
        iris = examples.iris_h1()
        assert iris.name == "iris"
        assert iris.axis_name == "sepal_length"
        assert iris.total == 150
        assert iris.min_edge == 4.2
        assert iris.max_edge == 8.0

        iris2 = examples.iris_h1("sepal_width")
        assert iris2.name == "iris"
        assert iris2.axis_name == "sepal_width"
        assert iris2.total == 150

        iris3 = examples.iris_h2()
        assert iris3.name == "iris"
        assert iris3.axis_names == ("sepal_length", "sepal_width")
        assert iris3.total == 150

        iris4 = examples.iris_h2("petal_width", "sepal_length")
        assert iris4.axis_names == ("petal_width", "sepal_length")

    @pytest.mark.skipif('pandas' not in sys.modules, reason="requires the pandas library")
    def test_munros(self):
        h1 = examples.munros()
        assert h1.total == 282
        assert h1.name == "munros"
        assert h1.axis_names == ("lat", "long")
        assert np.allclose(56.166666667, h1.get_bin_left_edges(0)[0])
        assert np.allclose(58.333333333, h1.get_bin_left_edges(0)[-1])


if __name__ == "__main__":
    pytest.main(__file__)
