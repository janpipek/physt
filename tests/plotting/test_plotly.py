from abc import ABC
from typing import Any

import pytest

pytest.importorskip("plotly")
from plotly.graph_objs import Figure

from physt.plotting import plotly

from .shared import AbstractPlot1DTest, AbstractPlot2DTest, AbstractPlotTest


@pytest.fixture()
def default_kwargs() -> dict[str, Any]:
    return {}


class _TestBase(AbstractPlotTest, ABC):
    module = plotly

    def assert_valid_output(self, output):
        assert isinstance(output, Figure)


class _TestBase1D(_TestBase, AbstractPlot1DTest, ABC):
    pass


class TestBar(_TestBase1D):
    function_name = "bar"


class TestLine(_TestBase1D):
    function_name = "line"


class TestScatter(_TestBase1D):
    function_name = "scatter"


class _TestBase2D(_TestBase, AbstractPlot2DTest, ABC):
    pass


class TestMap(_TestBase2D):
    function_name = "map"
