from abc import ABC
from typing import TYPE_CHECKING

from physt.plotting import vega
from .shared import AbstractTest, AbstractTest1D, AbstractTest2D

if TYPE_CHECKING:
    from physt.histogram1d import Histogram1D
    from physt.histogram_nd import Histogram2D


class _TestBase(AbstractTest, ABC):
    module = vega

    def assert_valid_output(self, output):
        assert isinstance(output, output)


class _TestBase1D(_TestBase, AbstractTest1D, ABC):
    def test_axes_labels(self, simple_h1: "Histogram1D"):
        vega_data = self.method(simple_h1)
        assert vega_data["axes"][0]["title"] == simple_h1.axis_name

    # TODO: Add many more shared tests


class TestLine(_TestBase1D):
    function_name = "line"


class TestScatter(_TestBase1D):
    function_name = "scatter"


class TestBar(_TestBase1D):
    function_name = "bar"


class _TestBase2D(_TestBase, AbstractTest2D, ABC):
    def test_axes_labels(self, simple_h2: "Histogram2D"):
        vega_data = self.method(simple_h2)
        assert vega_data["axes"][0]["title"] == simple_h2.axis_names[0]
        assert vega_data["axes"][1]["title"] == simple_h2.axis_names[1]


class TestMap(_TestBase2D):
    function_name = "map"
