from abc import ABC

import pytest

pytest.importorskip("matplotlib")
from matplotlib.axes import Axes

from physt.plotting import matplotlib
from .shared import AbstractTest1D, AbstractTest2D, AbstractTest


class _TestBase(AbstractTest, ABC):
    module = matplotlib

    def assert_valid_output(self, output) -> None:
        assert isinstance(output, Axes)


class _TestBase1D(_TestBase, AbstractTest1D, ABC):
    pass


class TestBar(_TestBase1D):
    function_name = "bar"


class TestLine(_TestBase1D):
    function_name = "line"


class TestFill(_TestBase1D):
    function_name = "fill"


class TestScatter(_TestBase1D):
    function_name = "scatter"


class TestStep(_TestBase1D):
    function_name = "step"


class _TestBase2D(_TestBase, AbstractTest2D, ABC):
    pass


class TestMap(_TestBase2D):
    function_name = "map"


class TestImage(_TestBase2D):
    function_name = "image"


class TestBar3D(_TestBase2D):
    function_name = "bar3d"


# TODO: Test transformed histograms
