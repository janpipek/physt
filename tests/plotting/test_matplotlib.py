from abc import ABC

from physt.plotting import matplotlib
from .shared import AbstractTest1D, AbstractTest2D


class _TestBase1D(AbstractTest1D, ABC):
    module = matplotlib


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


class _TestBase2D(AbstractTest2D, ABC):
    module = matplotlib


class TestMap(_TestBase2D):
    function_name = "map"


class TestImage(_TestBase2D):
    function_name = "image"


class TestBar3D(_TestBase2D):
    function_name = "bar3d"


# TODO: Test transformed histograms
