from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import pytest

from physt.plotting import vega

if TYPE_CHECKING:
    from physt.histogram1d import Histogram1D
    from physt.histogram_nd import Histogram2D


class _TestBase1D(ABC):
    @abstractmethod
    def method(self, h1, *args, **kwargs): ...

    def test_2d_fail(self, simple_h2: "Histogram2D"):
        with pytest.raises(TypeError):
            self.method(simple_h2)

    def test_axes_labels(self, simple_h1: "Histogram1D"):
        vega_data = self.method(simple_h1)
        assert vega_data["axes"][0]["title"] == simple_h1.axis_name

    def test_title(self, simple_h1: "Histogram1D"):
        vega_data = self.method(simple_h1)


    # TODO: Add many more shared tests


class TestLine(_TestBase1D):
    def method(self, h1, *args, **kwargs):
        return vega.line(h1, *args, **kwargs)


class TestScatter(_TestBase1D):
    def method(self, h1, *args, **kwargs):
        return vega.scatter(h1, *args, **kwargs)


class TestBar(_TestBase1D):
    def method(self, h1, *args, **kwargs):
        return vega.bar(h1, *args, **kwargs)


class _TestBase2D(ABC):
    @abstractmethod
    def method(self, h2, *args, **kwargs): ...

    def test_axes_labels(self, simple_h2: "Histogram2D"):
        vega_data = self.method(simple_h2)
        assert vega_data["axes"][0]["title"] == simple_h2.axis_names[0]
        assert vega_data["axes"][1]["title"] == simple_h2.axis_names[1]

    def test_1d_fail(self, simple_h1: "Histogram1D"):
        with pytest.raises(TypeError):
            self.method(simple_h1)


class TestMap(_TestBase2D):
    def method(self, h2, *args, **kwargs):
        return vega.map(h2, *args, **kwargs)