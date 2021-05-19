from abc import ABC
from typing import Any, Dict

import pytest

pytest.importorskip("plotly")
from plotly.graph_objs import Figure

from physt.plotting import plotly

from .shared import AbstractTest, AbstractTest1D, AbstractTest2D


@pytest.fixture()
def default_kwargs() -> Dict[str, Any]:
    return {}


class _TestBase(AbstractTest, ABC):
    module = plotly

    def assert_valid_output(self, output):
        assert isinstance(output, Figure)


class _TestBase1D(_TestBase, AbstractTest1D, ABC):
    pass


class TestBar(_TestBase1D):
    function_name = "bar"


class TestLine(_TestBase1D):
    function_name = "line"


class TestScatter(_TestBase1D):
    function_name = "scatter"


class _TestBase2D(_TestBase, AbstractTest2D, ABC):
    pass


class TestMap(_TestBase2D):
    function_name = "map"
