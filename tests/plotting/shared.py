from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from physt.histogram_nd import Histogram2D
    from physt.histogram1d import Histogram1D

import pytest


class AbstractTest1D(ABC):
    module: Any
    function_name: str

    def method(self, h1, *args, **kwargs) -> Any:
        f = getattr(self.module, self.function_name)
        return f(h1, *args, *kwargs)

    def test_2d_fail(self, simple_h2: "Histogram2D"):
        with pytest.raises(TypeError):
            self.method(simple_h2)

    def test_simple_does_not_fail(self, simple_h1):
        _ = self.method(simple_h1)


class AbstractTest2D(ABC):
    module: Any
    function_name: str

    def method(self, h2, *args, **kwargs) -> Any:
        f = getattr(self.module, self.function_name)
        return f(h2, *args, *kwargs)

    def test_1d_fail(self, simple_h1: "Histogram1D"):
        with pytest.raises(TypeError):
            self.method(simple_h1)