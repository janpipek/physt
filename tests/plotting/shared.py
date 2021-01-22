from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from physt.histogram_nd import Histogram2D
    from physt.histogram1d import Histogram1D


class AbstractTest(ABC):
    module: Any
    function_name: str

    def method(self, h, *args, **kwargs) -> Any:
        f = getattr(self.module, self.function_name)
        return f(h, *args, **kwargs)

    @abstractmethod
    def assert_valid_output(self, output) -> None:
        ...


class AbstractTest1D(AbstractTest, ABC):
    def test_2d_fail(self, simple_h2: "Histogram2D", default_kwargs):
        with pytest.raises(TypeError):
            self.method(simple_h2, **default_kwargs)

    def test_simple_does_not_fail(self, simple_h1, default_kwargs):
        _ = self.method(simple_h1, **default_kwargs)


class AbstractTest2D(AbstractTest, ABC):
    def test_1d_fail(self, simple_h1: "Histogram1D", default_kwargs):
        with pytest.raises(TypeError):
            self.method(simple_h1, **default_kwargs)

    def test_simple_does_not_fail(self, simple_h2, default_kwargs):
        _ = self.method(simple_h2, **default_kwargs)
