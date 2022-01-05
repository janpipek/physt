from typing import Tuple

import numpy as np
import pytest

from physt.binnings import FixedWidthBinning
from physt.examples import normal_h2
from physt.statistics import Statistics
from physt.types import HistogramBase, Histogram1D, Histogram2D, HistogramND
from physt.typing_aliases import ArrayLike


@pytest.fixture
def create_adaptive():
    """Adaptive fixed-width histogram of a defined shape and some values."""

    def inner(shape: Tuple[int]) -> HistogramBase:
        binnings = [
            FixedWidthBinning(
                bin_width=1,
                bin_count=dim,
                bin_times_min=0 if shape[i] else None,
                adaptive=True,
            )
            for i, dim in enumerate(shape)
        ]
        data = np.linspace(0, np.prod(shape) - 1, np.product(shape)).reshape(shape)
        klass = HistogramND
        if len(shape) == 2:
            klass = Histogram2D
        elif len(shape) == 1:
            return Histogram1D(binning=binnings[0], frequencies=data)
        return klass(binnings=binnings, frequencies=data)

    return inner


@pytest.fixture
def simple_edges() -> ArrayLike:
    return [0, 1, 1.5, 2, 3]


@pytest.fixture
def simple_h1(simple_edges) -> Histogram1D:
    frequencies = [1, 25, 0, 12]
    return Histogram1D(
        binning=simple_edges,
        frequencies=frequencies,
        axis_name="axis_x",
        dtype="int64",
        name="Name",
        title="Title",
        stats=Statistics(
            min=0.5,
            max=2.87,
            sum=52,
            sum2=71,
            weight=40,  # a bit over 38
        ),
    )


@pytest.fixture
def empty_h1(simple_edges) -> Histogram1D:
    return Histogram1D(
        binning=simple_edges,
        frequencies=None,
        axis_name="axis_x",
        name="Name",
        title="Title",
    )


@pytest.fixture
def simple_h2() -> Histogram2D:
    edges = [[0, 1, 2, 3], [4, 5, 6]]
    frequencies = [[1, 2], [3, 4], [5, 6]]
    return Histogram2D(
        binnings=edges, frequencies=frequencies, axis_names=["x", "y"], name="Name", title="Title"
    )
