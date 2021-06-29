from typing import Tuple

import numpy as np
import pytest

from physt.binnings import FixedWidthBinning
from physt.examples import normal_h2
from physt.histogram_base import HistogramBase
from physt.histogram1d import Histogram1D
from physt.histogram_nd import Histogram2D, HistogramND


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
def simple_h1() -> Histogram1D:
    edges = [0, 1, 1.5, 2, 3]
    frequencies = [1, 25, 0, 12]
    return Histogram1D(
        binning=edges,
        frequencies=frequencies,
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
