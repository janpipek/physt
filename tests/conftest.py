from typing import Tuple

import numpy as np
import pytest

from physt.binnings import FixedWidthBinning
from physt.histogram_base import HistogramBase
from physt.histogram1d import Histogram1D
from physt.histogram_nd import Histogram2D, HistogramND


@pytest.fixture
def create_adaptive():
    """Adaptive fixed-width histogram of a defined shape and some values."""
    def inner(shape: Tuple[int]) -> HistogramBase:
        binnings=[
            FixedWidthBinning(bin_width=1, bin_count=dim, bin_times_min=0 if shape[i] else None, adaptive=True)
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