from typing import Optional

import hypothesis.strategies as st
import numpy as np
from hypothesis.extra.numpy import array_shapes, arrays, from_dtype

from physt.histogram_nd import HistogramND
from physt.types import Histogram1D

float64 = np.dtype("float64")


@st.composite
def bins(draw, nbins: Optional[int] = None) -> np.ndarray:
    if nbins is None:
        nbins = draw(st.integers(min_value=1, max_value=10))
    unsorted_edges = draw(
        arrays(
            dtype=float64,
            shape=(nbins + 1,),
            elements=from_dtype(float64, allow_nan=False, allow_infinity=False),
            unique=True,
        )
    )
    return np.sort(unsorted_edges)


@st.composite
def histograms_1d(draw):
    """Hypothesis strategy to generate 1D histograms."""

    # Turn this into a strategy itself
    bin_edges = draw(bins())

    # TODO: What about infinities?

    values = draw(
        arrays(
            shape=len(bin_edges) - 1,
            dtype=float64,  # TODO: Negative totals with int64!
            elements=from_dtype(float64, min_value=0),
        )
    )

    return Histogram1D(
        binning=bin_edges,
        frequencies=values,
        errors2=None,
        axis_name=draw(st.text()),
        title=draw(st.text()),
    )


@st.composite
def histograms_nd(draw):
    ndim = draw(st.integers(min_value=2, max_value=4))
    shape = draw(array_shapes(min_dims=ndim, max_dims=ndim, min_side=1, max_side=10))
    values = draw(
        arrays(shape=shape, dtype=float64, elements=from_dtype(float64, min_value=0))
    )
    bin_edges = [draw(bins(nbins=i)) for i in shape]

    return HistogramND(
        binnings=bin_edges,
        frequencies=values,
        axis_names=draw(st.lists(st.text(), min_size=ndim, max_size=ndim)),
    )
