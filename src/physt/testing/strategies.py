import hypothesis.strategies as st
import numpy as np
from hypothesis.extra.numpy import arrays, from_dtype

from physt.types import Histogram1D


@st.composite
def histograms_1d(draw):
    """Hypothesis strategy to generate 1D histograms."""

    # Turn this into a strategy itself
    bins = sorted(
        draw(st.lists(st.floats(allow_nan=False, allow_infinity=False), unique=True, min_size=2))
    )

    # TODO: What about infinities?

    values = draw(
        arrays(
            shape=len(bins) - 1,
            dtype=np.dtype("float64"),  # TODO: Negative totals with int64!
            elements=from_dtype(np.dtype("float64")).filter(lambda x: x > 0),
        )
    )

    return Histogram1D(
        binning=bins,
        frequencies=values,
        errors2=None,
        axis_name=draw(st.text()),
        title=draw(st.text()),
    )
