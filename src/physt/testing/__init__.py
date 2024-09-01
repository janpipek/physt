"""Support testing.

More detailed comparisons of histograms mostly for unit testing.
"""
from typing import Optional

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from physt.types import HistogramBase


def assert_histograms_equal(
    left: HistogramBase,
    right: HistogramBase,
    *,
    check_dtype: bool = True,
    check_frequencies: bool = True,
    check_bins: bool = True,
    check_binnings: bool = True,
    check_metadata: bool = True,
    rtol: float = 1e-7,
    atol: float = 0,
) -> None:
    """Helper function to compare two histograms."""
    assert type(left) == type(right)
    if check_dtype:
        assert left.dtype == right.dtype
    if check_frequencies:
        assert_allclose(left.frequencies, right.frequencies, atol=atol, rtol=rtol)
    if check_bins:
        assert_allclose(left.bins, right.bins, atol=atol, rtol=rtol)
    if check_binnings:
        assert left.binnings == right.binnings
    if check_metadata:
        assert (
            left.meta_data == right.meta_data
        ), f"meta_data differ: {left.meta_data} vs {right.meta_data}"


def assert_optional_array_equal(
    actual: Optional[np.ndarray], desired: Optional[np.ndarray], **kwargs
) -> None:
    """Assert the arrays are equal or both None (helper for out tests)."""
    if actual is None:
        assert desired is None
    else:
        assert desired is not None
        assert_array_equal(actual, desired, **kwargs)
