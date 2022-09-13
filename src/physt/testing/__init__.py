"""Support testing.

More detailed comparisons of histograms mostly for unit testing.
"""

from numpy.testing import assert_allclose

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
