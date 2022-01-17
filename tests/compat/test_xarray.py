import pytest
from numpy.testing import assert_allclose

pytest.importorskip("xarray")
from xarray import Dataset

from physt.testing import assert_histograms_equal
from physt.types import Histogram1D


class TestToXarray:
    @pytest.fixture
    def simple_h1_xarray(self, simple_h1) -> Dataset:
        return simple_h1.to_xarray()

    def test_simple_h1(self, simple_h1_xarray: Dataset):
        assert isinstance(simple_h1_xarray, Dataset)

    def test_frequencies(self, simple_h1, simple_h1_xarray: Dataset):
        assert_allclose(simple_h1_xarray["frequencies"].values, simple_h1.frequencies)

    def test_errors2(self, simple_h1, simple_h1_xarray: Dataset):
        assert_allclose(simple_h1_xarray["errors2"].values, simple_h1.errors2)

    # TODO: More tests


class TestFromXarray:
    # TODO: Add tests from a constructed xarray dataset
    # TODO: Add tests for invalid datasets

    def test_to_xarray_inverse(self, simple_h1) -> Dataset:
        h1_xarray = simple_h1.to_xarray()
        h1_from_xarray = Histogram1D.from_xarray(h1_xarray)
        assert_histograms_equal(simple_h1, h1_from_xarray)
