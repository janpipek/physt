import pytest
import numpy as np
import dask.array as da

from physt.compat.dask import h1, h2, h3


@pytest.fixture()
def huge_1d_array():
    # One extra element beyond a round number of 1000000
    array = np.arange(0, 1000001, 1)
    return da.from_array(array, chunks=(100000,))


class TestH1:
    @pytest.mark.parametrize("method", [None, "thread"])
    def test_huge(self, method, huge_1d_array):
        result = h1(huge_1d_array, "fixed_width", bin_width=100000, adaptive=True, compute=True, method=method)
        assert result.total == 1000001
        assert result.bin_right_edges[-1] == 1100000
        assert np.array_equal(result.frequencies, [100000] * 10 + [1])


class TestH2:
    @pytest.mark.parametrize("method", [None, "thread"])
    def test_huge(self, method, huge_1d_array):
        result = h2(huge_1d_array, huge_1d_array, "fixed_width", bin_width=100000, adaptive=True, compute=True, method=method)
        assert result.total == 1000001


class TestH3:
    @pytest.mark.parametrize("method", [None, "thread"])
    def test_huge(self, method, huge_1d_array):
        result = h3([huge_1d_array, huge_1d_array, huge_1d_array], "fixed_width", bin_width=100000, adaptive=True, compute=True, method=method)
        assert result.total == 1000001
