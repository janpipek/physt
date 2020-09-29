import pytest
import numpy as np
pytest.importorskip("dask")
import dask.array as da

from physt.compat.dask import h1, h2, h3


@pytest.fixture
def huge_1d_array():
    # One extra element beyond a round number of 1000000
    array = np.arange(0, 1000001, 1)
    return da.from_array(array, chunks=(100000,))

@pytest.fixture
def huge_2d_array():
    array = np.vstack(1000 * [np.arange(0, 1000)])
    return da.from_array(array, chunks=(500, 500))


@pytest.mark.parametrize("method", [None, "thread"])
class TestH1:
    def test_huge(self, method, huge_1d_array):
        result = h1(huge_1d_array, "fixed_width", bin_width=100000, adaptive=True, compute=True, method=method)
        assert result.total == 1000001
        assert result.bin_right_edges[-1] == 1100000
        assert np.array_equal(result.frequencies, [100000] * 10 + [1])

    def test_huge_2d(self, method, huge_2d_array):
        result = h1(huge_2d_array, "fixed_width", bin_width=100, adaptive=True, compute=True, method=method)
        assert result.total == 1000000
        assert result.bin_right_edges[-1] == 1000
        assert np.array_equal(result.frequencies, [100000] * 10)

    def test_computed(self, method, huge_2d_array):
        result = h1(huge_2d_array * 2, "fixed_width", bin_width=100, adaptive=True, compute=True, method=method)
        assert result.total == 1000000
        assert result.bin_right_edges[-1] == 2000
        assert np.array_equal(result.frequencies, [50000] * 20)

@pytest.mark.parametrize("method", [None, "thread"])
class TestH2:
    def test_huge(self, method, huge_1d_array):
        result = h2(huge_1d_array, huge_1d_array, "fixed_width", bin_width=100000, adaptive=True, compute=True, method=method)
        assert result.total == 1000001


@pytest.mark.parametrize("method", [None, "thread"])
class TestH3:
    def test_huge(self, method, huge_1d_array):
        result = h3([huge_1d_array, huge_1d_array, huge_1d_array], "fixed_width", bin_width=100000, adaptive=True, compute=True, method=method)
        assert result.total == 1000001
   
